import argparse
import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models.language_model import MiniTransformerLanguageModel
from models.tagalog_lm_classifier import TagalogLMClassifier, TagalogLMClassifierConfig


class TextClassificationDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]):
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if it doesn't exist as-is."""
    if path.is_absolute() or path.exists():
        return path
    resolved = PROJECT_ROOT / path
    return resolved if resolved.exists() else path


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}, but it was not found.")


def load_dataframe(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"File {path} must contain '{text_column}' and '{label_column}' columns.")
    return df[[text_column, label_column]].rename(columns={text_column: "text", label_column: "label"})


def _candidate_tokenizer_paths(identifier: str, checkpoint_path: Path) -> List[Path]:
    raw_path = Path(identifier)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(raw_path)
        candidates.append(PROJECT_ROOT / raw_path)
        candidates.append(checkpoint_path.parent / raw_path)

    seen: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            seen.append(resolved)
    return seen


def load_language_model_tokenizer(
    lm_state: Dict[str, object],
    checkpoint_path: Path,
    fallback_path: Path,
) -> Tuple[PreTrainedTokenizerBase, Optional[Path], str]:
    candidate_identifiers: List[str] = []
    state_tokenizer_path = lm_state.get("tokenizer_path")
    if isinstance(state_tokenizer_path, str) and state_tokenizer_path:
        candidate_identifiers.append(state_tokenizer_path)
    candidate_identifiers.append(str(fallback_path))

    tried: List[str] = []
    seen_raw: set[str] = set()
    for identifier in candidate_identifiers:
        if identifier in seen_raw:
            continue
        seen_raw.add(identifier)

        for path in _candidate_tokenizer_paths(identifier, checkpoint_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
                return tokenizer, path, identifier
            except Exception as exc:  # pragma: no cover - runtime resolution
                tried.append(f"{path}: {exc}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(identifier)
            local_paths = _candidate_tokenizer_paths(identifier, checkpoint_path)
            source_dir = local_paths[0] if local_paths else None
            return tokenizer, source_dir, identifier
        except Exception as exc:  # pragma: no cover - runtime resolution
            tried.append(f"{identifier}: {exc}")

    attempted = "; ".join(tried) if tried else ", ".join(candidate_identifiers)
    raise RuntimeError(f"Failed to load tokenizer from provided options. Attempts: {attempted}")


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = metric_list[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_list])) for key in keys}


def collate_batch(tokenizer, max_length: int, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts, labels = zip(*batch)
    encoding = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return encoding["input_ids"], encoding["attention_mask"], label_tensor


def train_one_epoch(
    model: TagalogLMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    grad_clip: float,
    use_autocast: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics_accumulator: List[Dict[str, float]] = []

    for input_ids, attention_mask, targets in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_autocast):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, targets)

        if scaler is not None and use_autocast:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        metrics_accumulator.append(compute_metrics(logits.detach().cpu(), targets.detach().cpu()))

    avg_loss = total_loss / max(len(loader), 1)
    avg_metrics = aggregate_metrics(metrics_accumulator)
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(
    model: TagalogLMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics_accumulator: List[Dict[str, float]] = []

    for input_ids, attention_mask, targets in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        metrics_accumulator.append(compute_metrics(logits.cpu(), targets.cpu()))

    avg_loss = total_loss / max(len(loader), 1)
    avg_metrics = aggregate_metrics(metrics_accumulator)
    return avg_loss, avg_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hate-speech classifier on top of the Tagalog LM backbone")
    parser.add_argument("--train-csv", type=Path, default=Path("data/combined/processed/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/combined/processed/validation.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/combined/processed/test.csv"))
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--lm-checkpoint", type=Path, default=Path("models/language_model_test/tagalog_lm.pt"))
    parser.add_argument("--lm-tokenizer", type=Path, default=Path("models/language_model_test/tokenizer"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/tagalog_classifier"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, choices=("mean", "last"), default="mean")
    parser.add_argument("--fine-tune-base", action="store_true", help="Allow gradients to update the LM backbone.")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/tagalog_classifier"))
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def build_backbone(lm_state: Dict[str, object], device: torch.device) -> Tuple[MiniTransformerLanguageModel, Dict[str, object]]:
    lm_config = lm_state.get("config")
    if lm_config is None:
        raise ValueError("Language-model checkpoint does not contain configuration metadata.")

    required_keys = {"vocab_size", "embed_dim", "num_heads", "num_layers", "ff_multiplier", "max_position_embeddings"}
    missing_keys = required_keys.difference(lm_config.keys())
    if missing_keys:
        raise ValueError(f"Language-model config missing keys: {missing_keys}")

    language_model = MiniTransformerLanguageModel(
        vocab_size=int(lm_config["vocab_size"]),
        embed_dim=int(lm_config["embed_dim"]),
        num_heads=int(lm_config["num_heads"]),
        num_layers=int(lm_config["num_layers"]),
        max_position_embeddings=int(lm_config["max_position_embeddings"]),
        dropout=float(lm_config.get("dropout", 0.1)),
        ff_multiplier=int(lm_config["ff_multiplier"]),
    )
    language_model.load_state_dict(lm_state["model_state_dict"], strict=True)
    language_model.to(device)
    return language_model, lm_config


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve all paths relative to PROJECT_ROOT if needed
    args.train_csv = resolve_path(args.train_csv)
    args.val_csv = resolve_path(args.val_csv)
    args.test_csv = resolve_path(args.test_csv)
    args.lm_checkpoint = resolve_path(args.lm_checkpoint)
    args.lm_tokenizer = resolve_path(args.lm_tokenizer)
    args.output_dir = resolve_path(args.output_dir) if not args.output_dir.is_absolute() else args.output_dir

    ensure_file(args.train_csv, "training CSV")
    ensure_file(args.val_csv, "validation CSV")
    ensure_file(args.test_csv, "test CSV")
    ensure_file(args.lm_checkpoint, "language-model checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_save_path = args.output_dir / "tokenizer"
    model_path = args.output_dir / "tagalog_classifier.pt"
    last_model_path = args.output_dir / "tagalog_classifier_last.pt"
    metrics_path = args.output_dir / "metrics.json"

    lm_state = torch.load(args.lm_checkpoint, map_location="cpu")
    language_model, lm_config = build_backbone(lm_state, device)

    tokenizer, tokenizer_source_dir, tokenizer_source_identifier = load_language_model_tokenizer(
        lm_state, args.lm_checkpoint, args.lm_tokenizer
    )
    tokenizer.model_max_length = args.max_length

    if tokenizer_save_path.exists():
        shutil.rmtree(tokenizer_save_path)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path, legacy_format=False)

    tokenizer_metadata: Optional[Dict[str, object]] = None
    if tokenizer_source_dir is not None:
        metadata_file = tokenizer_source_dir / "training_metadata.json"
        if metadata_file.exists():
            tokenizer_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            shutil.copy2(metadata_file, tokenizer_save_path / metadata_file.name)

    tokenizer_source_display = (
        str(tokenizer_source_dir) if tokenizer_source_dir is not None else tokenizer_source_identifier
    )
    print(f"Using tokenizer from {tokenizer_source_display}")

    max_position_embeddings = int(lm_config["max_position_embeddings"])
    if args.max_length > max_position_embeddings:
        raise ValueError(
            f"Requested max_length {args.max_length} exceeds language model limit {max_position_embeddings}."
        )

    train_df = load_dataframe(args.train_csv, args.text_column, args.label_column)
    val_df = load_dataframe(args.val_csv, args.text_column, args.label_column)
    test_df = load_dataframe(args.test_csv, args.text_column, args.label_column)

    num_labels = len(sorted(set(train_df["label"].astype(int).tolist())))
    if num_labels < 2:
        raise ValueError("Classification task requires at least two distinct labels.")

    classifier_config = TagalogLMClassifierConfig(
        num_labels=num_labels,
        hidden_dim=args.classifier_hidden_dim,
        classifier_dropout=args.classifier_dropout,
        pooling=args.pooling,
        fine_tune_base=args.fine_tune_base,
    )
    model = TagalogLMClassifier(language_model, classifier_config).to(device)

    train_dataset = TextClassificationDataset(train_df["text"].tolist(), train_df["label"].tolist())
    val_dataset = TextClassificationDataset(val_df["text"].tolist(), val_df["label"].tolist())
    test_dataset = TextClassificationDataset(test_df["text"].tolist(), test_df["label"].tolist())

    collate_fn = lambda batch: collate_batch(tokenizer, args.max_length, batch)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=args.mixed_precision and device.type == "cuda")
    use_autocast = args.mixed_precision and device.type == "cuda"

    history: List[Dict[str, Dict[str, float]]] = []
    best_val_f1 = float("-inf")
    best_state: Optional[Dict[str, object]] = None
    patience_counter = 0

    writer: Optional[SummaryWriter] = None
    run_log_dir: Optional[Path] = None
    if not args.no_tensorboard:
        run_log_dir = args.log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_log_dir))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.grad_clip,
            use_autocast,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            for metric, value in train_metrics.items():
                writer.add_scalar(f"Metrics/train/{metric}", value, epoch)
            for metric, value in val_metrics.items():
                writer.add_scalar(f"Metrics/validation/{metric}", value, epoch)

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        improved = val_metrics.get("f1", 0.0) > best_val_f1 + args.min_delta
        if improved:
            best_val_f1 = val_metrics.get("f1", 0.0)
            patience_counter = 0
            best_state = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "model_type": "tagalog_lm_classifier",
                "classifier_config": model.get_classifier_config(),
                "language_model_config": lm_config,
                "training_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "tokenizer_path": str(tokenizer_save_path),
                "tokenizer_source": tokenizer_source_display,
                "tokenizer_source_identifier": tokenizer_source_identifier,
                "tokenizer_metadata": tokenizer_metadata,
                "language_model_checkpoint": str(args.lm_checkpoint),
            }
            torch.save(best_state, model_path)
            print(f"New best model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter > args.patience:
                print("Early stopping triggered.")
                break

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            last_model_path,
        )

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs,
            "val_loss": history[-1]["val_loss"] if history else float("inf"),
            "val_metrics": history[-1]["val_metrics"] if history else {},
            "model_type": "tagalog_lm_classifier",
            "classifier_config": model.get_classifier_config(),
            "language_model_config": lm_config,
            "training_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "tokenizer_path": str(tokenizer_save_path),
            "tokenizer_source": tokenizer_source_display,
            "tokenizer_source_identifier": tokenizer_source_identifier,
            "tokenizer_metadata": tokenizer_metadata,
            "language_model_checkpoint": str(args.lm_checkpoint),
        }
        torch.save(best_state, model_path)

    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f}"
    )

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if writer is not None:
        writer.add_text("config/json", json.dumps(cfg, indent=2))
        writer.add_scalar("Loss/test", test_loss, best_state["epoch"])
        for metric_name, metric_value in test_metrics.items():
            writer.add_scalar(f"Metrics/test/{metric_name}", metric_value, best_state["epoch"])
        writer.flush()
        writer.close()

    tokenizer_summary: Dict[str, object] = {
        "path": str(tokenizer_save_path),
        "type": "huggingface",
        "source": str(tokenizer_source_dir) if tokenizer_source_dir is not None else tokenizer_source_identifier,
        "source_identifier": tokenizer_source_identifier,
    }
    if tokenizer_source_dir is not None:
        tokenizer_summary["source_path"] = str(tokenizer_source_dir)
    if tokenizer_metadata is not None:
        tokenizer_summary["metadata"] = tokenizer_metadata

    summary = {
        "history": history,
        "best_epoch": best_state["epoch"],
        "best_val_loss": best_state["val_loss"],
        "best_val_metrics": best_state["val_metrics"],
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "config": cfg,
        "tokenizer": tokenizer_summary,
        "artifacts": {
            "best_model": str(model_path),
            "last_checkpoint": str(last_model_path),
            "tokenizer": str(tokenizer_save_path),
        },
    }

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Model saved to {model_path}")
    print(f"Last checkpoint saved to {last_model_path}")
    print(f"Tokenizer saved to {tokenizer_save_path}")
    print(f"Metrics saved to {metrics_path}")
    if run_log_dir is not None:
        print(f"TensorBoard logs written to {run_log_dir}")


if __name__ == "__main__":
    main()
