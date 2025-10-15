import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models.language_model import MiniTransformerLanguageModel


@dataclass
class TextCorpus:
    texts: List[str]
    source: str


class CharacterSequenceDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
        stride: int,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if stride < 1:
            raise ValueError("stride must be at least 1")

        token_sequence: List[int] = []
        for text in texts:
            text = str(text)
            if not text.strip():
                continue
            encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if not encoded:
                continue
            token_sequence.extend(encoded)

        if len(token_sequence) < block_size + 1:
            raise ValueError(
                "Not enough tokens to create sequences. Try reducing block_size or increasing the number of samples."
            )

        self.tokens = torch.tensor(token_sequence, dtype=torch.long)
        self.block_size = block_size
        self.stride = stride
        max_start = len(self.tokens) - (block_size + 1)
        if max_start < 0:
            raise ValueError("Not enough tokens to create input-target pairs for the requested block size.")
        starts = list(range(0, max_start + 1, stride))
        if not starts or starts[-1] != max_start:
            starts.append(max_start)
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean_texts(texts: Iterable[str]) -> List[str]:
    return [str(text).strip() for text in texts if isinstance(text, str) and str(text).strip()]


def load_text_corpus(args: argparse.Namespace) -> TextCorpus:
    if args.local_text_file:
        path = Path(args.local_text_file)
        if not path.exists():
            raise FileNotFoundError(f"Local text file not found: {path}")
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return TextCorpus(texts=texts, source=str(path))

    if args.local_csv:
        path = Path(args.local_csv)
        if not path.exists():
            raise FileNotFoundError(f"Local CSV file not found: {path}")
        df = pd.read_csv(path)
        if args.csv_text_column not in df.columns:
            raise ValueError(f"CSV file {path} does not contain column '{args.csv_text_column}'")
        texts = _clean_texts(df[args.csv_text_column].tolist())
        return TextCorpus(texts=texts, source=str(path))

    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        dataset_id = f"{args.dataset_name}/{args.dataset_config}:{args.split}"
    else:
        dataset = load_dataset(args.dataset_name, split=args.split)
        dataset_id = f"{args.dataset_name}:{args.split}"

    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{args.text_column}' not found in dataset. Available columns: {dataset.column_names}"
        )

    total_length = len(dataset)
    if args.max_samples and args.max_samples < total_length:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))
    elif args.shuffle_before_split:
        dataset = dataset.shuffle(seed=args.seed)

    texts = _clean_texts(dataset[args.text_column])
    if not texts:
        raise ValueError("Dataset did not yield any non-empty text entries.")

    return TextCorpus(texts=texts, source=dataset_id)


def load_lm_tokenizer(
    tokenizer_name: str,
    cache_dir: Optional[Path],
    bos_token: Optional[str],
    eos_token: Optional[str],
    pad_token: Optional[str],
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(cache_dir) if cache_dir else None)

    def ensure_token(attribute: str, candidate: Optional[str], fallback_attr: Optional[str], default: str) -> None:
        current = getattr(tokenizer, attribute)
        if current:
            return
        if candidate:
            tokenizer.add_special_tokens({attribute: candidate})
            return
        if fallback_attr:
            fallback_value = getattr(tokenizer, fallback_attr)
            if fallback_value:
                setattr(tokenizer, attribute, fallback_value)
                return
        tokenizer.add_special_tokens({attribute: default})

    ensure_token("bos_token", bos_token, "cls_token", "<s>")
    ensure_token("eos_token", eos_token, "sep_token", "</s>")
    ensure_token("pad_token", pad_token, "eos_token", "<pad>")

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS and EOS token ids after configuration.")

    return tokenizer


def tokenizer_vocab_size(tokenizer: PreTrainedTokenizerBase) -> int:
    try:
        return len(tokenizer)
    except TypeError:
        return len(tokenizer.get_vocab())


def create_datasets(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    stride: int,
    val_ratio: float,
    seed: int,
) -> Tuple[CharacterSequenceDataset, CharacterSequenceDataset]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    texts = list(texts)
    random.Random(seed).shuffle(texts)
    val_count = max(1, int(len(texts) * val_ratio))
    if val_count >= len(texts):
        val_count = min(len(texts) // 5, len(texts) - 1) or 1
    train_texts = texts[:-val_count]
    val_texts = texts[-val_count:]
    if not train_texts:
        raise ValueError("Training texts are empty after splitting. Reduce val_ratio or provide more data.")

    train_dataset = CharacterSequenceDataset(train_texts, tokenizer, block_size, stride)
    val_dataset = CharacterSequenceDataset(val_texts, tokenizer, block_size, stride)
    return train_dataset, val_dataset


def train_one_epoch(
    model: MiniTransformerLanguageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
    log_interval: Optional[int] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(inputs)
            loss = criterion(logits.view(-1, model.vocab_size), targets.reshape(-1))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        loss_value = loss.detach().float().item()
        step_tokens = targets.numel()
        total_loss += loss_value * step_tokens
        total_tokens += step_tokens

        if log_interval and step % log_interval == 0:
            running_avg = total_loss / max(total_tokens, 1)
            running_ppl = math.exp(min(running_avg, 20.0))
            print(f"  Step {step}/{len(loader)} - Loss: {running_avg:.4f} | PPL: {running_ppl:.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


@torch.no_grad()
def evaluate(
    model: MiniTransformerLanguageModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits.view(-1, model.vocab_size), targets.reshape(-1))
        loss_value = loss.detach().float().item()
        total_loss += loss_value * targets.numel()
        total_tokens += targets.numel()
    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


def sample_text(
    model: MiniTransformerLanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
    length: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.bos_token_id]
    else:
        prompt_tokens = [tokenizer.bos_token_id] + prompt_tokens
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(input_ids, max_new_tokens=length, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Tagalog Transformer language model")
    parser.add_argument("--dataset-name", type=str, default="oscar", help="Hugging Face dataset name")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="unshuffled_deduplicated_tl",
        help="Dataset configuration (e.g., language subset)",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--text-column", type=str, default="text", help="Column containing raw text")
    parser.add_argument("--max-samples", type=int, default=100_000, help="Maximum number of texts to use")
    parser.add_argument("--shuffle-before-split", action="store_true", help="Shuffle dataset before splitting")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--block-size", type=int, default=256, help="Sequence length for training")
    parser.add_argument("--stride", type=int, default=128, help="Stride for sliding window over token stream")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("models/language_model"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-prompt", type=str, default="Mabuhay ang Pilipinas!")
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=20)
    parser.add_argument("--local-text-file", type=Path, default=None, help="Use lines from a local UTF-8 text file")
    parser.add_argument("--local-csv", type=Path, default=None, help="Use a text column from a local CSV file")
    parser.add_argument("--csv-text-column", type=str, default="text")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training logs")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu or cuda)")
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="jcblaise/roberta-tagalog-base",
        help="Hugging Face tokenizer identifier to use for text encoding.",
    )
    parser.add_argument(
        "--tokenizer-cache",
        type=Path,
        default=None,
        help="Optional cache directory for tokenizer files.",
    )
    parser.add_argument("--bos-token", type=str, default=None, help="Override BOS token text if undefined.")
    parser.add_argument("--eos-token", type=str, default=None, help="Override EOS token text if undefined.")
    parser.add_argument("--pad-token", type=str, default=None, help="Override PAD token text if undefined.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.local_text_file and args.local_csv:
        raise ValueError("Specify at most one of --local-text-file or --local-csv")

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = load_text_corpus(args)
    tokenizer = load_lm_tokenizer(
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.tokenizer_cache,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
    )

    tokenizer_vocab = tokenizer_vocab_size(tokenizer)
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        tokenizer_max_length
        and isinstance(tokenizer_max_length, int)
        and tokenizer_max_length > 0
        and tokenizer_max_length != int(1e30)
        and tokenizer_max_length < args.block_size + 1
    ):
        raise ValueError(
            f"Token sequence length {args.block_size + 1} exceeds tokenizer model_max_length {tokenizer_max_length}. "
            "Reduce --block-size or provide a tokenizer with a larger maximum length."
        )

    train_dataset, val_dataset = create_datasets(
        corpus.texts,
        tokenizer,
        block_size=args.block_size,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Adjust block size, stride, or provide more data.")
    if args.max_position_embeddings < args.block_size:
        raise ValueError("max_position_embeddings must be greater than or equal to block_size")

    model = MiniTransformerLanguageModel(
        vocab_size=tokenizer_vocab,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_position_embeddings=args.max_position_embeddings,
        dropout=args.dropout,
        ff_multiplier=args.ff_multiplier,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_autocast = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=use_autocast) if use_autocast else None

    history: List[dict] = []
    best_val_loss = float("inf")
    best_state: Optional[dict] = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer"
    checkpoint_path = args.output_dir / "tagalog_lm.pt"
    sample_path = args.output_dir / "sample.txt"
    metrics_path = args.output_dir / "language_model_metrics.json"

    print(f"Loaded {len(corpus.texts)} documents from {corpus.source}")
    print(f"Vocabulary size: {tokenizer_vocab}")
    print(f"Train sequences: {len(train_dataset)}, validation sequences: {len(val_dataset)}")
    print(f"Training on device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.grad_clip,
            args.log_interval,
        )
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_perplexity": train_ppl,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "tokenizer_path": str(tokenizer_path),
                "config": {
                    "vocab_size": tokenizer_vocab,
                    "tokenizer_name": args.tokenizer_name,
                    "embed_dim": args.embed_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "ff_multiplier": args.ff_multiplier,
                    "dropout": args.dropout,
                    "block_size": args.block_size,
                    "max_position_embeddings": args.max_position_embeddings,
                    "stride": args.stride,
                },
                "training_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            }
            torch.save(best_state, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

    if best_state is None:
        best_state = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "tokenizer_path": str(tokenizer_path),
            "config": {
                "vocab_size": tokenizer_vocab,
                "tokenizer_name": args.tokenizer_name,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "ff_multiplier": args.ff_multiplier,
                "dropout": args.dropout,
                "block_size": args.block_size,
                "max_position_embeddings": args.max_position_embeddings,
                "stride": args.stride,
            },
            "training_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    tokenizer.save_pretrained(tokenizer_path)

    generated_text = sample_text(
        model,
        tokenizer,
        prompt=args.sample_prompt,
        device=device,
        length=args.sample_length,
        temperature=args.sample_temperature,
        top_k=args.sample_top_k,
    )
    sample_path.write_text(generated_text, encoding="utf-8")

    summary = {
        "history": history,
        "best_epoch": best_state["epoch"],
        "best_val_loss": best_state["val_loss"],
        "best_val_perplexity": best_state["val_perplexity"],
        "corpus_source": corpus.source,
        "tokenizer": {
            "path": str(tokenizer_path),
            "name": args.tokenizer_name,
            "vocab_size": tokenizer_vocab,
        },
        "checkpoint_path": str(checkpoint_path),
        "sample_path": str(sample_path),
        "config": best_state["config"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Sample text written to {sample_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
