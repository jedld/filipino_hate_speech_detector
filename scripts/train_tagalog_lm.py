import argparse
import json
import math
import random
import sys
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    from torch.optim.lr_scheduler import LRScheduler  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for older PyTorch versions
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore


class LongInputWrapper(nn.Module):
    """Cast incoming tensors to long before delegating to the language model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != torch.long:
            inputs = inputs.to(dtype=torch.long)
        return self.model(inputs)


def _clear_module_hooks(module: nn.Module) -> None:
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()
    module._backward_hooks.clear()

try:
    from torchsummary import summary as summarize_model  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    summarize_model = None

from models.language_model import MiniTransformerLanguageModel


DEFAULT_CORPUS_CANDIDATES: Tuple[Path, ...] = (
    Path("data/tagalog_corpus/all_texts.txt"),
    Path("data/tagalog_corpus_test/all_texts.txt"),
)

DEFAULT_TOKENIZER_CANDIDATES: Tuple[Path, ...] = (
    Path("models/language_model/tokenizer"),
    Path("models/language_model_test/tokenizer"),
)

DEFAULT_COMBINED_CSV = Path("data/combined/processed/train.csv")


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


def resolve_default_corpus_file() -> Optional[Path]:
    for candidate in DEFAULT_CORPUS_CANDIDATES:
        candidate_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def resolve_default_tokenizer_dir() -> Optional[Path]:
    for candidate in DEFAULT_TOKENIZER_CANDIDATES:
        candidate_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def load_additional_combined_texts() -> Tuple[List[str], Optional[str]]:
    csv_path = DEFAULT_COMBINED_CSV if DEFAULT_COMBINED_CSV.is_absolute() else PROJECT_ROOT / DEFAULT_COMBINED_CSV
    if not csv_path.exists():
        return [], None
    try:
        df = pd.read_csv(csv_path, usecols=[0])
    except Exception:
        df = pd.read_csv(csv_path)
    if df.empty:
        return [], None
    first_col = df.columns[0]
    texts = _clean_texts(df[first_col].tolist())
    if not texts:
        return [], None
    return texts, str(csv_path)


def load_text_corpus(args: argparse.Namespace) -> TextCorpus:
    if args.local_text_file:
        path = Path(args.local_text_file)
        if not path.exists():
            raise FileNotFoundError(f"Local text file not found: {path}")
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        combined_texts, combined_source = load_additional_combined_texts()
        if combined_texts:
            texts.extend(combined_texts)
            print(f"Appended {len(combined_texts)} entries from {combined_source}")
        sources = [str(path)]
        if combined_source:
            sources.append(combined_source)
        return TextCorpus(texts=texts, source=" + ".join(sources))

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
    print(f"Loading dataset from {dataset_id} with {len(dataset)} samples...")
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
    tokenizer_path: Optional[Path],
    local_files_only: bool,
) -> PreTrainedTokenizerBase:
    resolved_path: Optional[Path] = None
    if tokenizer_path is not None:
        resolved_path = tokenizer_path if tokenizer_path.is_absolute() else PROJECT_ROOT / tokenizer_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {resolved_path}")

    if resolved_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load tokenizer. Provide --tokenizer-path pointing to a local tokenizer directory or "
                "ensure network access is available."
            ) from exc

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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_ratio: float,
) -> Optional[LRScheduler]:
    if scheduler_name == "none" or total_steps <= 0:
        return None

    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")

    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")


def train_tokenizer_from_corpus(
    texts: Sequence[str],
    vocab_size: int,
    min_frequency: int,
    limit_alphabet: int,
    lowercase: bool,
    sample_size: int,
    seed: int,
    max_length: int,
) -> Tuple[PreTrainedTokenizerBase, int]:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import Lowercase, NFKC, Sequence as NormalizerSequence
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The 'tokenizers' package is required to train a tokenizer from scratch. Install tokenizers>=0.13."
        ) from exc

    if not texts:
        raise ValueError("Cannot train tokenizer without any text samples.")

    population = range(len(texts))
    if sample_size > 0 and sample_size < len(texts):
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(population, sample_size))
    else:
        selected_indices = population

    def iterator() -> Iterable[str]:
        for idx in selected_indices:
            text = texts[idx]
            if text:
                yield text

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    normalizers = [NFKC()]
    if lowercase:
        normalizers.append(Lowercase())
    tokenizer.normalizer = normalizers[0] if len(normalizers) == 1 else NormalizerSequence(normalizers)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer_kwargs = {
        "vocab_size": vocab_size,
        "min_frequency": max(1, min_frequency),
        "special_tokens": ["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        "initial_alphabet": ByteLevel.alphabet(),
    }
    if limit_alphabet > 0:
        trainer_kwargs["limit_alphabet"] = limit_alphabet

    trainer = BpeTrainer(**trainer_kwargs)
    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=len(selected_indices))

    if tokenizer.token_to_id("<s>") is None or tokenizer.token_to_id("</s>") is None:
        raise ValueError("Tokenizer training failed to include required special tokens <s> and </s>.")

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    fast_tokenizer.model_max_length = max_length
    fast_tokenizer.init_kwargs["model_max_length"] = max_length
    fast_tokenizer.padding_side = "right"
    fast_tokenizer.truncation_side = "right"
    return fast_tokenizer, len(selected_indices)


def save_tokenizer_artifacts(
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir, legacy_format=False)
    if metadata:
        metadata_path = save_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
    scheduler: Optional[LRScheduler],
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
        if scheduler is not None:
            scheduler.step()
        loss_value = loss.detach().float().item()
        step_tokens = targets.numel()
        total_loss += loss_value * step_tokens
        total_tokens += step_tokens

        if log_interval and step % log_interval == 0:
            running_avg = total_loss / max(total_tokens, 1)
            running_ppl = math.exp(min(running_avg, 20.0))
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Step {step}/{len(loader)} - Loss: {running_avg:.4f} | PPL: {running_ppl:.2f} | LR: {current_lr:.6e}"
            )

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
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=("none", "linear", "cosine"),
        default="cosine",
        help="Learning rate schedule to use during training.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for linear warmup before decaying the LR.",
    )
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
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Load tokenizer configuration from a local folder (avoids network download).",
    )
    parser.add_argument(
        "--train-new-tokenizer",
        action="store_true",
        help="Train a fresh byte-level BPE tokenizer on the training corpus before model training.",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=32_000,
        help="Vocabulary size when training a new tokenizer (only used with --train-new-tokenizer).",
    )
    parser.add_argument(
        "--tokenizer-min-frequency",
        type=int,
        default=2,
        help="Minimum token occurrence for inclusion when training a new tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-limit-alphabet",
        type=int,
        default=0,
        help="Limit alphabet for tokenizer training (0 disables the limit).",
    )
    parser.add_argument(
        "--tokenizer-lowercase",
        action="store_true",
        help="Lowercase text during tokenizer training for case-insensitive models.",
    )
    parser.add_argument(
        "--tokenizer-sample-size",
        type=int,
        default=0,
        help="Sample size for tokenizer training to reduce compute; 0 uses the full corpus.",
    )
    parser.add_argument("--bos-token", type=str, default=None, help="Override BOS token text if undefined.")
    parser.add_argument("--eos-token", type=str, default=None, help="Override EOS token text if undefined.")
    parser.add_argument("--pad-token", type=str, default=None, help="Override PAD token text if undefined.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Prevent remote downloads when loading tokenizer assets (requires local artifacts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.local_text_file and args.local_csv:
        raise ValueError("Specify at most one of --local-text-file or --local-csv")

    if args.local_text_file is None and args.local_csv is None:
        default_corpus = resolve_default_corpus_file()
        if default_corpus is not None:
            args.local_text_file = default_corpus
            print(f"Defaulting to local text corpus: {default_corpus}")

    if args.train_new_tokenizer and args.tokenizer_path is not None:
        print("Ignoring --tokenizer-path because --train-new-tokenizer is enabled.")
        args.tokenizer_path = None

    if not args.train_new_tokenizer and args.tokenizer_path is None:
        default_tokenizer_dir = resolve_default_tokenizer_dir()
        if default_tokenizer_dir is not None:
            args.tokenizer_path = default_tokenizer_dir
            print(f"Defaulting to local tokenizer: {default_tokenizer_dir}")

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = load_text_corpus(args)
    tokenizer_sampled = len(corpus.texts)
    tokenizer_target_max_length = max(args.max_position_embeddings, args.block_size + 1, args.block_size * 4)
    if args.train_new_tokenizer:
        print("Training new tokenizer from corpus...")
        tokenizer, tokenizer_sampled = train_tokenizer_from_corpus(
            texts=corpus.texts,
            vocab_size=args.tokenizer_vocab_size,
            min_frequency=args.tokenizer_min_frequency,
            limit_alphabet=args.tokenizer_limit_alphabet,
            lowercase=args.tokenizer_lowercase,
            sample_size=args.tokenizer_sample_size,
            seed=args.seed,
            max_length=tokenizer_target_max_length,
        )
        tokenizer_identifier = "custom-trained"
        print(
            "Finished tokenizer training | vocab size: "
            f"{tokenizer_vocab_size(tokenizer)} | samples used: {tokenizer_sampled}"
        )
    else:
        tokenizer = load_lm_tokenizer(
            tokenizer_name=args.tokenizer_name,
            cache_dir=args.tokenizer_cache,
            bos_token=args.bos_token,
            eos_token=args.eos_token,
            pad_token=args.pad_token,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.offline,
        )
        tokenizer_identifier = args.tokenizer_name

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

    total_train_steps = len(train_loader) * args.epochs

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
    # Warmup + decay scheduling is a robust default for compact language models.
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.lr_scheduler,
        total_steps=total_train_steps,
        warmup_ratio=args.warmup_ratio,
    )

    history: List[dict] = []
    best_train_perplexity = float("inf")
    best_state: Optional[dict] = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_save_path = args.output_dir / "tokenizer"
    checkpoint_path = args.output_dir / "tagalog_lm.pt"
    sample_path = args.output_dir / "sample.txt"
    metrics_path = args.output_dir / "language_model_metrics.json"

    print(f"Loaded {len(corpus.texts)} documents from {corpus.source}")
    print(f"Vocabulary size: {tokenizer_vocab}")
    print(f"Train sequences: {len(train_dataset)}, validation sequences: {len(val_dataset)}")
    print(f"Training on device: {device}")

    if summarize_model is not None:
        try:
            print("\nModel summary (via torchsummary):")
            summary_model = LongInputWrapper(model).to(device)
            summarize_model(
                summary_model,
                input_size=(args.block_size,),
                batch_size=args.batch_size,
                device=device.type,
            )
        except Exception as exc:
            print(f"Unable to generate model summary: {exc}")
            summary_model.apply(_clear_module_hooks)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                "Fallback parameter stats - total params: "
                f"{total_params:,}, trainable params: {trainable_params:,}"
            )
    else:
        print("torchsummary not available; install dependency to view model summary.")
    
    tokenizer_metadata: Dict[str, object] = {
        "name": tokenizer_identifier,
        "vocab_size": tokenizer_vocab,
        "trained_from_scratch": args.train_new_tokenizer,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "byte_level_decoder": True if args.train_new_tokenizer else False,
    }

    save_tokenizer_artifacts(tokenizer, tokenizer_save_path, tokenizer_metadata)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.grad_clip,
            scheduler,
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
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
        )

        if train_ppl < best_train_perplexity:
            best_train_perplexity = train_ppl
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_perplexity": train_ppl,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "tokenizer_path": str(tokenizer_save_path),
                "config": {
                    "vocab_size": tokenizer_vocab,
                    "tokenizer_name": tokenizer_identifier,
                    "tokenizer_trained_from_scratch": args.train_new_tokenizer,
                    "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
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
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "tokenizer_training_params": {
                    "vocab_size": args.tokenizer_vocab_size,
                    "min_frequency": args.tokenizer_min_frequency,
                    "limit_alphabet": args.tokenizer_limit_alphabet,
                    "lowercase": args.tokenizer_lowercase,
                    "sample_size": tokenizer_sampled,
                }
                if args.train_new_tokenizer
                else None,
            }
            torch.save(best_state, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

    if best_state is None:
        best_state = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_perplexity": train_ppl,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "tokenizer_path": str(tokenizer_save_path),
            "config": {
                "vocab_size": tokenizer_vocab,
                "tokenizer_name": tokenizer_identifier,
                "tokenizer_trained_from_scratch": args.train_new_tokenizer,
                "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
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
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "tokenizer_training_params": {
                "vocab_size": args.tokenizer_vocab_size,
                "min_frequency": args.tokenizer_min_frequency,
                "limit_alphabet": args.tokenizer_limit_alphabet,
                "lowercase": args.tokenizer_lowercase,
                "sample_size": tokenizer_sampled,
            }
            if args.train_new_tokenizer
            else None,
        }
        torch.save(best_state, checkpoint_path)

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()


    if args.train_new_tokenizer:
        tokenizer_metadata.update(
            {
                "min_frequency": args.tokenizer_min_frequency,
                "limit_alphabet": args.tokenizer_limit_alphabet,
                "lowercase": args.tokenizer_lowercase,
                "sample_size": tokenizer_sampled,
            }
        )

    save_tokenizer_artifacts(tokenizer, tokenizer_save_path, tokenizer_metadata)

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
        "best_train_loss": best_state["train_loss"],
        "best_train_perplexity": best_state["train_perplexity"],
        "best_val_loss": best_state["val_loss"],
        "best_val_perplexity": best_state["val_perplexity"],
        "corpus_source": corpus.source,
        "tokenizer": {"path": str(tokenizer_save_path), **tokenizer_metadata},
        "scheduler": {
            "type": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "total_steps": total_train_steps,
            "warmup_steps": int(total_train_steps * args.warmup_ratio),
        },
        "checkpoint_path": str(checkpoint_path),
        "sample_path": str(sample_path),
        "config": best_state["config"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Tokenizer saved to {tokenizer_save_path}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Sample text written to {sample_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
