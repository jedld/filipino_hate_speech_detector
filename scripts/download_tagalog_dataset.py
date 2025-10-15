import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download


def _clean_texts(texts: Iterable[str], min_chars: int) -> List[str]:
    cleaned = []
    for text in texts:
        if not isinstance(text, str):
            continue
        value = text.strip()
        if len(value) >= min_chars:
            cleaned.append(value)
    return cleaned


def _read_text_file(path: Path, min_chars: int) -> List[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1", errors="ignore")
    texts = [line.strip() for line in content.splitlines() if line.strip()]
    return [text for text in texts if len(text) >= min_chars]


def _read_csv_file(path: Path, min_chars: int, preferred_column: Optional[str]) -> List[str]:
    df = pd.read_csv(path)
    columns: Sequence[str]
    if preferred_column and preferred_column in df.columns:
        columns = [preferred_column]
    else:
        columns = [c for c in df.columns if df[c].dtype == "object"]
    texts: List[str] = []
    for column in columns:
        texts.extend(_clean_texts(df[column].tolist(), min_chars))
    return texts


def _read_jsonl_file(path: Path, min_chars: int, preferred_key: Optional[str]) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if preferred_key and preferred_key in payload:
                value = payload[preferred_key]
                if isinstance(value, str) and len(value.strip()) >= min_chars:
                    texts.append(value.strip())
            else:
                for value in payload.values():
                    if isinstance(value, str) and len(value.strip()) >= min_chars:
                        texts.append(value.strip())
    return texts


def collect_repo_texts(repo_dir: Path, min_chars: int, preferred_field: Optional[str]) -> List[str]:
    supported_suffixes = {".txt", ".csv", ".jsonl"}
    texts: List[str] = []
    for file_path in repo_dir.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in supported_suffixes:
            continue
        if suffix == ".txt":
            texts.extend(_read_text_file(file_path, min_chars))
        elif suffix == ".csv":
            texts.extend(_read_csv_file(file_path, min_chars, preferred_field))
        elif suffix == ".jsonl":
            texts.extend(_read_jsonl_file(file_path, min_chars, preferred_field))
    return texts


def download_repo(repo_id: str, cache_dir: Path, allow_patterns: Sequence[str]) -> Path:
    local_path = snapshot_download(
        repo_id,
        repo_type="model",
        cache_dir=str(cache_dir),
        allow_patterns=list(allow_patterns),
    )
    return Path(local_path)


def _resolve_text_column(dataset, preferred_column: Optional[str]) -> str:
    if preferred_column and preferred_column in dataset.column_names:
        return preferred_column
    for column, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype in {"string", "large_string"}:
            return column
    for column in dataset.column_names:
        if dataset[column] and isinstance(dataset[column][0], str):
            return column
    raise ValueError(
        "No string-like column was found in the fallback dataset. Specify --fallback-text-column explicitly."
    )


def load_fallback_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: Optional[str],
    max_samples: Optional[int],
    seed: int,
    min_chars: int,
) -> List[str]:
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    resolved_column = _resolve_text_column(dataset, text_column)

    if max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(min(len(dataset), max_samples)))

    texts = _clean_texts(dataset[resolved_column], min_chars)
    if not texts:
        raise ValueError(
            "Fallback dataset did not produce any text samples. "
            f"Check column '{resolved_column}' for dataset '{dataset_name}'."
        )
    return texts


def save_splits(texts: Sequence[str], output_dir: Path, seed: int, val_ratio: float) -> None:
    if not texts:
        raise ValueError("No texts available to save.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    texts = list(dict.fromkeys(texts))
    rng.shuffle(texts)
    val_count = max(1, int(len(texts) * val_ratio)) if val_ratio > 0 else 0
    val_count = min(val_count, len(texts) - 1) if len(texts) > 1 else 0
    val_texts = texts[:val_count]
    train_texts = texts[val_count:]
    pd.DataFrame({"text": train_texts}).to_csv(output_dir / "train.csv", index=False)
    if val_texts:
        pd.DataFrame({"text": val_texts}).to_csv(output_dir / "val.csv", index=False)
    Path(output_dir / "all_texts.txt").write_text("\n".join(texts), encoding="utf-8")


def dump_metadata(
    output_dir: Path,
    repo_id: str,
    source_texts: int,
    fallback_used: bool,
    fallback_name: Optional[str],
    total_samples: int,
) -> None:
    metadata = {
        "repo_id": repo_id,
        "source_texts": source_texts,
        "fallback_used": fallback_used,
        "fallback_dataset": fallback_name,
        "total_samples": total_samples,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download textual resources from a Hugging Face model repository and prepare a corpus for the Tagalog "
            "language model trainer."
        )
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="jcblaise/bert-tagalog-base-uncased",
        help="Hugging Face repository identifier.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/hf_repos"),
        help="Cache directory for downloaded repositories.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=("*.txt", "*.csv", "*.jsonl"),
        help="Glob patterns of files to download from the repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tagalog_corpus"),
        help="Directory where prepared corpus files will be stored.",
    )
    parser.add_argument("--min-chars", type=int, default=15, help="Minimum character length to keep a text sample.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of text samples to retain.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--preferred-field",
        type=str,
        default="text",
        help="Preferred column/key to extract from structured files (CSV/JSONL).",
    )
    parser.add_argument(
        "--fallback-dataset",
        type=str,
        default="oscar",
        help="Optional Hugging Face dataset to use when the repository lacks textual resources.",
    )
    parser.add_argument(
        "--fallback-config",
        type=str,
        default="unshuffled_deduplicated_tl",
        help="Dataset configuration for the fallback dataset.",
    )
    parser.add_argument(
        "--fallback-split",
        type=str,
        default="train",
        help="Dataset split for the fallback dataset.",
    )
    parser.add_argument(
        "--fallback-text-column",
        type=str,
        default="text",
        help="Text column for the fallback dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and sample selection.",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable downloading from a fallback dataset if the repository lacks usable text files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_path = download_repo(args.repo_id, args.cache_dir, args.allow_patterns)
    repo_texts = collect_repo_texts(repo_path, args.min_chars, args.preferred_field)

    if args.max_samples and repo_texts:
        repo_texts = repo_texts[: args.max_samples]

    fallback_used = False
    texts: List[str] = repo_texts

    fallback_config = args.fallback_config
    if isinstance(fallback_config, str) and fallback_config.lower() in {"", "none", "null"}:
        fallback_config = None

    if len(texts) < 100 and not args.disable_fallback:
        fallback_used = True
        fallback_name = args.fallback_dataset if fallback_config is None else f"{args.fallback_dataset}/{fallback_config}"
        fallback_texts = load_fallback_dataset(
            args.fallback_dataset,
            fallback_config,
            args.fallback_split,
            args.fallback_text_column,
            args.max_samples,
            args.seed,
            args.min_chars,
        )
        texts = fallback_texts
    else:
        fallback_name = None

    if not texts:
        raise RuntimeError(
            "No usable text samples were found in the repository and fallback dataset is disabled or empty."
        )

    save_splits(texts, args.output_dir, args.seed, args.val_ratio)
    dump_metadata(
        args.output_dir,
        args.repo_id,
        source_texts=len(repo_texts),
        fallback_used=fallback_used,
        fallback_name=fallback_name,
        total_samples=len(texts),
    )

    print(
        f"Prepared corpus with {len(texts)} samples. Train/validation CSV files are available in "
        f"{args.output_dir}."
    )


if __name__ == "__main__":
    main()
