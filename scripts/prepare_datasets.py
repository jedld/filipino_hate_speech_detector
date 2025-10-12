import argparse
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}, but it was not found.")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_readme(dir_path: Path, title: str, body: str) -> None:
    content = f"# {title}\n\n{body}\n"
    (dir_path / "README.md").write_text(content, encoding="utf-8")


def normalize_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Try to normalize dataframe to have 'text' and 'label' columns (binary 0/1).
    Heuristics are used; details stored in report.
    """
    original_cols = list(df.columns)
    text_candidates = [
        "text",
        "tweet",
        "content",
        "sentence",
        "message",
        "body",
        "comment",
        "statement",
        "utterance",
    ]
    label_candidates = [
        "label",
        "labels",
        "target",
        "class",
        "hs",
        "is_hatespeech",
        "hate_speech",
        "toxicity",
        "offensive",
    ]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        # pick first object/string-like column
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
                text_col = c
                break

    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        # pick first non-text, low-cardinality column
        for c in df.columns:
            if c == text_col:
                continue
            nunique = df[c].nunique(dropna=True)
            if nunique <= 5:
                label_col = c
                break

    decisions = {"original_columns": original_cols, "text_col": text_col, "label_col": label_col}

    if text_col is None or label_col is None:
        decisions["normalized"] = False
        report.setdefault("normalization_attempts", []).append(decisions)
        raise ValueError("Could not infer text/label columns for normalization")

    # Normalize labels to binary 0/1 where 1 = hate/offensive, 0 = not-hate
    series = df[label_col]

    def to_binary(val):
        # booleans
        if isinstance(val, (bool,)):
            return int(val)
        # numeric
        if pd.api.types.is_number(val):
            try:
                iv = int(val)
                return 1 if iv >= 1 else 0
            except Exception:
                pass
        # strings
        if isinstance(val, str):
            s = val.strip().lower()
            # common positive/hate terms
            positive_terms = [
                "hate", "hatespeech", "hate_speech", "toxic", "offensive", "abusive", "insult", "hs",
            ]
            negative_terms = [
                "not_hate", "non-toxic", "clean", "neutral", "normal", "nonhate", "non-hate", "no",
            ]
            if any(t in s for t in positive_terms):
                return 1
            if any(t in s for t in negative_terms):
                return 0
            # fall back: map class names like hate/none
            if s in {"hate", "offensive", "toxic", "abusive"}:
                return 1
            if s in {"none", "neutral", "clean"}:
                return 0
            # try categorical encodings like '0', '1'
            if s.isdigit():
                return 1 if int(s) >= 1 else 0
        # unknown -> 0
        return 0

    try:
        labels = series.map(to_binary).astype(int)
    except Exception:
        labels = series.apply(to_binary).astype(int)

    out = pd.DataFrame({"text": df[text_col].astype(str), "label": labels})
    decisions["normalized"] = True
    report.setdefault("normalization_attempts", []).append(decisions)
    return out


def _validate_split_ratios(ratios: Tuple[float, float, float]) -> Tuple[float, float, float]:
    train_ratio, val_ratio, test_ratio = ratios
    if any(r <= 0 for r in ratios):
        raise ValueError("Split ratios must be greater than 0.")
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0.")
    return train_ratio, val_ratio, test_ratio


def _split_dataframe(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    train_ratio, val_ratio, test_ratio = ratios
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    if total < 3:
        raise ValueError("Dataset too small to split into train/val/test")

    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    remaining = total - train_count - val_count
    if remaining <= 0:
        val_count = max(1, val_count - 1)
        remaining = total - train_count - val_count
        if remaining <= 0:
            train_count = max(1, train_count - 1)
            remaining = total - train_count - val_count
    test_count = remaining
    if test_count <= 0:
        raise ValueError("Split ratios resulted in empty test set. Adjust the ratios.")

    train_end = train_count
    val_end = train_end + val_count

    splits = {
        "train": shuffled.iloc[:train_end].reset_index(drop=True),
        "validation": shuffled.iloc[train_end:val_end].reset_index(drop=True),
        "test": shuffled.iloc[val_end:].reset_index(drop=True),
    }
    return splits


def prepare_hf_dataset(
    base_dir: Path,
    report: dict,
    split_ratios: Optional[Tuple[float, float, float]] = None,
    split_seed: int = 42,
) -> Dict[str, object]:
    """Download and prepare jcblaise/hatespeech_filipino via Hugging Face datasets."""
    from datasets import load_dataset  # lazy import to avoid hard dependency in non-use cases

    dataset_name = "jcblaise/hatespeech_filipino"
    ds_dir = base_dir / "hatespeech_filipino"
    raw_dir = ds_dir / "raw"
    proc_dir = ds_dir / "processed"
    ensure_dir(raw_dir)
    ensure_dir(proc_dir)

    logging.info("Loading dataset %s from Hugging Face...", dataset_name)
    ds_dict = load_dataset(dataset_name, trust_remote_code=True)

    split_reports = {}
    normalized_frames: dict[str, pd.DataFrame] = {}
    for split, ds in ds_dict.items():
        raw_path = raw_dir / f"{split}.jsonl"
        logging.info("Saving raw split '%s' to %s", split, raw_path)
        ds.to_json(str(raw_path), orient="records", lines=True, force_ascii=False)

        r = {"split": split}
        try:
            df = ds.to_pandas()
            norm_df = normalize_columns(df, r)
            normalized_frames[split] = norm_df
            if split_ratios is None:
                out_path = proc_dir / f"{split}.csv"
                norm_df.to_csv(out_path, index=False)
                r["processed_output"] = str(out_path)
            r["num_rows"] = len(norm_df)
        except Exception as e:
            r["error"] = f"Normalization failed: {e}"
        split_reports[split] = r

    if split_ratios is not None:
        if not normalized_frames:
            raise ValueError("No normalized splits available to recombine. Cannot apply custom ratios.")
        combined_df = pd.concat(normalized_frames.values(), ignore_index=True)
        logging.info(
            "Applying custom split ratios train/val/test = %.3f/%.3f/%.3f with seed %d",
            split_ratios[0],
            split_ratios[1],
            split_ratios[2],
            split_seed,
        )
        custom_splits = _split_dataframe(combined_df, split_ratios, split_seed)
        custom_sizes = {}
        for split_name, split_df in custom_splits.items():
            out_path = proc_dir / f"{split_name}.csv"
            split_df.to_csv(out_path, index=False)
            custom_sizes[split_name] = len(split_df)
            split_reports[split_name] = {
                "split": split_name,
                "custom_split": True,
                "processed_output": str(out_path),
                "num_rows": len(split_df),
            }
        report.setdefault("huggingface_custom_split", {})["ratios"] = {
            "train": split_ratios[0],
            "validation": split_ratios[1],
            "test": split_ratios[2],
            "seed": split_seed,
        }
        report["huggingface_custom_split"]["sizes"] = custom_sizes

    hf_summary = {
        "name": dataset_name,
        "output_dir": str(ds_dir),
        "custom_split_applied": split_ratios is not None,
        "splits": split_reports,
    }
    report["huggingface_dataset"] = hf_summary

    save_readme(
        ds_dir,
        "Hugging Face: hatespeech_filipino",
        "This folder contains the raw and processed splits of the Hugging Face dataset 'jcblaise/hatespeech_filipino'.\n\n"
        "- Raw: JSON Lines per split under 'raw/'.\n"
        "- Processed: CSV with columns [text,label] under 'processed/'.\n"
        "- Label normalization is heuristic: 1 indicates hate/offensive; 0 otherwise.\n",
    )

    return {
        "processed_dir": proc_dir,
        "split_reports": split_reports,
        "normalized_frame_counts": {k: v.get("num_rows") for k, v in split_reports.items() if v.get("processed_output")},
    }


def download_github_repo_zip(owner_repo: str, dest_dir: Path) -> Path:
    """Download GitHub repository zip (tries main then master) and extract to dest_dir.
    Returns the path to the extracted top-level directory.
    """
    import requests

    ensure_dir(dest_dir)
    owner, repo = owner_repo.split("/")
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        logging.info("Attempting to download %s branch '%s'...", owner_repo, branch)
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(dest_dir)
            # determine top directory
            top_dirs = [p for p in dest_dir.iterdir() if p.is_dir() and p.name.startswith(repo + "-")]
            if top_dirs:
                return top_dirs[0]
            break
    raise RuntimeError(f"Failed to download or extract GitHub repo {owner_repo}")


def find_candidate_files(root: Path) -> list[Path]:
    exts = {".csv", ".tsv", ".jsonl", ".json"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def load_file_as_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        # try JSONL first then JSON array/object
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def prepare_github_dataset(base_dir: Path, report: dict) -> Dict[str, object]:
    owner_repo = "imperialite/filipino-tiktok-hatespeech"
    ds_dir = base_dir / "filipino_tiktok_hatespeech"
    raw_dir = ds_dir / "raw"
    proc_dir = ds_dir / "processed"
    ensure_dir(raw_dir)
    ensure_dir(proc_dir)

    extracted_dir = download_github_repo_zip(owner_repo, raw_dir)
    logging.info("Extracted GitHub dataset to %s", extracted_dir)
    candidates = find_candidate_files(extracted_dir)
    logging.info("Found %d candidate data files", len(candidates))

    processed = None
    norm_report = {}
    for f in candidates:
        try:
            df = load_file_as_df(f)
            # Require minimum two columns for text/label
            if df.shape[1] < 2:
                continue
            r = {"source_file": str(f)}
            norm_df = normalize_columns(df, r)
            processed = norm_df
            norm_report = r
            break
        except Exception as e:
            logging.debug("Skipping %s due to error: %s", f, e)
            continue

    if processed is not None:
        out_path = proc_dir / "all.csv"
        processed.to_csv(out_path, index=False)
        norm_report["processed_output"] = str(out_path)
        success = True
    else:
        out_path = None
        success = False

    github_summary = {
        "repo": owner_repo,
        "raw_dir": str(raw_dir),
        "processed": success,
        "details": norm_report if success else {"reason": "No candidate file could be normalized"},
    }
    report["github_dataset"] = github_summary

    save_readme(
        ds_dir,
        "GitHub: filipino-tiktok-hatespeech",
        "This folder contains the raw files extracted from the GitHub repository 'imperialite/filipino-tiktok-hatespeech'.\n\n"
        "- Raw: full repository content under 'raw/'.\n"
        "- Processed: If a suitable file was found, 'processed/all.csv' contains normalized [text,label].\n"
        "- Label normalization is heuristic: 1 indicates hate/offensive; 0 otherwise.\n",
    )

    return {
        "processed_path": out_path,
        "success": success,
    }


def prepare_combined_dataset(
    base_dir: Path,
    report: dict,
    hf_info: Dict[str, object],
    github_info: Dict[str, object],
    split_ratios: Optional[Tuple[float, float, float]],
    split_seed: int,
) -> Dict[str, object]:
    if not github_info.get("success"):
        raise ValueError("GitHub dataset was not successfully processed; cannot create combined dataset.")

    hf_proc_dir = Path(hf_info["processed_dir"])
    train_path = hf_proc_dir / "train.csv"
    val_path = hf_proc_dir / "validation.csv"
    test_path = hf_proc_dir / "test.csv"

    for required_path in (train_path, val_path, test_path):
        ensure_file_exists(required_path, f"Hugging Face processed split '{required_path.name}'")

    hf_train = pd.read_csv(train_path)
    hf_val = pd.read_csv(val_path)
    hf_test = pd.read_csv(test_path)

    hf_counts = {
        "train": len(hf_train),
        "validation": len(hf_val),
        "test": len(hf_test),
    }

    github_path = Path(github_info["processed_path"])
    ensure_file_exists(github_path, "GitHub processed dataset")
    github_df = pd.read_csv(github_path)
    github_rows = len(github_df)

    combined_df = pd.concat([hf_train, hf_val, hf_test, github_df], ignore_index=True)
    if combined_df.empty:
        raise ValueError("Combined dataset is empty after merging source datasets.")

    if split_ratios is not None:
        ratios_to_use = _validate_split_ratios(split_ratios)
    else:
        total_hf = sum(hf_counts.values())
        if total_hf == 0:
            ratios_to_use = (0.8, 0.1, 0.1)
        else:
            ratios_to_use = _validate_split_ratios(
                (
                    hf_counts["train"] / total_hf,
                    hf_counts["validation"] / total_hf,
                    hf_counts["test"] / total_hf,
                )
            )

    combined_splits = _split_dataframe(combined_df, ratios_to_use, split_seed)

    combined_dir = base_dir / "combined"
    proc_dir = combined_dir / "processed"
    ensure_dir(proc_dir)

    split_sizes = {}
    for split_name, split_df in combined_splits.items():
        out_path = proc_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        split_sizes[split_name] = len(split_df)

    summary = {
        "output_dir": str(proc_dir),
        "github_rows_added": github_rows,
        "total_rows": len(combined_df),
        "ratios_used": {
            "train": ratios_to_use[0],
            "validation": ratios_to_use[1],
            "test": ratios_to_use[2],
        },
        "split_sizes": split_sizes,
    }
    report["combined_dataset"] = summary
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare and download Filipino hate speech datasets")
    parser.add_argument("--output-dir", type=str, default="data", help="Base directory to place datasets")
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face dataset download")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub dataset download")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Custom train split ratio (use with --val-ratio and --test-ratio; ratios must sum to 1.0)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Custom validation split ratio (use with --train-ratio and --test-ratio)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Custom test split ratio (use with --train-ratio and --val-ratio)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed applied when recomputing custom train/val/test splits",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    base_dir = Path(args.output_dir)
    ensure_dir(base_dir)

    report = {"output_dir": str(base_dir)}

    ratio_values = (args.train_ratio, args.val_ratio, args.test_ratio)
    split_ratios: Optional[Tuple[float, float, float]] = None
    if any(v is not None for v in ratio_values):
        if None in ratio_values:
            parser.error("All of --train-ratio, --val-ratio, and --test-ratio must be provided together.")
        split_ratios = _validate_split_ratios((args.train_ratio, args.val_ratio, args.test_ratio))

    hf_info: Optional[Dict[str, object]] = None
    github_info: Optional[Dict[str, object]] = None

    if not args.skip_hf:
        try:
            hf_info = prepare_hf_dataset(base_dir, report, split_ratios=split_ratios, split_seed=args.split_seed)
        except Exception as e:
            logging.exception("Failed to prepare Hugging Face dataset: %s", e)
            report["huggingface_error"] = str(e)
    else:
        logging.info("Skipping Hugging Face dataset as requested")

    if not args.skip_github:
        try:
            github_info = prepare_github_dataset(base_dir, report)
        except Exception as e:
            logging.exception("Failed to prepare GitHub dataset: %s", e)
            report["github_error"] = str(e)
    else:
        logging.info("Skipping GitHub dataset as requested")

    if hf_info and not report.get("huggingface_error") and github_info and not report.get("github_error"):
        try:
            prepare_combined_dataset(
                base_dir,
                report,
                hf_info,
                github_info,
                split_ratios=split_ratios,
                split_seed=args.split_seed,
            )
        except Exception as e:
            logging.exception("Failed to prepare combined dataset: %s", e)
            report["combined_error"] = str(e)

    write_json(base_dir / "dataset_preparation_report.json", report)
    logging.info("Preparation complete. Report written to %s", base_dir / "dataset_preparation_report.json")


if __name__ == "__main__":
    main()
