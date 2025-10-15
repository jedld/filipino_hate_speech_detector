import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    missing_columns = {col for col in ["text", "label"] if col not in df.columns}
    if missing_columns:
        raise ValueError(f"CSV {path} must contain columns: text, label. Missing: {missing_columns}")
    return df[["text", "label"]]


def summarize_split(name: str, df: pd.DataFrame) -> Dict[str, object]:
    total = len(df)
    label_counts = df["label"].value_counts().sort_index().to_dict()
    label_percentages = {
        str(label): (count / total * 100.0 if total else 0.0) for label, count in label_counts.items()
    }
    avg_length = df["text"].astype(str).str.len().mean() if total else 0.0
    median_length = df["text"].astype(str).str.len().median() if total else 0.0
    return {
        "split": name,
        "total_rows": total,
        "class_counts": {str(label): int(count) for label, count in label_counts.items()},
        "class_percentages": label_percentages,
        "avg_text_length": avg_length,
        "median_text_length": median_length,
    }


def summarize_dataset(paths: List[Path]) -> Dict[str, object]:
    summaries = []
    combined = []
    for path in paths:
        df = load_dataframe(path)
        split_name = path.stem
        summaries.append(summarize_split(split_name, df))
        df = df.assign(__split=split_name)
        combined.append(df)

    overall_summary = {}
    if combined:
        concat_df = pd.concat(combined, ignore_index=True)
        overall_summary = summarize_split("combined", concat_df.drop(columns=["__split"]))
        overall_summary["split_breakdown"] = {
            summary["split"]: summary["total_rows"] for summary in summaries
        }

    return {
        "splits": summaries,
        "overall": overall_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print statistics about dataset CSV files")
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        default=[
            Path("data/combined/processed/train.csv"),
            Path("data/combined/processed/validation.csv"),
            Path("data/combined/processed/test.csv"),
        ],
        help="CSV files to analyse (default: combined processed train/validation/test)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the summary as JSON instead of a human-readable table",
    )
    return parser.parse_args()


def format_summary_human(summary: Dict[str, object]) -> str:
    lines = []
    lines.append("Dataset Summary\n===============")
    for split_summary in summary["splits"]:
        lines.append(f"\nSplit: {split_summary['split']}")
        lines.append(f"  Total rows: {split_summary['total_rows']}")
        for label, count in split_summary["class_counts"].items():
            pct = split_summary["class_percentages"][label]
            lines.append(f"  Label {label}: {count} ({pct:.2f}%)")
        lines.append(
            "  Avg/Median text length: "
            f"{split_summary['avg_text_length']:.1f} / {split_summary['median_text_length']:.1f} chars"
        )

    if summary.get("overall"):
        overall = summary["overall"]
        lines.append("\nOverall")
        lines.append(f"  Total rows: {overall['total_rows']}")
        for label, count in overall["class_counts"].items():
            pct = overall["class_percentages"][label]
            lines.append(f"  Label {label}: {count} ({pct:.2f}%)")
        lines.append(
            "  Avg/Median text length: "
            f"{overall['avg_text_length']:.1f} / {overall['median_text_length']:.1f} chars"
        )
        if overall.get("split_breakdown"):
            lines.append("  Split breakdown:")
            for split, count in overall["split_breakdown"].items():
                lines.append(f"    {split}: {count}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(args.csv)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_summary_human(summary))


if __name__ == "__main__":
    main()
