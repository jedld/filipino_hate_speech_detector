import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover - optional optimization
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency guard
    LoraConfig = TaskType = None  # type: ignore[assignment]
    get_peft_model = prepare_model_for_kbit_training = None  # type: ignore[assignment]
    PEFT_AVAILABLE = False


@dataclass
class TokenizedDataset:
    train: Dataset
    validation: Dataset
    test: Dataset
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}, but it was not found.")


def load_dataframe(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"File {path} must contain '{text_column}' and '{label_column}' columns.")
    return df[[text_column, label_column]].rename(columns={text_column: "text", label_column: "label"})


def prepare_label_mappings(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique_labels = sorted({str(label) for label in train_df["label"].tolist()})
    if len(unique_labels) < 2:
        raise ValueError("Classification requires at least two unique labels.")
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def convert_labels(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    mapped = df.copy()
    mapped["label"] = mapped["label"].astype(str).map(label2id)
    if mapped["label"].isnull().any():
        missing = df.loc[mapped["label"].isnull(), "label"].unique()
        raise ValueError(f"Encountered labels not in training set: {missing}")
    mapped["label"] = mapped["label"].astype(int)
    return mapped


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def log_trainable_parameters(model: torch.nn.Module) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    print(
        f"Trainable params: {trainable_params:,} | Total params: {total_params:,} | Trainable%: {ratio:.4f}"
    )


def build_training_arguments(args: argparse.Namespace, checkpoints_dir: Path) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    param_names = set(signature.parameters.keys())
    kwargs: Dict[str, Any] = {}

    def set_kw(name: str, value: Any) -> None:
        if name in param_names:
            kwargs[name] = value

    def set_kw_alias(value: Any, *aliases: str) -> None:
        for alias in aliases:
            if alias in param_names:
                kwargs[alias] = value
                return

    set_kw("output_dir", str(checkpoints_dir))
    set_kw("per_device_train_batch_size", args.batch_size)
    set_kw("per_device_eval_batch_size", args.eval_batch_size)
    set_kw("gradient_accumulation_steps", args.gradient_accumulation_steps)
    set_kw("num_train_epochs", args.epochs)
    set_kw("learning_rate", args.learning_rate)
    set_kw("weight_decay", args.weight_decay)
    set_kw("warmup_ratio", args.warmup_ratio)
    set_kw("logging_steps", args.logging_steps)
    set_kw_alias("steps", "evaluation_strategy", "eval_strategy")
    set_kw("eval_steps", args.eval_steps)
    set_kw_alias("steps", "save_strategy", "checkpointing_strategy")
    set_kw("save_steps", args.save_steps)
    set_kw("save_total_limit", args.save_total_limit)
    set_kw("load_best_model_at_end", True)
    set_kw("metric_for_best_model", "f1")
    set_kw("greater_is_better", True)
    set_kw("bf16", args.bf16)
    set_kw("fp16", args.fp16)
    set_kw("push_to_hub", args.push_to_hub)
    set_kw("hub_model_id", args.hub_model_id)
    set_kw("hub_token", args.hf_token)
    set_kw("remove_unused_columns", False)
    set_kw("seed", args.seed)

    report_to = [] if args.no_tensorboard else ["tensorboard"]
    set_kw("report_to", report_to)

    return TrainingArguments(**kwargs)


def tokenize_dataframe(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_length: int,
) -> TokenizedDataset:
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return TokenizedDataset(train=train_dataset, validation=val_dataset, test=test_dataset, label2id={}, id2label={})


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    labels = labels.astype(int)
    accuracy = (preds == labels).mean().item()

    classes = np.unique(labels)
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []

    for cls in classes:
        tp = np.logical_and(preds == cls, labels == cls).sum()
        fp = np.logical_and(preds == cls, labels != cls).sum()
        fn = np.logical_and(preds != cls, labels == cls).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": float(accuracy),
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }


def build_model(
    base_model: str,
    num_labels: int,
    device: str,
    bf16: bool,
    fp16: bool,
    train_head_only: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
) -> torch.nn.Module:
    torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    def quant_config(is_4bit: bool, is_8bit: bool):
        if not (is_4bit or is_8bit):
            return None
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes is required for quantized loading but is not available.")
        if is_4bit and is_8bit:
            raise ValueError("Only one of 4-bit or 8-bit quantization can be enabled at a time.")
        if is_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            )
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    quantization_config = quant_config(load_in_4bit, load_in_8bit)

    device_map = None
    if device == "auto":
        if quantization_config is not None:
            device_map = "auto"
        elif not train_head_only:
            device_map = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.config.problem_type = "single_label_classification"

    if train_head_only:
        for name, param in model.named_parameters():
            trainable = name.startswith("score") or name.startswith("classifier")
            param.requires_grad = trainable
        if hasattr(model, "score") and hasattr(model.score, "weight"):
            torch.nn.init.normal_(model.score.weight, mean=0.0, std=model.config.initializer_range)
            if getattr(model.score, "bias", None) is not None:
                torch.nn.init.zeros_(model.score.bias)
        if device != "auto":
            target_device = torch.device(device)
            model.to(target_device)
        elif device_map == "auto":
            if quantization_config is not None:
                if load_in_4bit:
                    setattr(model, "is_loaded_in_4bit", True)
                if load_in_8bit:
                    setattr(model, "is_loaded_in_8bit", True)
            else:
                setattr(model, "is_loaded_in_8bit", True)
        return model

    if not PEFT_AVAILABLE:
        raise ImportError(
            "LoRA fine-tuning requires the 'peft' package. Install it via 'pip install peft'."
        )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        setattr(model, "is_loaded_in_4bit", True)
    elif load_in_8bit:
        setattr(model, "is_loaded_in_8bit", True)

    target_modules = lora_target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hate-speech classifier using a pretrained LLM backbone (QLoRA)")
    parser.add_argument("--train-csv", type=Path, default=Path("data/combined/processed/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/combined/processed/validation.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/combined/processed/test.csv"))
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("models/llama_classifier"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of module names to apply LoRA to (defaults to common projection layers).",
    )
    parser.add_argument("--train-head-only", action="store_true", help="Freeze the base model and train only the classification head.")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Select only one of --bf16 or --fp16.")
    if args.load_in_4bit and args.device == "cpu":
        raise ValueError("4-bit quantization requires GPU execution. Remove --load-in-4bit or set --device auto.")
    if args.train_head_only and args.load_in_4bit:
        raise ValueError("Head-only training does not support 4-bit quantization. Drop --load-in-4bit or disable --train-head-only.")
    if args.load_in_8bit and BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes>=0.41 is required for 8-bit loading. Install it or disable --load-in-8bit.")
    if args.load_in_8bit and args.device == "cpu":
        raise ValueError("8-bit quantization requires GPU execution. Remove --load-in-8bit or set --device auto.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available but --device cuda was requested.")

    set_seed(args.seed)

    ensure_file(args.train_csv, "training CSV")
    ensure_file(args.val_csv, "validation CSV")
    ensure_file(args.test_csv, "test CSV")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataframe(args.train_csv, args.text_column, args.label_column)
    val_df = load_dataframe(args.val_csv, args.text_column, args.label_column)
    test_df = load_dataframe(args.test_csv, args.text_column, args.label_column)

    label2id, id2label = prepare_label_mappings(train_df)
    train_df = convert_labels(train_df, label2id)
    val_df = convert_labels(val_df, label2id)
    test_df = convert_labels(test_df, label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = tokenize_dataframe(train_df, val_df, test_df, tokenizer, args.max_length)
    tokenized = TokenizedDataset(
        train=tokenized.train,
        validation=tokenized.validation,
        test=tokenized.test,
        label2id=label2id,
        id2label=id2label,
    )

    device = args.device
    model = build_model(
        base_model=args.base_model,
        num_labels=len(label2id),
        device=device,
        bf16=args.bf16,
        fp16=args.fp16,
        train_head_only=args.train_head_only,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    model.config.label2id = label2id
    model.config.id2label = id2label
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.train_head_only:
        log_trainable_parameters(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = build_training_arguments(args, checkpoints_dir)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized.train,
        "eval_dataset": tokenized.validation,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    if args.patience > 0:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.patience))

    train_result = trainer.train()

    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=tokenized.test)

    model_save_dir = args.output_dir / "model"
    tokenizer_save_dir = args.output_dir / "tokenizer"
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(tokenizer_save_dir, legacy_format=False)

    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    metrics = {
        "train_metrics": to_serializable(train_result.metrics),
        "validation_metrics": to_serializable(eval_metrics),
        "test_metrics": to_serializable(test_metrics),
        "label2id": label2id,
        "id2label": id2label,
        "config": config_dict,
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    label_map_path = args.output_dir / "label_map.json"
    label_map_path.write_text(json.dumps({"label2id": label2id, "id2label": id2label}, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Validation metrics: {json.dumps(eval_metrics, indent=2)}")
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Model saved to {model_save_dir}")
    print(f"Tokenizer saved to {tokenizer_save_dir}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Label mapping saved to {label_map_path}")


if __name__ == "__main__":
    main()
