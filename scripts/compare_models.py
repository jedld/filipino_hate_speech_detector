import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import importlib.util
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Workaround for "ValueError: cv2.__spec__ is None"
# This often happens when importing transformers/peft in some environments
original_find_spec = importlib.util.find_spec
def patched_find_spec(name, package=None):
    if name == 'cv2':
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = patched_find_spec

from models.transformer import SmallTransformerClassifier
from utils.data_utils import SimpleTokenizer, EncodedTextDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_exp1_model(checkpoint_path, train_csv_path):
    print("Loading Experiment 1 Model (Small Transformer)...")
    
    # Rebuild Tokenizer (assuming it was built from train.csv)
    print(f"Rebuilding tokenizer from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    train_texts = train_df['text'].astype(str).tolist()
    
    # Hyperparams from experiment_1.ipynb
    MAX_LEN = 256
    EMBED_DIM = 6
    NUM_HEADS = 2
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    tokenizer = SimpleTokenizer(train_texts, max_len=MAX_LEN)
    
    model = SmallTransformerClassifier(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run Experiment 1 notebook first.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

def get_latest_checkpoint(base_dir):
    search_path = Path(base_dir)
    if not search_path.exists():
        return None

    if (search_path / "adapter_config.json").exists():
        return str(search_path)

    runs = sorted(list(search_path.glob("run_*")))
    if not runs:
        checkpoints = sorted(list(search_path.glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
        if checkpoints:
            return str(checkpoints[-1])
        return None

    latest_run = runs[0]
    checkpoints = sorted(list(latest_run.glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
    if checkpoints:
        return str(checkpoints[-1])

    return None

def load_exp3_model(base_model_id, checkpoint_dir):
    print("Loading Experiment 3 Model (Llama 3)...")
    
    real_checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if not real_checkpoint_path:
        raise FileNotFoundError(f"Could not find checkpoint in {checkpoint_dir}")
    
    print(f"Using checkpoint: {real_checkpoint_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=2,
        low_cpu_mem_usage=True
    )

    model = PeftModel.from_pretrained(model, real_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Ensure pad token is set for batching
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Also update model config
    model.config.pad_token_id = tokenizer.pad_token_id
        
    model.eval()
    return model, tokenizer

def run_inference_exp1(model, tokenizer, texts, batch_size=32):
    encoded = [tokenizer.encode(t) for t in texts]
    # Create dummy labels
    dataset = EncodedTextDataset(texts, [0]*len(texts), tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Exp 1 Inference"):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return all_preds, all_probs

def run_inference_exp3(model, tokenizer, texts, batch_size=8):
    all_preds = []
    all_probs = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Exp 3 Inference"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return all_preds, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="data/combined/processed/test.csv")
    parser.add_argument("--train_csv", default="data/combined/processed/train.csv")
    parser.add_argument("--exp1_checkpoint", default="models/experiment_1/best_model.pt")
    parser.add_argument("--exp3_checkpoint_dir", default="models/llama_classifier")
    parser.add_argument("--output_dir", default="results/comparison")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"Loading test data from {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    # --- Experiment 1 ---
    try:
        exp1_model, exp1_tokenizer = load_exp1_model(args.exp1_checkpoint, args.train_csv)
        exp1_preds, exp1_probs = run_inference_exp1(exp1_model, exp1_tokenizer, texts)
        
        # Store confidence of predicted class
        exp1_conf = [p[pred] for p, pred in zip(exp1_probs, exp1_preds)]
        
        df['exp1_pred'] = exp1_preds
        df['exp1_conf'] = exp1_conf
        
        # Free memory
        del exp1_model
        del exp1_tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to run Experiment 1 inference: {e}")
        df['exp1_pred'] = -1
        df['exp1_conf'] = 0.0

    # --- Experiment 3 ---
    try:
        exp3_model, exp3_tokenizer = load_exp3_model("meta-llama/Meta-Llama-3.1-8B-Instruct", args.exp3_checkpoint_dir)
        exp3_preds, exp3_probs = run_inference_exp3(exp3_model, exp3_tokenizer, texts)
        
        exp3_conf = [p[pred] for p, pred in zip(exp3_probs, exp3_preds)]
        
        df['exp3_pred'] = exp3_preds
        df['exp3_conf'] = exp3_conf
        
        del exp3_model
        del exp3_tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to run Experiment 3 inference: {e}")
        df['exp3_pred'] = -1
        df['exp3_conf'] = 0.0

    # --- Analysis ---
    print("\nGenerating Analysis Report...")
    
    # Filter out failed runs
    valid_mask = (df['exp1_pred'] != -1) & (df['exp3_pred'] != -1)
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("No valid predictions to compare.")
        return

    # Categories
    # 1. Exp1 Correct, Exp3 Wrong
    e1_corr_e3_wrong = valid_df[(valid_df['exp1_pred'] == valid_df['label']) & (valid_df['exp3_pred'] != valid_df['label'])]
    
    # 2. Exp3 Correct, Exp1 Wrong
    e3_corr_e1_wrong = valid_df[(valid_df['exp3_pred'] == valid_df['label']) & (valid_df['exp1_pred'] != valid_df['label'])]
    
    # 3. Both Correct
    both_correct = valid_df[(valid_df['exp1_pred'] == valid_df['label']) & (valid_df['exp3_pred'] == valid_df['label'])]
    
    # 4. Both Wrong
    both_wrong = valid_df[(valid_df['exp1_pred'] != valid_df['label']) & (valid_df['exp3_pred'] != valid_df['label'])]
    
    # Save full results
    full_results_path = output_dir / "full_comparison_results.csv"
    df.to_csv(full_results_path, index=False)
    print(f"Full results saved to {full_results_path}")
    
    # Generate Markdown Report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Comparison Report: Small Transformer vs Llama 3\n\n")
        
        f.write("## Summary Statistics\n")
        f.write(f"- Total Test Samples: {len(valid_df)}\n")
        f.write(f"- Exp 1 Accuracy: {(valid_df['exp1_pred'] == valid_df['label']).mean():.4f}\n")
        f.write(f"- Exp 3 Accuracy: {(valid_df['exp3_pred'] == valid_df['label']).mean():.4f}\n\n")
        
        f.write("## Disagreement Analysis\n")
        f.write(f"- **Exp 1 Correct, Exp 3 Wrong**: {len(e1_corr_e3_wrong)} samples\n")
        f.write(f"- **Exp 3 Correct, Exp 1 Wrong**: {len(e3_corr_e1_wrong)} samples\n")
        f.write(f"- **Both Correct**: {len(both_correct)} samples\n")
        f.write(f"- **Both Wrong**: {len(both_wrong)} samples\n\n")
        
        def write_samples(title, sample_df, limit=20):
            f.write(f"### {title}\n")
            if len(sample_df) == 0:
                f.write("No samples found.\n\n")
                return
            
            # Sort by confidence of the correct model (or high confidence of wrong model for 'Both Wrong')
            # For simplicity, just take head
            samples = sample_df.head(limit)
            
            for _, row in samples.iterrows():
                f.write(f"**Text:** {row['text']}\n\n")
                f.write(f"- Label: {row['label']}\n")
                f.write(f"- Exp 1 Pred: {row['exp1_pred']} (Conf: {row['exp1_conf']:.2f})\n")
                f.write(f"- Exp 3 Pred: {row['exp3_pred']} (Conf: {row['exp3_conf']:.2f})\n")
                f.write("---\n")
            f.write("\n")

        write_samples("Exp 1 Correct, Exp 3 Wrong (Small Model Wins)", e1_corr_e3_wrong)
        write_samples("Exp 3 Correct, Exp 1 Wrong (Llama 3 Wins)", e3_corr_e1_wrong)
        write_samples("Both Wrong (Hard Samples)", both_wrong)
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
