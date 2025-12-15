import argparse
import torch
import sys
import importlib.util

# Workaround for "ValueError: cv2.__spec__ is None" in broken environments
# This happens when opencv-python is installed but broken, and transformers tries to check for it.
original_find_spec = importlib.util.find_spec
def patched_find_spec(name, package=None):
    try:
        return original_find_spec(name, package)
    except ValueError:
        if name == "cv2":
            return None
        raise
importlib.util.find_spec = patched_find_spec

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import pandas as pd
from pathlib import Path
import glob
import os

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_checkpoint(base_dir):
    # Try to find the latest checkpoint in run_42 or similar
    # Pattern: models/llama_classifier/run_*/checkpoint-*
    search_path = Path(base_dir)
    if not search_path.exists():
        return None
    
    # If it's a direct checkpoint dir
    if (search_path / "adapter_config.json").exists():
        return str(search_path)
        
    # Search for run directories
    runs = sorted(list(search_path.glob("run_*")))
    if not runs:
        # Try looking for checkpoints directly
        checkpoints = sorted(list(search_path.glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
        if checkpoints:
            return str(checkpoints[-1])
        return None
        
    # Look in the first run (usually run_42)
    latest_run = runs[0]
    checkpoints = sorted(list(latest_run.glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
    if checkpoints:
        return str(checkpoints[-1])
    
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Gradio Demo for Llama 3 Hate Speech Classifier")
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model path or identifier"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="models/llama_classifier", # Point to the base experiment dir
        help="Path to the fine-tuned LoRA checkpoint or experiment directory"
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share the Gradio app publicly"
    )
    return parser.parse_args()

def load_model(base_model_id, checkpoint_dir):
    # Resolve checkpoint path
    real_checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if real_checkpoint_path:
        print(f"Found latest checkpoint: {real_checkpoint_path}")
        checkpoint_dir = real_checkpoint_path
    else:
        print(f"Warning: Could not auto-discover checkpoint in {checkpoint_dir}. Trying as is.")
    
    print(f"Loading base model: {base_model_id}")
    
    # Quantization Config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load Base Model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=2, # Placeholder, will be overwritten by adapter or config
        low_cpu_mem_usage=True
    )
    
    print(f"Loading LoRA adapter from: {checkpoint_dir}")
    # Load LoRA Adapter
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def predict(text, model, tokenizer, id2label):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
    # Get top prediction
    pred_id = torch.argmax(probabilities, dim=1).item()
    pred_label = id2label.get(pred_id, str(pred_id))
    confidence = probabilities[0][pred_id].item()
    
    # Format all scores
    scores = {id2label.get(i, str(i)): prob.item() for i, prob in enumerate(probabilities[0])}
    
    return scores

def main():
    args = parse_args()
    
    # 1. Load Labels (Try to infer or default)
    # In the notebook, labels were derived from data. 
    # We'll try to load from the checkpoint config if available, else default to Hate/Non-Hate
    # or try to read from the dataset if present.
    
    # Default fallback
    id2label = {0: "Non-Hate", 1: "Hate"} 
    
    # Try to find dataset to get real labels if possible
    train_csv = Path("data/combined/processed/train.csv")
    if train_csv.exists():
        try:
            df = pd.read_csv(train_csv)
            unique_labels = sorted({str(label) for label in df["label"].tolist()})
            id2label = {idx: label for idx, label in enumerate(unique_labels)}
            print(f"Loaded labels from {train_csv}: {id2label}")
        except Exception as e:
            print(f"Could not load labels from csv: {e}")
    
    # 2. Load Model
    try:
        model, tokenizer = load_model(args.base_model, args.checkpoint_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the checkpoint path is correct.")
        return

    # 3. Define Gradio Interface
    def gradio_predict(text):
        return predict(text, model, tokenizer, id2label)

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Textbox(lines=3, placeholder="Enter text here...", label="Input Text"),
        outputs=gr.Label(num_top_classes=2, label="Predictions"),
        title="Filipino Hate Speech Detector (Llama 3)",
        description="Detects hate speech in Tagalog/Filipino text using a fine-tuned Llama 3 model.",
        examples=[
            ["Ang ganda ng araw ngayon!"],
            ["Putang ina mo!"],
            ["Nakakainis ka talaga."]
        ]
    )
    
    demo.launch(share=args.share, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
