import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from models.transformer import SmallTransformerClassifier
from utils.data_utils import SimpleTokenizer


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "processed" / "sentiment_transformer.pt"
DEFAULT_LAST_MODEL_PATH = PROJECT_ROOT / "models" / "processed" / "sentiment_transformer_last.pt"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "processed" / "tokenizer.json"
DEFAULT_FALLBACK_DATA = PROJECT_ROOT / "data" / "sample_data.csv"


def load_tokenizer(tokenizer_path: Path, fallback_data_path: Path) -> SimpleTokenizer:
	if tokenizer_path.exists():
		return SimpleTokenizer.load(tokenizer_path)
	if fallback_data_path.exists():
		df = pd.read_csv(fallback_data_path)
		return SimpleTokenizer(df["text"].astype(str).tolist())
	raise FileNotFoundError(
		"Tokenizer file not found and fallback data unavailable. Prepare datasets or train the model first."
	)


def _normalize_hparams(raw_hparams: Dict[str, object], tokenizer: SimpleTokenizer) -> Dict[str, object]:
	return {
		"embed_dim": int(raw_hparams.get("embed_dim", 128)),
		"num_heads": int(raw_hparams.get("num_heads", 2)),
		"num_layers": int(raw_hparams.get("num_layers", 2)),
		"dropout": float(raw_hparams.get("dropout", 0.1)),
		"max_len": int(raw_hparams.get("max_len", getattr(tokenizer, "max_len", 128))),
	}


def extract_model_hparams(state: Dict[str, object], tokenizer: SimpleTokenizer) -> Dict[str, object]:
	if state is None:
		state = {}
	if "model_hyperparams" in state:
		return _normalize_hparams(state["model_hyperparams"], tokenizer)
	config = state.get("config", {})
	inferred = {
		"embed_dim": config.get("embed_dim", 128),
		"num_heads": config.get("num_heads", 2),
		"num_layers": config.get("num_layers", 2),
		"dropout": config.get("dropout", 0.1),
		"max_len": config.get("max_len", getattr(tokenizer, "max_len", 128)),
	}
	return _normalize_hparams(inferred, tokenizer)


def load_model(
	model_path: Path,
	tokenizer: SimpleTokenizer,
	device: torch.device,
) -> Tuple[SmallTransformerClassifier, Dict[str, object]]:
	checkpoint = torch.load(model_path, map_location=device)
	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
		metadata = checkpoint
	else:
		state_dict = checkpoint
		metadata = {}

	hparams = extract_model_hparams(metadata, tokenizer)
	tokenizer.max_len = hparams["max_len"]

	model = SmallTransformerClassifier(
		vocab_size=tokenizer.vocab_size(),
		embed_dim=hparams["embed_dim"],
		num_heads=hparams["num_heads"],
		num_layers=hparams["num_layers"],
		max_len=hparams["max_len"],
		dropout=hparams["dropout"],
	)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	return model, hparams


def predict_proba(text: str, model: SmallTransformerClassifier, tokenizer: SimpleTokenizer, device: torch.device) -> Dict[str, float]:
	sanitized = text.strip()
	if not sanitized:
		raise ValueError("Input text must be non-empty.")
	encoded = torch.tensor([tokenizer.encode(sanitized)], dtype=torch.long, device=device)
	with torch.no_grad():
		logits = model(encoded)
		probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
	return {
		"Not hate speech": float(probs[0]),
		"Hate speech": float(probs[1]),
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run hate speech inference with the trained transformer model")
	parser.add_argument("--text", type=str, required=True, help="Text to analyze")
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the best model checkpoint")
	parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH, help="Path to the saved tokenizer")
	parser.add_argument(
		"--fallback-data",
		type=Path,
		default=DEFAULT_FALLBACK_DATA,
		help="Fallback CSV containing a text column, used only if tokenizer.json is missing",
	)
	parser.add_argument("--json", action="store_true", help="Print probabilities as JSON")
	parser.add_argument("--show-hparams", action="store_true", help="Display model hyperparameters before inference")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_path = args.model_path
	if not model_path.exists():
		if DEFAULT_LAST_MODEL_PATH.exists():
			print(f"Warning: {model_path} not found. Using last checkpoint at {DEFAULT_LAST_MODEL_PATH}")
			model_path = DEFAULT_LAST_MODEL_PATH
		else:
			raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

	tokenizer = load_tokenizer(args.tokenizer_path, args.fallback_data)
	model, hparams = load_model(model_path, tokenizer, device)

	if args.show_hparams:
		print("Resolved model hyperparameters:")
		for key, value in hparams.items():
			print(f"  {key}: {value}")

	probabilities = predict_proba(args.text, model, tokenizer, device)
	label = "Hate speech" if probabilities["Hate speech"] >= 0.5 else "Not hate speech"

	output = {
		"input": args.text,
		"label": label,
		"probabilities": probabilities,
	}

	if args.json:
		print(json.dumps(output, indent=2))
	else:
		print(f"Input: {output['input']}")
		print(f"Prediction: {output['label']}")
		print(f"Hate speech probability: {probabilities['Hate speech']:.4f}")
		print(f"Not hate speech probability: {probabilities['Not hate speech']:.4f}")


if __name__ == "__main__":
	main()
 
