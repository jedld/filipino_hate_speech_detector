import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import pandas as pd
import torch

from models.transformer import SmallTransformerClassifier
from utils.data_utils import SimpleTokenizer


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "processed" / "sentiment_transformer.pt"
DEFAULT_LAST_MODEL_PATH = PROJECT_ROOT / "models" / "processed" / "sentiment_transformer_last.pt"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "processed" / "tokenizer.json"
DEFAULT_FALLBACK_DATA = PROJECT_ROOT / "data" / "sample_data.csv"


class HateSpeechPipeline:
    """Loads the trained transformer and provides prediction utilities."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
        fallback_data_path: Path = DEFAULT_FALLBACK_DATA,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.fallback_data_path = fallback_data_path

        self.model_hparams: Dict[str, object] = {}

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()

    def _load_tokenizer(self) -> SimpleTokenizer:
        if self.tokenizer_path.exists():
            return SimpleTokenizer.load(self.tokenizer_path)
        if self.fallback_data_path.exists():
            df = pd.read_csv(self.fallback_data_path)
            return SimpleTokenizer(df["text"].astype(str).tolist())
        raise FileNotFoundError(
            "Tokenizer file not found and fallback data unavailable. Run dataset preparation/training first."
        )

    def _resolve_checkpoint(self) -> Path:
        if self.model_path.exists():
            return self.model_path
        if DEFAULT_LAST_MODEL_PATH.exists():
            return DEFAULT_LAST_MODEL_PATH
        raise FileNotFoundError(
            "No trained model checkpoint found. Train the model before launching the UI."
        )

    def _load_model(self) -> SmallTransformerClassifier:
        checkpoint_path = self._resolve_checkpoint()
        state = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state:
            hyperparams = state.get("model_hyperparams")
            if hyperparams is None:
                config = state.get("config", {})
                hyperparams = {
                    "embed_dim": config.get("embed_dim", 128),
                    "num_heads": config.get("num_heads", 2),
                    "num_layers": config.get("num_layers", 2),
                    "dropout": config.get("dropout", 0.1),
                    "max_len": config.get("max_len", getattr(self.tokenizer, "max_len", 128)),
                }
            embed_dim = int(hyperparams.get("embed_dim", 128))
            num_heads = int(hyperparams.get("num_heads", 2))
            num_layers = int(hyperparams.get("num_layers", 2))
            dropout = float(hyperparams.get("dropout", 0.1))
            max_len = int(hyperparams.get("max_len", getattr(self.tokenizer, "max_len", 128)))
            self.model_hparams = {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "max_len": max_len,
            }
            self.tokenizer.max_len = max_len
            model_state = state["model_state_dict"]
        else:
            # Backwards compatibility with bare state dicts
            embed_dim = 128
            num_heads = 2
            num_layers = 2
            dropout = 0.1
            max_len = getattr(self.tokenizer, "max_len", 128)
            self.model_hparams = {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "max_len": max_len,
            }
            model_state = state

        model = SmallTransformerClassifier(
            vocab_size=self.tokenizer.vocab_size(),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )
        model.load_state_dict(model_state)
        model.to(self.device)
        return model

    def predict_proba(self, text: str) -> Dict[str, float]:
        sanitized = text.strip()
        if not sanitized:
            return {"Not hate speech": 1.0, "Hate speech": 0.0}

        encoded = torch.tensor([self.tokenizer.encode(sanitized)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(encoded)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return {
            "Not hate speech": float(probs[0]),
            "Hate speech": float(probs[1]),
        }

    def classify(self, text: str) -> Dict[str, object]:
        probs = self.predict_proba(text)
        hate_prob = probs["Hate speech"]
        label = "Hate speech" if hate_prob >= 0.5 else "Not hate speech"
        return {
            "label": label,
            "probabilities": probs,
            "confidence": hate_prob if label == "Hate speech" else probs["Not hate speech"],
        }


def build_interface(pipeline: HateSpeechPipeline) -> gr.Blocks:
    def inference_fn(text: str) -> Dict[str, float]:
        return pipeline.predict_proba(text)

    def summary_fn(text: str) -> str:
        classification = pipeline.classify(text)
        probs = classification["probabilities"]
        hate_pct = probs["Hate speech"] * 100
        not_hate_pct = probs["Not hate speech"] * 100
        label = classification["label"]
        if not text.strip():
            return "Please enter some text to analyze."
        return (
            f"**Prediction:** {label}\n\n"
            f"- Hate speech probability: {hate_pct:.2f}%\n"
            f"- Not hate speech probability: {not_hate_pct:.2f}%"
        )

    with gr.Blocks(title="Filipino Hate Speech Detector") as demo:
        gr.Markdown(
            """
            # 🇵🇭 Filipino Hate Speech Detector

            Enter a snippet of text to evaluate whether it contains hate speech. The model shown here loads the
            best-performing checkpoint saved under `models/processed/sentiment_transformer.pt`.
            """
        )

        with gr.Row():
            text_input = gr.Textbox(
                label="Input text",
                placeholder="Type a message in Filipino or English...",
                lines=4,
            )

        with gr.Row():
            prediction_label = gr.Label(label="Class probabilities")

        summary = gr.Markdown()
        analyze_button = gr.Button("Analyze", variant="primary")

        analyze_button.click(
            fn=inference_fn,
            inputs=text_input,
            outputs=prediction_label,
        ).then(
            fn=summary_fn,
            inputs=text_input,
            outputs=summary,
        )

        gr.Examples(
            examples=[
                "Mahal kita kahit ano pa ang sabihin nila.",
                "Ang pangit mo, wala kang kwenta!",
                "Respeto lang sa kapwa, hindi mahirap yun.",
            ],
            inputs=text_input,
        )

        gr.Markdown(
            """
            **Note:** This model was trained on hate-speech datasets and may reflect their biases. Always review the
            predictions before acting on them.
            """
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Gradio hate speech detection UI")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--fallback-data", type=Path, default=DEFAULT_FALLBACK_DATA)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host/IP for the Gradio server")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server")
    parser.add_argument(
        "--test", type=str, default=None, metavar="TEXT", help="Run a one-off prediction and exit without launching the UI"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = HateSpeechPipeline(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        fallback_data_path=args.fallback_data,
    )

    if args.test is not None:
        result = pipeline.classify(args.test)
        probs = result["probabilities"]
        print(
            f"Input: {args.test}\nPrediction: {result['label']}\n"
            f"Hate speech probability: {probs['Hate speech']:.4f}\n"
            f"Not hate speech probability: {probs['Not hate speech']:.4f}"
        )
        return

    demo = build_interface(pipeline)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
