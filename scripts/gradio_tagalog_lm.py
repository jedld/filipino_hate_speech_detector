import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models.language_model import MiniTransformerLanguageModel

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "language_model" / "tagalog_lm.pt"
DEFAULT_ALT_CHECKPOINT = PROJECT_ROOT / "models" / "tagalog_lm" / "tagalog_lm.pt"
DEFAULT_FALLBACK_CHECKPOINT = PROJECT_ROOT / "models" / "language_model_test" / "tagalog_lm.pt"


class LanguageModelPipeline:
    """Loads the Tagalog transformer language model for text generation."""

    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Optional[Path] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.state = torch.load(self.checkpoint_path, map_location=self.device)
        self.tokenizer_metadata: Dict[str, object] = {}

        if "model_state_dict" not in self.state:
            raise ValueError("Checkpoint does not contain model_state_dict. Re-train the language model.")

        self.config = self.state.get("config", {})
        self.training_args = self.state.get("training_args", {})

        self.tokenizer = self._load_tokenizer(tokenizer_path, tokenizer_name)
        self.model = self._load_model()
        self.model.eval()

    def _resolve_checkpoint(self, requested: Path) -> Path:
        if requested.exists():
            return requested
        if DEFAULT_CHECKPOINT.exists():
            return DEFAULT_CHECKPOINT
        if DEFAULT_ALT_CHECKPOINT.exists():
            return DEFAULT_ALT_CHECKPOINT
        if DEFAULT_FALLBACK_CHECKPOINT.exists():
            return DEFAULT_FALLBACK_CHECKPOINT
        raise FileNotFoundError(
            "No language-model checkpoint found. Train the model before launching the generator UI."
        )

    def _load_tokenizer(
        self,
        tokenizer_path: Optional[Path],
        tokenizer_name: Optional[str],
    ) -> PreTrainedTokenizerBase:
        path_candidates = []
        if tokenizer_path is not None:
            path_candidates.append(tokenizer_path)
        checkpoint_tokenizer_path = self.state.get("tokenizer_path")
        if checkpoint_tokenizer_path:
            path_candidates.append(Path(checkpoint_tokenizer_path))
        checkpoint_dir = self.checkpoint_path.parent / "tokenizer"
        path_candidates.append(checkpoint_dir)

        for candidate in path_candidates:
            candidate_path = candidate
            if not candidate_path.is_absolute():
                candidate_path = PROJECT_ROOT / candidate_path
            if candidate_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(candidate_path)
                self._apply_tokenizer_metadata(tokenizer, candidate_path)
                return tokenizer

        tokenizer_identifier = (
            tokenizer_name
            or self.config.get("tokenizer_name")
            or self.training_args.get("tokenizer_name")
        )
        if tokenizer_identifier:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
            self._apply_config_overrides(tokenizer)
            return tokenizer

        raise FileNotFoundError(
            "Tokenizer could not be located. Specify --tokenizer-path or --tokenizer-name."
        )

    def _apply_tokenizer_metadata(self, tokenizer: PreTrainedTokenizerBase, directory: Path) -> None:
        metadata_path = directory / "training_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(metadata, dict):
                    self.tokenizer_metadata = metadata
                    max_len = metadata.get("model_max_length")
                    if isinstance(max_len, int) and max_len > 0:
                        self._set_tokenizer_max_length(tokenizer, max_len)
                    padding_side = metadata.get("padding_side")
                    if isinstance(padding_side, str):
                        tokenizer.padding_side = padding_side
                    truncation_side = metadata.get("truncation_side")
                    if isinstance(truncation_side, str):
                        tokenizer.truncation_side = truncation_side
                    if metadata.get("byte_level_decoder"):
                        self._ensure_byte_level_decoder(tokenizer)
            except Exception:
                pass
        else:
            self._apply_config_overrides(tokenizer)

    def _apply_config_overrides(self, tokenizer: PreTrainedTokenizerBase) -> None:
        max_len = self.config.get("tokenizer_model_max_length")
        if isinstance(max_len, int) and max_len > 0:
            self._set_tokenizer_max_length(tokenizer, max_len)
            self.tokenizer_metadata.setdefault("model_max_length", max_len)
        if self.config.get("tokenizer_trained_from_scratch"):
            self._ensure_byte_level_decoder(tokenizer)

    def _set_tokenizer_max_length(self, tokenizer: PreTrainedTokenizerBase, max_len: int) -> None:
        try:
            tokenizer.model_max_length = max_len
            init_kwargs = getattr(tokenizer, "init_kwargs", None)
            if isinstance(init_kwargs, dict):
                init_kwargs["model_max_length"] = max_len
        except Exception:
            pass

    def _ensure_byte_level_decoder(self, tokenizer: PreTrainedTokenizerBase) -> None:
        try:
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore

            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is not None:
                backend.decoder = ByteLevelDecoder()
        except Exception:
            pass

    def _tokenizer_vocab_size(self) -> int:
        try:
            return len(self.tokenizer)
        except TypeError:
            return len(self.tokenizer.get_vocab())

    def _load_model(self) -> MiniTransformerLanguageModel:
        vocab_size = self._tokenizer_vocab_size()
        embed_dim = int(self.config.get("embed_dim", 256))
        num_heads = int(self.config.get("num_heads", 4))
        num_layers = int(self.config.get("num_layers", 4))
        ff_multiplier = int(self.config.get("ff_multiplier", 4))
        dropout = float(self.config.get("dropout", 0.1))
        max_position_embeddings = int(self.config.get("max_position_embeddings", 512))

        model = MiniTransformerLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
        )
        model.load_state_dict(self.state["model_state_dict"])
        model.to(self.device)
        return model

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        sanitized = (prompt or "").strip()
        prompt_tokens = self.tokenizer.encode(sanitized, add_special_tokens=False, truncation=False)
        bos_token_id = self._bos_token_id()
        if not prompt_tokens:
            if bos_token_id is not None:
                prompt_tokens = [bos_token_id]
            else:
                raise ValueError("Unable to determine a starting token for generation; provide a non-empty prompt.")
        else:
            prompt_tokens = ([bos_token_id] if bos_token_id is not None else []) + prompt_tokens

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max(1, int(max_new_tokens)),
                temperature=max(float(temperature), 1e-5),
                top_k=max(int(top_k), 0),
            )
        text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        return text.strip()

    def _bos_token_id(self) -> Optional[int]:
        for attr in ("bos_token_id", "cls_token_id", "sep_token_id", "pad_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if value is not None:
                return value
        special_tokens = getattr(self.tokenizer, "special_tokens_map", {})
        if isinstance(special_tokens, dict):
            for key in ("bos_token", "cls_token", "sep_token", "pad_token"):
                token = special_tokens.get(key)
                if token is not None:
                    return self.tokenizer.convert_tokens_to_ids(token)
        return None


def build_interface(pipeline: LanguageModelPipeline, default_max_tokens: int, default_temperature: float, default_top_k: int) -> gr.Blocks:
    def generate_fn(prompt: str, max_tokens: int, temperature: float, top_k: int) -> str:
        return pipeline.generate_text(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    with gr.Blocks(title="Tagalog Language Model Playground") as demo:
        gr.Markdown(
            """
            # 🇵🇭 Tagalog Language Model Playground

            Provide a prompt in Filipino (or code-switching) and let the miniature Transformer continue the text. The
            model is the same one trained with `scripts/train_tagalog_lm.py`.
            """
        )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Halimbawa: Ang Pilipinas ay...",
                lines=4,
            )

        with gr.Row():
            max_tokens_input = gr.Slider(
                label="Max new tokens",
                minimum=10,
                maximum=512,
                step=10,
                value=default_max_tokens,
            )
            temperature_input = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=1.5,
                step=0.05,
                value=default_temperature,
            )
            top_k_input = gr.Slider(
                label="Top-k sampling (0 = disabled)",
                minimum=0,
                maximum=200,
                step=5,
                value=default_top_k,
            )

        output_box = gr.Textbox(label="Generated text", lines=8)
        generate_button = gr.Button("Generate", variant="primary")

        generate_button.click(
            fn=generate_fn,
            inputs=[prompt_input, max_tokens_input, temperature_input, top_k_input],
            outputs=output_box,
        )

        gr.Examples(
            examples=[
                "Ang Pilipinas ay kilala sa",
                "Noong unang panahon sa isang baryo,",
                "Sa hinaharap, ang teknolohiya ay magdudulot ng",
            ],
            inputs=prompt_input,
        )

        gr.Markdown(
            """
            **Tip:** Adjust temperature for more or less randomness (lower = safer, higher = more creative). Use top-k to
            limit sampling to the most likely tokens.
            """
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio text generation UI for the Tagalog language model")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu or cuda)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--default-max-tokens", type=int, default=100)
    parser.add_argument("--default-temperature", type=float, default=0.8)
    parser.add_argument("--default-top-k", type=int, default=20)
    parser.add_argument("--test", type=str, default=None, metavar="PROMPT", help="Run one generation and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else None
    pipeline = LanguageModelPipeline(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        tokenizer_name=args.tokenizer_name,
        device=device,
    )

    if args.test is not None:
        output = pipeline.generate_text(
            prompt=args.test,
            max_new_tokens=args.default_max_tokens,
            temperature=args.default_temperature,
            top_k=args.default_top_k,
        )
        print(output)
        return

    demo = build_interface(pipeline, args.default_max_tokens, args.default_temperature, args.default_top_k)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
