import torch
from models.transformer import SmallTransformerClassifier
from utils.data_utils import SimpleTokenizer
import pandas as pd
import argparse
import os

def load_tokenizer(data_path):
    df = pd.read_csv(data_path)
    return SimpleTokenizer(df['text'].tolist())

def predict(text, model, tokenizer, device):
    model.eval()
    x = torch.tensor([tokenizer.encode(text)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    args = parser.parse_args()

    data_path = 'data/sample_data.csv'
    model_path = 'models/sentiment_transformer.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer(data_path)
    model = SmallTransformerClassifier(tokenizer.vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=device))
    import argparse
    from pathlib import Path

    import pandas as pd
    import torch

    from models.transformer import SmallTransformerClassifier
    from utils.data_utils import SimpleTokenizer


    def load_tokenizer(tokenizer_path: Path, fallback_data_path: Path, max_len: int) -> SimpleTokenizer:
        if tokenizer_path and tokenizer_path.exists():
            return SimpleTokenizer.load(tokenizer_path)
        df = pd.read_csv(fallback_data_path)
        return SimpleTokenizer(df['text'].tolist(), max_len=max_len)


    def load_model(model_path: Path, vocab_size: int, max_len: int, device: torch.device, dropout: float = 0.1):
        model = SmallTransformerClassifier(vocab_size=vocab_size, max_len=max_len, dropout=dropout)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.to(device)
        return model


    def predict(text: str, model: SmallTransformerClassifier, tokenizer: SimpleTokenizer, device: torch.device) -> int:
        model.eval()
        x = torch.tensor([tokenizer.encode(text)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
        return pred


    def main():
        parser = argparse.ArgumentParser(description='Run sentiment inference')
        parser.add_argument('--text', type=str, required=True, help='Text to analyze')
        parser.add_argument('--model-path', type=Path, default=Path('models/processed/sentiment_transformer.pt'))
        parser.add_argument('--tokenizer-path', type=Path, default=Path('models/processed/tokenizer.json'))
        parser.add_argument('--fallback-data-path', type=Path, default=Path('data/sample_data.csv'))
        parser.add_argument('--max-len', type=int, default=128, help='Maximum sequence length used during training')
        args = parser.parse_args()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = load_tokenizer(args.tokenizer_path, args.fallback_data_path, args.max_len)
        if not args.model_path.exists():
            print(f"Warning: {args.model_path} not found. Falling back to models/sentiment_transformer.pt")
            args.model_path = Path('models/sentiment_transformer.pt')

        ensure_path = args.model_path if args.model_path.exists() else None
        if ensure_path is None:
            raise FileNotFoundError("No model file available. Train a model first.")

        model = load_model(args.model_path, tokenizer.vocab_size(), tokenizer.max_len, device)
        pred = predict(args.text, model, tokenizer, device)
        print(f"Sentiment: {'positive' if pred == 1 else 'negative'}")


    if __name__ == '__main__':
        main()
