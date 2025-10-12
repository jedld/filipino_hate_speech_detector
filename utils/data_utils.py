import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    def __init__(self, texts: Optional[List[str]] = None, max_len: int = 64, word2idx: Optional[Dict[str, int]] = None):
        self.max_len = max_len
        if word2idx is not None:
            self.word2idx = word2idx
        else:
            self.word2idx = {'<PAD>': 0, '<UNK>': 1}
            idx = 2
            texts = texts or []
            for text in texts:
                for word in text.lower().split():
                    if word not in self.word2idx:
                        self.word2idx[word] = idx
                        idx += 1

    def encode(self, text: str):
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.lower().split()]
        tokens = tokens[:self.max_len]
        tokens += [self.word2idx['<PAD>']] * (self.max_len - len(tokens))
        return tokens

    def vocab_size(self):
        return len(self.word2idx)

    @property
    def pad_token_id(self) -> int:
        return self.word2idx['<PAD>']

    def to_dict(self) -> Dict[str, object]:
        return {'max_len': self.max_len, 'word2idx': self.word2idx}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(texts=None, max_len=data['max_len'], word2idx=data['word2idx'])

class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer: SimpleTokenizer):
        df = pd.read_csv(csv_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokenizer.encode(self.texts[idx]), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
