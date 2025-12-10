import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


"""Utility functions for data processing and tokenization."""
class SimpleTokenizer:
    """
    A simple whitespace tokenizer that builds a vocabulary from given texts.
    It can encode texts into sequences of token IDs and handle padding.
    """
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


class CharTokenizer:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(
        self,
        texts: Optional[Iterable[str]] = None,
        char2idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if char2idx is not None:
            # Ensure keys are strings and values are ints
            self.char2idx = {str(k): int(v) for k, v in char2idx.items()}
        else:
            base_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
            self.char2idx = {token: idx for idx, token in enumerate(base_tokens)}
            idx = len(self.char2idx)
            texts = texts or []
            for text in texts:
                for ch in str(text):
                    if ch not in self.char2idx:
                        self.char2idx[ch] = idx
                        idx += 1
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    @property
    def pad_token_id(self) -> int:
        return self.char2idx[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.char2idx[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.char2idx[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.char2idx[self.EOS_TOKEN]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [self.char2idx.get(ch, self.unk_token_id) for ch in str(text)]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        chars: List[str] = []
        for token_id in token_ids:
            token = int(token_id)
            if skip_special_tokens and token in special:
                continue
            chars.append(self.idx2char.get(token, self.UNK_TOKEN))
        return ''.join(chars)

    def vocab_size(self) -> int:
        return len(self.char2idx)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {'char2idx': self.char2idx}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(texts=None, char2idx=data['char2idx'])

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
