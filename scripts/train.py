import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import SmallTransformerClassifier
from utils.data_utils import SimpleTokenizer, SentimentDataset
import pandas as pd
import random
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    data_path = 'data/sample_data.csv'
    df = pd.read_csv(data_path)
    tokenizer = SimpleTokenizer(df['text'].tolist())
    dataset = SentimentDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallTransformerClassifier(tokenizer.vocab_size()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), 'models/sentiment_transformer.pt')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    main()
