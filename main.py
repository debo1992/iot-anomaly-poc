# main.py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.anomaly_dataset import AnomalyDataset
from models.lstm_basic import LSTMAnomalyClassifier

def train_model(train_df, val_df, window_size=12, batch_size=64, num_epochs=20, lr=1e-3):
    train_dataset = AnomalyDataset(train_df, window_size)
    val_dataset = AnomalyDataset(val_df, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMAnomalyClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = 100 * correct / total

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    return model

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])

    # Train model
    model = train_model(train_df, val_df)
