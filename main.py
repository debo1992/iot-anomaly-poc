# main.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow
import wandb
wandb.login(key="635b31fb4fd119939505dad031a9f425aabe1747")

from datasets.anomaly_dataset import AnomalyDataset
from models.lstm_basic import LSTMAnomalyClassifier
from models.cnn_basic import CNNAnomalyClassifier
from models.transformer_basic import TransformerAnomalyClassifier
from models.tcn_basic import TCNAnomalyClassifier




def train_model(train_df, val_df, config):
    # Initialize W&B

    wandb.init(project="iot-anomaly-detection", config=config)
    config = wandb.config

    # Set up MLflow
    mlflow.set_experiment("IoT_Anomaly_Models")
    with mlflow.start_run(run_name=config["model_type"]):
        mlflow.log_params(dict(config))

        # Dataset
        train_dataset = AnomalyDataset(train_df, config["window_size"])
        val_dataset = AnomalyDataset(val_df, config["window_size"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        # Model
        if config["model_type"] == "LSTM":
            model = LSTMAnomalyClassifier()
        elif config["model_type"] == "CNN":
            model = CNNAnomalyClassifier()
        elif config["model_type"] == "TRANSFORMER":
            model = TransformerAnomalyClassifier()
        elif config["model_type"] == "TCN":
            model = TCNAnomalyClassifier()
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(config["epochs"]):
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
            train_loss = total_loss / len(train_loader)
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
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total

            # Logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            print(f"Epoch {epoch+1}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Optional: Save model
        os.makedirs("outputs/models", exist_ok=True)
        model_path = f"outputs/models/{config['model_type'].lower()}_model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        wandb.save(model_path)

        wandb.finish()

    return model


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])

    for model in ["LSTM", "CNN", "TRANSFORMER", "TCN"]:
        config = {
        "model_type": model,  
        "window_size": 12,
        "batch_size": 64,
        "epochs": 20,
        "lr": 1e-3
     }

        # Train model
        model = train_model(train_df, val_df, config)
