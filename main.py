# main.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

import mlflow
import wandb
wandb.login(key="635b31fb4fd119939505dad031a9f425aabe1747")

from datasets.anomaly_dataset import AnomalyDataset
from models.lstm_basic import LSTMAnomalyClassifier
from models.cnn_basic import CNNAnomalyClassifier
from models.transformer_basic import TransformerAnomalyClassifier
from models.tcn_basic import TCNAnomalyClassifier
from utils.evaluation_metrics import compute_classwise_metrics
from utils.logging import log_intialize, log_loss_accuracy




def train_model(train_df, val_df, config, log = True):
    run_name = config["model_type"] + "_" + f"run_{int(time.time())}"
    if log:
    # Initialize W&B
        log_intialize(run_name, project_name = "iot-anomaly-detection",  config = None)

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
    best_val_loss = float("inf")
    epochs_without_improvement = 0  # Counter to track epochs without improvement
    best_model_state_dict = None  # To store the model's best state
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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        compute_classwise_metrics(all_labels, all_preds, ignore_class=0, verbose=True, log_mlflow=False, step=None)

        if log:
            log_loss_accuracy(epoch, train_loss, train_acc, val_loss, val_acc)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state_dict = model.state_dict()  # Save the best model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered. No improvement in validation loss for {config['patience']} epochs.")
                break
        # Restore the best model state (based on validation loss)
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)

        # Optional: Save model
        os.makedirs("outputs/models", exist_ok=True)
        model_path = f"outputs/models/{run_name}_model.pt"
        torch.save(model.state_dict(), model_path)
    if log:
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
        "epochs": 100,
        "lr": 1e-3,
        "patience": 7
     }

        # Train model
        model = train_model(train_df, val_df, config, log = False)
