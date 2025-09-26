# main.py
import os
import pandas as pd
import time
import mlflow

import torch

from torch.utils.data import DataLoader
from datasets.anomaly_dataset import AnomalyDataset, make_balanced_loader
from models.initialize_model import my_model, my_loss
from utils.evaluation_metrics import compute_classwise_metrics
from utils.logging import log_loss_accuracy, log_confusion_matrix, log_pr_curves


def train_model(train_df, val_df, config, log=True):
    run_name = config["model_type"] + "_" + f"run_{int(time.time())}"
    if log:
        mlflow.set_experiment("IoT_Anomaly_Models")
        mlflow.set_tracking_uri("http://127.0.0.1/:5000")

        with mlflow.start_run(run_name = run_name):
            mlflow.log_params(dict(config))


            # Dataset
            train_dataset = AnomalyDataset(train_df, config["window_size"])
            val_dataset = AnomalyDataset(val_df, config["window_size"])

            # ✅ Use balanced sampler for training
            if config.get("balanced_loader", False):
                train_loader = make_balanced_loader(train_dataset, batch_size=config["batch_size"])
            else:
                train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

            # Model
            model = my_model(config)
            criterion = my_loss(config)        
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            best_val_loss = float("inf")
            epochs_without_improvement = 0
            best_model_state_dict = None

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
                all_preds, all_labels, all_probs = [], [], []
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y.cpu().numpy())
                        all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                        val_correct += (preds == y).sum().item()
                        val_total += y.size(0)

                val_loss /= len(val_loader)
                val_acc = 100 * val_correct / val_total

                # Compute per-class precision/recall/f1
                compute_classwise_metrics(
                    all_labels, all_preds, ignore_class=0,
                    verbose=True, log_mlflow=True, step=epoch
                )

                if log:
                    log_loss_accuracy(epoch, train_loss, train_acc, val_loss, val_acc)

                print(f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # ---- Early Stopping ----
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_model_state_dict = model.state_dict()
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= config['patience']:
                        print(f"Early stopping triggered. No improvement in validation loss for {config['patience']} epochs.")
                        break
                # log_confusion_matrix(all_labels, all_preds, class_names=["Normal", "Type1", "Type2", "Type3", "Type4", "Type5"], normalized=True)
                # log_pr_curves(all_labels, all_probs, class_names=["Normal", "Type1", "Type2", "Type3", "Type4", "Type5"], artifact_name="pr_curves.png")
            # Restore best model
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

            # Save model
            os.makedirs("outputs/models", exist_ok=True)
            model_path = f"outputs/models/{run_name}_model.pt"
            torch.save(model.state_dict(), model_path)

            if log:
                log_confusion_matrix(all_labels, all_preds,
                     class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], normalized=True)
                log_pr_curves(all_labels, all_probs, class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], artifact_name="pr_curves.png")

                mlflow.log_artifact(model_path)

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
                "epochs": 60,
                "lr": 1e-3,
                "patience": 7,
                "balanced_loader": True,
                "loss_type": "cross_entropy",  # could be "focal", "weighted_ce", etc.
                "class_weights": None           # optional for weighted CE / focal
            }

        model = train_model(train_df, val_df, config, log=True)
