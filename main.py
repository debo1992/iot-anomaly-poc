# main.py
import os
import pandas as pd
import time
import mlflow
import json
import argparse

import torch
from datasets.build_multiuser_datasets import build_multiuser_datasets
from datasets.anomaly_dataset import load_dataset
from models.initialize_model import my_model
from tests.load_eval_model import load_model_from_mlflow
from models.losses import my_loss
from utils.logging import log_loss_accuracy, log_confusion_matrix, log_pr_curves, compute_classwise_metrics
from utils.class_weight import get_class_weights
from models.training_eval_loops import training, evaluation




def train_model(train_df, val_df, config, log=True):
    run_name = config["model_type"] + "_" + f"run_{int(time.time())}"
    if log:
        mlflow.set_experiment("IoT_Anomaly_Models")

        with mlflow.start_run(run_name = run_name):
            mlflow.log_params(dict(config))

            # Dataset
            train_dataset, _, train_loader, val_loader = load_dataset(config, train_df, val_df)

            # Model setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = my_model(config, input_dim=train_dataset.X.shape[2])
            criterion = my_loss(config, device=device)      
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            model.to(device)

            best_val_loss = float("inf")
            best_val_f1 = 0.0
            epochs_without_improvement = 0
            best_model_state_dict = None
            best_all_labels = None
            best_all_preds = None
            best_all_probs = None

            for epoch in range(config["epochs"]):
                # ---- Training ----
                train_loss, train_acc = training(model, train_loader, criterion, optimizer, device)

                # ---- Validation ----
                val_loss, val_acc, all_labels, all_preds, all_probs = evaluation(model, val_loader, criterion, device)

                # Compute per-class precision/recall/f1
                results = compute_classwise_metrics(
                    all_labels, all_preds, ignore_class=0,
                    verbose=True, log_mlflow=True, step=epoch
                )

                if log:
                    log_loss_accuracy(epoch, train_loss, train_acc, val_loss, val_acc)

                print(f"Epoch {epoch+1}/{config['epochs']} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # ---- Early Stopping ----
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_model_state_dict = model.state_dict()
                    best_all_labels = all_labels
                    best_all_preds = all_preds
                    best_all_probs = all_probs
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= config['patience']:
                        print(f"Early stopping triggered. No improvement in validation loss for {config['patience']} epochs.")
                        break
                
            # Restore best model
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

            # Save model
            os.makedirs("outputs/models", exist_ok=True)
            model_path = f"outputs/models/{run_name}_model.pt"
            torch.save(model.state_dict(), model_path)

            if log:
                log_confusion_matrix(best_all_labels, best_all_preds,
                     class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], normalized=True,
                     artifact_name="cf_matrix_"+run_name+".png")
                log_pr_curves(best_all_labels, best_all_probs, class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], artifact_name="pr_"+run_name+".png")
                mlflow.log_artifact(model_path)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")

    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--build-data", action="store_true", help="Build dataset if not exists")
    parser.add_argument("--log", action="store_false", help="Enable MLflow logging")

    # Optional overrides
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--balanced_loader", action="store_true")
    parser.add_argument("--loss_type", type=str)

    return parser.parse_args()


def load_config(config_path=None):
    default_config = {
        "model_type": "CNN",
        "window_size": 32,
        "batch_size": 64,
        "epochs": 10,
        "lr": 0.001,
        "patience": 5,
        "balanced_loader": False,
        "loss_type": "weighted_ce"
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = json.load(f)
        default_config.update(file_config)
    else:
        print("⚠️ No config file provided or found. Using default config.")

    return default_config


def override_config(config, args):
    for key in config.keys():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            config[key] = arg_val
    return config


if __name__ == "__main__":
    args = parse_args()

    # Check dataset
    if args.build_data or not os.path.exists('datasets/data'):
        print("📦 Building dataset...")
        build_multiuser_datasets()

    # Load and merge config
    config = load_config(args.config)
    config = override_config(config, args)

    # Load data
    train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])

    train_labels = train_df['anomaly_class'].values
    class_weights, _ = get_class_weights(train_labels)
    config["class_weights"] = class_weights

    print(f"🚀 Training model: {config['model_type']}")
    model = train_model(train_df, val_df, config, log=args.log)