"""
Demonstrations scripts with requested visualizations
Key components

- Argument parsing:
Uses argparse to accept a config file, options to build the dataset if missing, and hyperparameter overrides like model type, learning rate, epochs, etc.

- Configuration loading:
Loads a default config, optionally overridden by a JSON config file and command-line arguments.

- Dataset preparation:
Builds datasets if missing, loads training and validation CSV files containing time-stamped anomaly data.

- Class weights calculation:
Computes class weights from training labels to handle class imbalance, used in the loss function.

- Model training (train_model function):

- Loads data into PyTorch datasets and dataloaders.

- Initializes the model, loss, optimizer, and device (CPU/GPU).

- Runs training and validation loops for configured epochs.

- Calculates class-wise precision, recall, and F1-score after each epoch.

- Implements early stopping based on validation loss.

- Saves the best performing model’s predictions for visualization.

- Visualization (plot_anomaly_vs_others):
Plots time-series binary curves comparing ground truth and predicted labels for each anomaly class against others.
Will generate the plot and save as timeseries_labels.png

"""

import os
import pandas as pd
import json
import argparse

import torch
from datasets.build_multiuser_datasets import build_multiuser_datasets
from datasets.anomaly_dataset import load_dataset
from models.initialize_model import my_model
from models.losses import my_loss
from utils.logging import compute_classwise_metrics
from utils.class_weight import get_class_weights
from models.training_eval_loops import training, evaluation


import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_vs_others(y_true, y_pred, num_classes=6):
    t = np.arange(len(y_true))  # time axis
    anomaly_classes = range(1, num_classes)  # classes 1–5

    fig, axes = plt.subplots(len(anomaly_classes), 1, figsize=(15, 12), sharex=True)

    for i, cls in enumerate(anomaly_classes):
        ax = axes[i]

        # Binarize for this anomaly vs others
        y_true_bin = [int(y == cls) for y in y_true]
        y_pred_bin = [int(y == cls) for y in y_pred]

        # Plot
        ax.plot(t, y_true_bin, label=f"Ground Truth (class {cls})", color="blue", linewidth=2)
        ax.plot(t, y_pred_bin, label=f"Predicted (class {cls})", color="red", linestyle="--", alpha=0.7)


        ax.set_ylabel(f"Class {cls}")
        ax.set_ylim(-0.2, 1.2)
        ax.legend(loc="upper right")

    plt.xlabel("Time (sample index)")
    plt.suptitle("Others vs Anomaly Classes (1–5)", fontsize=14)
    plt.tight_layout()
    plt.savefig("timeseries_labels.png")




def train_model(train_df, val_df, config):

    # Dataset
    train_dataset, _, train_loader, val_loader = load_dataset(config, train_df, val_df)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_model(config, input_dim=train_dataset.X.shape[2])
    criterion = my_loss(config, device=device)      
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.to(device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
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
            verbose=True, log_mlflow=False, step=epoch
        )

        print(f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # ---- Early Stopping ----
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_all_labels = all_labels
            best_all_preds = all_preds
            best_all_probs = all_probs
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered. No improvement in validation loss for {config['patience']} epochs.")
                break
        plot_anomaly_vs_others(best_all_labels, best_all_preds, num_classes=6)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")

    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--build-data", action="store_true", help="Build dataset if not exists")


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
    model = train_model(train_df, val_df, config)