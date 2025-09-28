import pandas as pd
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.initialize_model import my_model
from datasets.anomaly_dataset import load_dataset
from utils.class_weight import get_class_weights
from sklearn.preprocessing import label_binarize

#  CNN_run_1758931464_model

def load_model_from_mlflow(artifact_path="CNN_run_1758931464_model.pt"):
    # Replace with your actual run ID   
    run_id = "724a77bb90034332bdb123578c23d6da"
    run = mlflow.get_run(run_id)
    config = run.data.params
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    model = my_model(config, input_dim=7)   # must match training
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict)
    for key in config:
        if key in ["epochs", "batch_size", "num_classes", "hidden_dim", "window_size","patience"]:
            config[key] = int(config[key])
        
    return config, model, artifact_path



def prediction_model(model, val_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return all_labels, all_preds, all_probs

def log_confusion_matrix(y_true, y_pred, class_names, normalized=False, artifact_name="confusion_matrix.png"):
    """
    Logs a confusion matrix to MLflow.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        class_names (list): Names of classes
        normalized (bool): If True, row-normalizes the confusion matrix
        artifact_name (str): MLflow artifact name (default auto)
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    if normalized:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # handle divide-by-zero if class missing
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
        default_name = "normalized_confusion_matrix.png"
    else:
        fmt = "d"
        title = "Confusion Matrix"
        default_name = "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("conf_mat.png", dpi=300)
    plt.close()

def log_pr_curves(y_true, y_pred_probs, class_names, artifact_name="pr_curves.png"):
    """
    Logs per-class Precision-Recall curves to MLflow.

    Args:
        y_true (array-like): True labels (ints, shape [n_samples])
        y_pred_probs (array-like): Predicted probabilities (shape [n_samples, n_classes])
        class_names (list): List of class names
        artifact_name (str): File name for MLflow artifact
    """
    n_classes = len(class_names)

    # Binarize true labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_probs = np.array(y_pred_probs)

    # Plot PR curve for each class
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        if y_true_bin[:, i].sum() == 0:
            continue  # skip classes not present in y_true
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
        ax.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("pr.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])
    config, model = load_model_from_mlflow()
    train_dataset, val_dataset, _, val_loader = load_dataset(config, train_df, val_df)
    labels, predicted_class, predicted_probabilities = prediction_model(model, val_loader)
    log_confusion_matrix(labels, predicted_class,
                     class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], normalized=True)
    log_pr_curves(labels, predicted_probabilities, class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], artifact_name="pr_curves.png")

   
    
