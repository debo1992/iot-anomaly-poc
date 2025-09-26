# evaluate_metrics.py
import time

import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize




def log_loss_accuracy(epoch, train_loss, train_acc, val_loss, val_acc):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)



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

    mlflow.log_figure(fig, artifact_name or default_name)
    plt.close(fig)



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

    # Log to MLflow
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)