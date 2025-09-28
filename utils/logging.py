# evaluate_metrics.py
import time

import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, classification_report




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



def compute_classwise_metrics(all_labels, all_preds, ignore_class=0, verbose=True, log_mlflow=True, step=None):
    """
    Compute precision, recall, and F1 per class (optionally excluding class 0).
    Optionally log metrics to MLflow.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation
        device: 'cuda' or 'cpu'
        ignore_class: class label to ignore (default=0 for 'normal')
        verbose: if True, prints results to console
        log_mlflow: if True, logs metrics to MLflow
        step: optional logging step for MLflow

    Returns:
        results: dict with precision, recall, f1 per class and macro F1
    """

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    results = {}
    for cls, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        if cls == ignore_class:
            continue
        results[f"class_{cls}_precision"] = p
        results[f"class_{cls}_recall"] = r
        results[f"class_{cls}_f1"] = f
        results[f"class_{cls}_support"] = s

    # Macro F1 excluding ignored class
    non_zero_f1s = [results[k] for k in results if "_f1" in k]
    if non_zero_f1s:
        results["macro_f1_excl_normal"] = sum(non_zero_f1s) / len(non_zero_f1s)

    if verbose:
        print("\n📊 Class-wise Precision/Recall/F1 (ignoring class 0):")
        for k, v in results.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        print()

        # Optionally full classification report
        print("Detailed Classification Report (all classes):")
        print(classification_report(all_labels, all_preds))

    # 🔥 Log metrics to MLflow if enabled
    if log_mlflow:
        for k, v in results.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v), step=step)

    return results