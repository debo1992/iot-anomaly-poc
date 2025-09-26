# utils/metrics.py
import torch
import mlflow
from sklearn.metrics import precision_recall_fscore_support, classification_report

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
