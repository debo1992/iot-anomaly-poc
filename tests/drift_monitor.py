"""
Data Drift Detection using Kolmogorov-Smirnov Test

This script compares feature distributions between a training dataset and a new incoming dataset
to identify potential data drift. It uses the two-sample Kolmogorov-Smirnov (KS) test on each feature.

- Inputs:
    - train_df: DataFrame containing the training data
    - new_df: DataFrame containing the new incoming data to compare against training data
    - feature_cols: List of feature column names to test for drift
    - alpha: Significance level for hypothesis testing (default 0.05)

- Output:
    - DataFrame with KS test statistic, p-value, and drift flag for each feature

- Logs the drift report JSON to MLflow for tracking.

Typical Usage:

drift_report = detect_drift(train_df, incoming_df, feature_cols)
print(drift_report)
mlflow.log_dict(drift_report.to_dict(), "drift_report.json")

"""

import pandas as pd
from scipy.stats import ks_2samp
import mlflow

def detect_drift(train_df, new_df, feature_cols, alpha=0.05):
    drift_report = {}
    for col in feature_cols:
        stat, pval = ks_2samp(train_df[col], new_df[col])
        drift_report[col] = {
            "statistic": stat,
            "pvalue": pval,
            "drift": pval < alpha
        }
    return pd.DataFrame(drift_report).T



