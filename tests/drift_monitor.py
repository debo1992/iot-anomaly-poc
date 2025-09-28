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

drift_report = detect_drift(train_df, incoming_df, feature_cols)
print(drift_report)
mlflow.log_dict(drift_report.to_dict(), "drift_report.json")
