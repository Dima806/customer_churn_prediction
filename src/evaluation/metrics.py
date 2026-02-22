"""Comprehensive evaluation metrics for the churn model."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Compute the full set of churn model evaluation metrics.

    Args:
        y_true: Ground-truth binary labels (0 / 1).
        y_prob: Predicted churn probabilities in [0, 1].
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dict containing roc_auc, pr_auc, brier_score, precision, recall, f1.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob)),
        "precision": float(
            precision_score(y_true_arr, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "threshold": threshold,
        "n_samples": int(len(y_true_arr)),
        "churn_rate": float(y_true_arr.mean()),
    }
