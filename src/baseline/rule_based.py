"""Rule-based churn baseline model.

Rule (from PRD §7):
    IF recency > CHURN_PERIOD_DAYS
    OR orders_last_60d == 0
    THEN is_churned = 1

For continuous-score metrics (ROC-AUC, PR-AUC), we derive a soft score
from normalised recency so the model is comparable with probability outputs.
"""

import pandas as pd

from src.config import CHURN_PERIOD_DAYS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_churn_rule_based(
    features: pd.DataFrame,
    churn_period_days: int = CHURN_PERIOD_DAYS,
) -> tuple[pd.Series, pd.Series]:
    """Apply the rule-based churn heuristic.

    Args:
        features: Feature matrix that must contain ``recency`` and
                  (optionally) ``orders_last_60d``.
        churn_period_days: Threshold for the recency rule.

    Returns:
        Tuple of:
        - ``y_pred``  – binary prediction Series (0 / 1).
        - ``y_score`` – continuous churn score in [0, 1] suitable for
                        ROC-AUC / PR-AUC computation.
    """
    orders_60d = features.get("orders_last_60d", pd.Series(1, index=features.index, dtype=int))

    y_pred = ((features["recency"] > churn_period_days) | (orders_60d == 0)).astype(int)

    # Continuous score: normalised recency, boosted when 60-day window is empty
    max_recency = max(float(features["recency"].max()), 1.0)
    y_score = (features["recency"] / max_recency).clip(0.0, 1.0)
    y_score = y_score.where(orders_60d > 0, other=y_score.clip(lower=0.70))

    return y_pred, y_score


def evaluate_baseline(
    features: pd.DataFrame,
    target: pd.DataFrame,
    churn_period_days: int = CHURN_PERIOD_DAYS,
) -> dict:
    """Compute evaluation metrics for the rule-based baseline.

    Args:
        features: Feature matrix with ``customer_id`` column.
        target: Target DataFrame with ``customer_id`` and ``is_churned``.
        churn_period_days: Churn recency threshold.

    Returns:
        Dict with precision, recall, roc_auc, pr_auc.
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    df = features.merge(target, on="customer_id", how="inner")
    y_true = df["is_churned"]
    y_pred, y_score = predict_churn_rule_based(df, churn_period_days)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }

    logger.info(
        f"Baseline  ROC-AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}"
    )
    return metrics
