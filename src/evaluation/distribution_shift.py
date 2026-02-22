"""Adversarial validation: detect covariate shift between train and test splits.

An adversarial classifier is trained to separate train (label 0) from test
(label 1) samples.  A classifier that barely beats chance (ROC-AUC ≈ 0.5)
means the two splits are drawn from the same distribution.  A high AUC
signals systematic covariate shift that can compromise hold-out evaluation.

Per-feature Kolmogorov–Smirnov statistics surface the features contributing
most to any detected shift.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOP_FEATURES: int = 5  # most-shifted features to include in the log line


def compute_adversarial_auc(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = SEED,
) -> dict:
    """Measure distributional similarity between train and test splits.

    Builds an adversarial dataset (train → label 0, test → label 1) and
    measures how well a random-forest can tell the two apart via 5-fold
    stratified cross-validation.

    Args:
        X_train: Training feature matrix (no target column).
        X_test: Test feature matrix (same columns as X_train).
        n_estimators: Trees in the adversarial random forest.
        random_state: Seed for reproducibility.

    Returns:
        Dict with:
        - adversarial_auc (float): mean CV ROC-AUC.  0.5 = no detectable
          shift; 1.0 = splits are completely separable.
        - feature_ks (dict[str, float]): per-feature KS statistic in [0, 1].
          Higher values indicate greater distributional divergence.
    """
    from scipy.stats import ks_2samp

    # ---- adversarial labels: train=0, test=1 --------------------------------
    X_adv = pd.concat([X_train, X_test], ignore_index=True)
    y_adv = np.concatenate(
        [np.zeros(len(X_train), dtype=np.int8), np.ones(len(X_test), dtype=np.int8)],
    )

    # ---- adversarial classifier ---------------------------------------------
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X_adv, y_adv, cv=cv, scoring="roc_auc", n_jobs=-1)
    adversarial_auc = float(scores.mean())

    # ---- per-feature KS statistics ------------------------------------------
    feature_ks = {
        col: float(ks_2samp(X_train[col].values, X_test[col].values).statistic)
        for col in X_train.columns
    }
    top = sorted(feature_ks.items(), key=lambda kv: kv[1], reverse=True)

    logger.info(
        f"Distribution shift  adversarial_auc={adversarial_auc:.4f}  "
        f"(0.50=no shift, 1.00=full shift)"
    )
    logger.info(
        "  Most shifted features (KS): " + "  ".join(f"{k}={v:.3f}" for k, v in top[:_TOP_FEATURES])
    )

    return {"adversarial_auc": adversarial_auc, "feature_ks": feature_ks}
