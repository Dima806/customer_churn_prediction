"""Hyperparameter tuning via random search.

Run as a module:
    uv run python -m src.models.tune
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import ARTIFACTS_DIR, DATA_DIR, SEED
from src.feature_engineering.pipeline import build_feature_matrix
from src.models.train import time_based_split
from src.preprocessing.clean import clean_customers, clean_orders
from src.target.churn_target import compute_churn_target
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Search space
_PARAM_SPACE: dict = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
    "subsample": [0.70, 0.80, 0.90],
    "colsample_bytree": [0.70, 0.80, 0.90],
    "min_child_weight": [1, 3, 5],
}


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
    seed: int = SEED,
) -> dict:
    """Random-search hyperparameter tuning with stratified CV.

    Args:
        X_train: Training feature matrix.
        y_train: Binary training labels.
        n_trials: Number of random parameter configurations to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Best hyperparameter dict found.
    """
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    best_score = -float("inf")
    best_params: dict = {}

    logger.info(f"Starting random search: {n_trials} trials, 3-fold CV …")

    for trial in range(1, n_trials + 1):
        params = {
            "n_estimators": int(rng.choice(_PARAM_SPACE["n_estimators"])),
            "max_depth": int(rng.choice(_PARAM_SPACE["max_depth"])),
            "learning_rate": float(rng.choice(_PARAM_SPACE["learning_rate"])),
            "subsample": float(rng.choice(_PARAM_SPACE["subsample"])),
            "colsample_bytree": float(rng.choice(_PARAM_SPACE["colsample_bytree"])),
            "min_child_weight": int(rng.choice(_PARAM_SPACE["min_child_weight"])),
            "eval_metric": "auc",
            "random_state": seed,
        }

        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        score = float(scores.mean())

        if score > best_score:
            best_score = score
            best_params = params
            logger.info(f"  Trial {trial:3d}: ROC-AUC={score:.4f}  *** new best ***")
        else:
            logger.info(f"  Trial {trial:3d}: ROC-AUC={score:.4f}")

    logger.info(f"Best ROC-AUC: {best_score:.4f}")
    logger.info(f"Best params:  {best_params}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "best_params.json", "w") as fh:
        json.dump({"params": best_params, "score": best_score}, fh, indent=2)

    logger.info(f"Best params saved → {ARTIFACTS_DIR / 'best_params.json'}")
    return best_params


if __name__ == "__main__":
    logger.info("=== Hyperparameter Tuning ===")

    orders_raw = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date", "contract_date"])
    customers_raw = pd.read_csv(
        DATA_DIR / "customers.csv",
        parse_dates=["registration_date", "birth_date", "last_profile_update"],
    )

    orders = clean_orders(orders_raw)
    customers = clean_customers(customers_raw)

    target_df, feature_end_date, _ = compute_churn_target(orders, customers)
    features = build_feature_matrix(orders, customers, feature_end_date)

    X_train, y_train, _, _, _ = time_based_split(features, target_df, customers)
    tune_hyperparameters(X_train, y_train)

    logger.info("=== Tuning complete ===")
