"""Hyperparameter tuning via random search with walk-forward temporal CV.

Run as a module:
    uv run python -m src.models.tune
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS_DIR, CHURN_PERIOD_DAYS, DATA_DIR, SEED
from src.feature_engineering.pipeline import build_feature_matrix
from src.models.train import time_based_split
from src.preprocessing.clean import clean_customers, clean_orders
from src.target.churn_target import compute_churn_target
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Search space — max_depth capped at 5, regularisation axes added
_PARAM_SPACE: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],  # was [3..7]: deeper trees overfit small datasets
    "learning_rate": [0.01, 0.03, 0.05, 0.10],
    "subsample": [0.70, 0.80, 0.90],
    "colsample_bytree": [0.70, 0.80, 0.90],
    "min_child_weight": [3, 5, 10],  # was [1,3,5]: floor raised to avoid tiny leaves
    "gamma": [0.0, 0.1, 0.3],  # min gain required to make a split
    "reg_alpha": [0.0, 0.1, 0.5],  # L1 regularisation on leaf weights
}


def tune_hyperparameters(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    train_customer_ids: list,
    n_splits: int = 3,
    n_trials: int = 30,
    seed: int = SEED,
    churn_period_days: int = CHURN_PERIOD_DAYS,
) -> dict:
    """Random-search hyperparameter tuning with walk-forward temporal CV.

    Unlike stratified k-fold, this method preserves temporal ordering: each
    fold trains on customers from earlier cohorts and evaluates on later ones.
    RFM and all other features are recomputed at each fold's reference date,
    so no information from the churn observation window or from "future" orders
    can leak into the feature values used during CV.

    Walk-forward CV structure (``n_splits`` folds, expanding window)::

        usable_start = min_order_date + churn_period_days
        usable_end   = max_order_date − churn_period_days
        step         = (usable_end − usable_start) / (n_splits + 1)

        Fold k  (k = 0 … n_splits−1):
          feature_end  = usable_start + (k+1) × step
          fold_t_max   = feature_end + churn_period_days
          eligible     = customers with ≥1 order strictly before feature_end
          train        = earliest 80 % of eligible (sorted by registration_date)
          val          = latest  20 % of eligible (sorted by registration_date)
          Features     recomputed on orders < feature_end  (zero future leakage)
          Labels       recomputed from activity in (feature_end, fold_t_max]

    Fold datasets are pre-computed once and reused across all trials so that
    the cost of feature recomputation is paid only ``n_splits`` times.

    Args:
        orders: Full cleaned orders fact table.
        customers: Cleaned customer dimension table.
        train_customer_ids: Customer IDs belonging to the training split.
        n_splits: Number of temporal CV folds.
        n_trials: Number of random parameter configurations to evaluate.
        seed: Random seed for reproducibility.
        churn_period_days: Length of the churn observation window in days.

    Returns:
        Best hyperparameter dict found.
    """
    rng = np.random.default_rng(seed)
    train_id_set = set(train_customer_ids)

    # Restrict to training customers to prevent any test-set contamination
    train_orders = orders[orders["customer_id"].isin(train_id_set)].copy()
    train_customers = customers[customers["customer_id"].isin(train_id_set)].copy()

    t_min = train_orders["order_date"].min()
    t_max_train = train_orders["order_date"].max()

    # Each fold needs churn_period_days of feature history *before* feature_end
    # and churn_period_days of observation time *after* feature_end for labels.
    usable_start = t_min + pd.Timedelta(days=churn_period_days)
    usable_end = t_max_train - pd.Timedelta(days=churn_period_days)

    if usable_end <= usable_start:
        raise ValueError(
            f"Training order-date range [{t_min.date()}, {t_max_train.date()}] is too "
            f"short for {n_splits}-fold temporal CV "
            f"(need > {2 * churn_period_days} days of orders)."
        )

    step = (usable_end - usable_start) / (n_splits + 1)
    fold_feature_ends = [usable_start + (k + 1) * step for k in range(n_splits)]

    logger.info(
        f"Walk-forward CV: {n_splits} folds, "
        f"feature_end dates=[{', '.join(str(d.date()) for d in fold_feature_ends)}]"
    )

    reg_dates = train_customers[["customer_id", "registration_date"]]

    # Pre-compute fold datasets once; reused across all hyperparameter trials
    fold_datasets: list[dict] = []

    for k, fold_feature_end in enumerate(fold_feature_ends):
        fold_t_max = fold_feature_end + pd.Timedelta(days=churn_period_days)

        # --- Labels anchored to this fold's observation window ---
        # Filtering train_orders to fold_t_max forces compute_churn_target to
        # derive T_max = fold_t_max, so churn_window_start = fold_feature_end.
        fold_target, _, _ = compute_churn_target(
            train_orders[train_orders["order_date"] <= fold_t_max],
            train_customers,
            churn_period_days,
        )

        # --- Features computed strictly before fold_feature_end ---
        # build_feature_matrix passes reference_date to every sub-module
        # (RFM, time features, trend features), all of which filter with
        # order_date < reference_date — zero leakage by construction.
        fold_features = build_feature_matrix(train_orders, train_customers, fold_feature_end)
        feature_cols = [c for c in fold_features.columns if c != "customer_id"]

        # Eligible customers = those present in both features AND labels,
        # i.e. they had ≥1 order before fold_feature_end (required by both).
        fold_df = (
            fold_features.merge(fold_target, on="customer_id", how="inner")
            .merge(reg_dates, on="customer_id", how="left")
            .sort_values("registration_date")
            .reset_index(drop=True)
        )

        # Temporal train / val split: train = earlier-registered 80 %,
        # val = later-registered 20 % — mirrors real deployment ordering.
        split_idx = int(len(fold_df) * 0.80)
        train_fold = fold_df.iloc[:split_idx]
        val_fold = fold_df.iloc[split_idx:]

        if len(train_fold) < 10 or len(val_fold) < 5:
            logger.warning(
                f"Fold {k + 1}/{n_splits}: insufficient eligible customers "
                f"(train={len(train_fold)}, val={len(val_fold)}) — skipping"
            )
            continue

        fold_datasets.append(
            {
                "X_train": train_fold[feature_cols].reset_index(drop=True),
                "y_train": train_fold["is_churned"].reset_index(drop=True),
                "X_val": val_fold[feature_cols].reset_index(drop=True),
                "y_val": val_fold["is_churned"].reset_index(drop=True),
            }
        )

        logger.info(
            f"  Fold {k + 1}/{n_splits}: "
            f"feature_end={fold_feature_end.date()}  t_max={fold_t_max.date()}  "
            f"eligible={len(fold_df):,}  "
            f"train={len(train_fold):,} (churn={train_fold['is_churned'].mean():.1%})  "
            f"val={len(val_fold):,} (churn={val_fold['is_churned'].mean():.1%})"
        )

    if not fold_datasets:
        raise RuntimeError(
            "No valid temporal CV folds could be constructed. "
            "Check that the training set covers a long enough time range."
        )

    logger.info(f"Starting random search: {n_trials} trials, {len(fold_datasets)} valid fold(s) …")

    best_score = -float("inf")
    best_params: dict = {}

    for trial in range(1, n_trials + 1):
        params = {
            "n_estimators": int(rng.choice(_PARAM_SPACE["n_estimators"])),
            "max_depth": int(rng.choice(_PARAM_SPACE["max_depth"])),
            "learning_rate": float(rng.choice(_PARAM_SPACE["learning_rate"])),
            "subsample": float(rng.choice(_PARAM_SPACE["subsample"])),
            "colsample_bytree": float(rng.choice(_PARAM_SPACE["colsample_bytree"])),
            "min_child_weight": int(rng.choice(_PARAM_SPACE["min_child_weight"])),
            "gamma": float(rng.choice(_PARAM_SPACE["gamma"])),
            "reg_alpha": float(rng.choice(_PARAM_SPACE["reg_alpha"])),
            "eval_metric": "auc",
            "random_state": seed,
        }

        fold_scores: list[float] = []
        for fold_data in fold_datasets:
            model = xgb.XGBClassifier(**params)
            model.fit(fold_data["X_train"], fold_data["y_train"], verbose=False)
            y_prob = model.predict_proba(fold_data["X_val"])[:, 1]
            try:
                fold_auc = float(roc_auc_score(fold_data["y_val"], y_prob))
                fold_scores.append(fold_auc)
            except ValueError:
                # Validation fold has only one class (rare edge case) — skip
                pass

        if not fold_scores:
            continue

        score = float(np.mean(fold_scores))

        if score > best_score:
            best_score = score
            best_params = params
            logger.info(f"  Trial {trial:3d}: ROC-AUC={score:.4f}  *** new best ***")
        else:
            logger.info(f"  Trial {trial:3d}: ROC-AUC={score:.4f}")

    logger.info(f"Best ROC-AUC (walk-forward CV): {best_score:.4f}")
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

    # Determine the canonical training split so CV is performed on exactly the
    # same customer cohort that will be used for final model training.
    _, _, _, _, _, test_customer_ids = time_based_split(features, target_df, customers)
    eligible_ids = set(features["customer_id"]) & set(target_df["customer_id"])
    train_customer_ids = list(eligible_ids - set(test_customer_ids))

    tune_hyperparameters(orders, customers, train_customer_ids)

    logger.info("=== Tuning complete ===")
