"""XGBoost churn model: training, evaluation, and artifact persistence.

Run as a module:
    uv run python -m src.models.train
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve

from src.baseline.rule_based import evaluate_baseline
from src.config import (
    ARTIFACTS_DIR,
    AUG_N_WINDOWS,
    AUG_STEP_DAYS,
    CHURN_PERIOD_DAYS,
    DATA_DIR,
    TEST_FRACTION,
    XGBOOST_PARAMS,
)
from src.evaluation.calibration import compute_calibration
from src.evaluation.distribution_shift import compute_adversarial_auc
from src.evaluation.guardrails import run_all_guardrails
from src.evaluation.metrics import compute_metrics
from src.feature_engineering.pipeline import build_feature_matrix
from src.preprocessing.clean import clean_customers, clean_orders
from src.target.churn_target import compute_churn_target
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_feature_cols(features: pd.DataFrame) -> list[str]:
    """Return all column names except ``customer_id``."""
    return [c for c in features.columns if c != "customer_id"]


def time_based_split(
    features: pd.DataFrame,
    target: pd.DataFrame,
    customers: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Stratified-by-registration-cohort train / test split – no shuffle.

    Customers are sorted by ``registration_date`` and then every
    ``round(1 / test_fraction)``-th customer (0-indexed) is assigned to the
    test set; the rest form the training set.  This interleaved selection
    ensures both splits cover the same registration-date range, preventing
    the cohort confound that arises from a hard temporal cut (which would
    place all late-joiners in the test set and all early-joiners in train).

    Returns:
        X_train, y_train, X_test, y_test, feature_cols, test_customer_ids
    """
    df = features.merge(target, on="customer_id", how="inner")
    reg_dates = customers[["customer_id", "registration_date"]]
    df = df.merge(reg_dates, on="customer_id", how="left")
    df = df.sort_values("registration_date").reset_index(drop=True)

    every_nth = max(2, round(1.0 / test_fraction))
    test_mask = df.index % every_nth == (every_nth - 1)
    train_df = df[~test_mask]
    test_df = df[test_mask]

    feature_cols = get_feature_cols(features)
    X_train = train_df[feature_cols]
    y_train = train_df["is_churned"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_churned"]
    test_customer_ids = test_df["customer_id"].tolist()

    logger.info(
        f"Split: train={len(X_train):,} (churn={y_train.mean():.2%})  "
        f"test={len(X_test):,} (churn={y_test.mean():.2%})"
    )
    return X_train, y_train, X_test, y_test, feature_cols, test_customer_ids


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier on the provided data.

    Args:
        X_train: Feature matrix.
        y_train: Binary target.
        params: Hyperparameter dict; defaults to ``XGBOOST_PARAMS``.

    Returns:
        Fitted XGBClassifier.
    """
    if params is None:
        params = XGBOOST_PARAMS.copy()

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost training complete")
    return model


# ---------------------------------------------------------------------------
# Sliding-window data augmentation
# ---------------------------------------------------------------------------


def build_augmented_training_set(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    train_customer_ids: list,
    feature_end_date: pd.Timestamp,
    feature_cols: list[str],
    n_windows: int = AUG_N_WINDOWS,
    step_days: int = AUG_STEP_DAYS,
    churn_period_days: int = CHURN_PERIOD_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate extra (features, label) rows by sliding the churn window back in time.

    For each window i ∈ {1, …, n_windows}:
      aug_feature_end  = feature_end_date − i × step_days
      aug_t_max        = aug_feature_end  + churn_period_days
      label            = no orders in (aug_feature_end, aug_t_max]
      features         = build_feature_matrix(orders, ..., reference_date=aug_feature_end)

    Only training-set customers are used; test-set customers are never included.
    All features are computed on orders strictly before ``aug_feature_end``, so there
    is no leakage from the churn observation window or from future data.

    Args:
        orders: Full cleaned orders fact table.
        customers: Customer dimension table.
        train_customer_ids: Customer IDs that belong to the training split.
        feature_end_date: The original feature cutoff (= T_max − churn_period_days).
        feature_cols: Column names expected by the model (must match the primary window).
        n_windows: Number of additional windows to generate.
        step_days: Days to step back per window.
        churn_period_days: Length of each churn observation window.

    Returns:
        Tuple of (X_aug, y_aug) — concatenated across all windows.
        Returns empty DataFrames if no eligible customers are found in any window.
    """
    train_id_set = set(train_customer_ids)
    train_customers = customers[customers["customer_id"].isin(train_id_set)]
    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for i in range(1, n_windows + 1):
        aug_feature_end: pd.Timestamp = feature_end_date - pd.Timedelta(days=i * step_days)
        aug_t_max: pd.Timestamp = aug_feature_end + pd.Timedelta(days=churn_period_days)

        # Eligible: placed at least one order on or before the augmented feature cutoff
        pre_window_ids = set(
            orders[orders["order_date"] <= aug_feature_end]["customer_id"]
        ) & train_id_set
        if not pre_window_ids:
            logger.warning(f"Aug window {i}: no eligible training customers — skipping")
            continue

        # Labels: churned = no orders in the augmented churn window
        active_ids = set(
            orders[
                (orders["order_date"] > aug_feature_end) & (orders["order_date"] <= aug_t_max)
            ]["customer_id"]
        )
        window_customers = train_customers[train_customers["customer_id"].isin(pre_window_ids)]
        target = pd.DataFrame({"customer_id": list(pre_window_ids)})
        target["is_churned"] = (~target["customer_id"].isin(active_ids)).astype(int)

        # Features: strictly before aug_feature_end
        feats = build_feature_matrix(orders, window_customers, aug_feature_end)
        merged = feats.merge(target, on="customer_id", how="inner")
        if merged.empty:
            continue

        X_parts.append(merged[feature_cols])
        y_parts.append(merged["is_churned"].reset_index(drop=True))
        logger.info(
            f"Aug window {i}: feature_end={aug_feature_end.date()}  "
            f"t_max={aug_t_max.date()}  n={len(merged):,}  "
            f"churn={merged['is_churned'].mean():.2%}"
        )

    if not X_parts:
        logger.warning("No augmentation windows produced data; returning empty DataFrames")
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=int, name="is_churned")

    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_model(model: xgb.XGBClassifier, path: Path | None = None) -> Path:
    """Pickle the trained model to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = ARTIFACTS_DIR / "xgb_churn_model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info(f"Model saved → {path}")
    return path


def load_model(path: Path | None = None) -> xgb.XGBClassifier:
    """Load a previously saved model from disk.

    Raises:
        FileNotFoundError: if the model file does not exist.
    """
    if path is None:
        path = ARTIFACTS_DIR / "xgb_churn_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `make train` first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)  # nosec B301 - loading trusted local model artifact


_REGULARIZATION_KEYS: frozenset[str] = frozenset({"gamma", "reg_alpha", "min_child_weight"})


def load_best_params() -> dict | None:
    """Load tuned hyperparameters saved by ``tune.py``, if available."""
    params_path = ARTIFACTS_DIR / "best_params.json"
    if params_path.exists():
        with open(params_path) as fh:
            data = json.load(fh)
        logger.info(f"Loaded tuned params (CV ROC-AUC={data['score']:.4f})")
        missing = _REGULARIZATION_KEYS - set(data["params"])
        if missing:
            logger.warning(
                f"Loaded params are missing regularisation keys {missing}. "
                "They were tuned without these constraints and may overfit. "
                f"Delete {params_path} or run `make tune` to re-tune."
            )
        return data["params"]
    logger.info("No tuned params found – using defaults")
    return None


# ---------------------------------------------------------------------------
# Artifact persistence for dashboard
# ---------------------------------------------------------------------------


def save_evaluation_artifacts(
    model: xgb.XGBClassifier,
    metrics: dict,
    calibration: dict,
    guardrails: list[dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    baseline_metrics: dict,
    shift_metrics: dict,
) -> None:
    """Write all evaluation artefacts consumed by the dashboard."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ARTIFACTS_DIR / "eval_metrics.json", "w") as fh:
        json.dump(
            {
                **metrics,
                "baseline_roc_auc": baseline_metrics.get("roc_auc"),
                "baseline_pr_auc": baseline_metrics.get("pr_auc"),
                "baseline_brier_score": baseline_metrics.get("brier_score"),
                "baseline_precision": baseline_metrics.get("precision"),
                "baseline_recall": baseline_metrics.get("recall"),
                "baseline_f1": baseline_metrics.get("f1"),
                "baseline_n_samples": baseline_metrics.get("n_samples"),
            },
            fh,
            indent=2,
        )

    with open(ARTIFACTS_DIR / "calibration.json", "w") as fh:
        json.dump(calibration, fh, indent=2)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    with open(ARTIFACTS_DIR / "roc_curve.json", "w") as fh:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, fh)

    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
    with open(ARTIFACTS_DIR / "feature_importance.json", "w") as fh:
        json.dump(importance_sorted, fh, indent=2)

    with open(ARTIFACTS_DIR / "feature_cols.json", "w") as fh:
        json.dump(feature_cols, fh)

    with open(ARTIFACTS_DIR / "guardrails.json", "w") as fh:
        json.dump(guardrails, fh, indent=2)

    with open(ARTIFACTS_DIR / "distribution_shift.json", "w") as fh:
        json.dump(shift_metrics, fh, indent=2)

    logger.info(f"Evaluation artefacts saved → {ARTIFACTS_DIR}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Churn Model Training ===")

    orders_raw = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date", "contract_date"])
    customers_raw = pd.read_csv(
        DATA_DIR / "customers.csv",
        parse_dates=["registration_date", "birth_date", "last_profile_update"],
    )

    orders = clean_orders(orders_raw)
    customers = clean_customers(customers_raw)

    target_df, feature_end_date, t_max = compute_churn_target(orders, customers, CHURN_PERIOD_DAYS)
    features = build_feature_matrix(orders, customers, feature_end_date)

    X_train, y_train, X_test, y_test, feature_cols, test_customer_ids = time_based_split(
        features, target_df, customers
    )

    # --- Distribution shift: can the model tell train from test? ---
    shift_metrics = compute_adversarial_auc(X_train, X_test)

    # --- Sliding-window data augmentation (training customers only, no leakage) ---
    train_customer_ids = [
        cid for cid in target_df["customer_id"].tolist() if cid not in set(test_customer_ids)
    ]
    X_aug, y_aug = build_augmented_training_set(
        orders,
        customers,
        train_customer_ids,
        feature_end_date,
        feature_cols,
        n_windows=AUG_N_WINDOWS,
        step_days=AUG_STEP_DAYS,
        churn_period_days=CHURN_PERIOD_DAYS,
    )
    if not X_aug.empty:
        X_train = pd.concat([X_train, X_aug], ignore_index=True)
        y_train = pd.concat([y_train, y_aug], ignore_index=True)
        logger.info(
            f"Augmented training set: {len(X_train):,} rows (churn={y_train.mean():.2%})"
        )

    params = load_best_params() or XGBOOST_PARAMS.copy()
    model = train_xgboost(X_train, y_train, params)

    # --- Test-set metrics ---
    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_prob_test)
    calibration = compute_calibration(y_test.to_numpy(), y_prob_test)

    logger.info(
        f"ML test   ROC-AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"Brier={metrics['brier_score']:.4f}"
    )

    # --- Train-set metrics (overfitting detection) ---
    y_prob_train = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train.to_numpy(), y_prob_train)
    logger.info(
        f"ML train  ROC-AUC={train_metrics['roc_auc']:.4f}  "
        f"gap={train_metrics['roc_auc'] - metrics['roc_auc']:.4f}"
    )

    # --- Baseline evaluated on the **same test set** for a fair comparison ---
    test_features = features[features["customer_id"].isin(test_customer_ids)]
    test_target = target_df[target_df["customer_id"].isin(test_customer_ids)]
    baseline_metrics = evaluate_baseline(test_features, test_target)

    if baseline_metrics["n_samples"] != len(y_test):
        raise RuntimeError(
            f"Sample count mismatch: ML test set has {len(y_test)} rows but "
            f"baseline evaluated on {baseline_metrics['n_samples']}. "
            "Ensure evaluate_baseline receives only test-split customers."
        )

    # --- Detailed side-by-side comparison (same N, same labels) ---
    n = len(y_test)
    cr = metrics["churn_rate"]
    logger.info(f"ML vs Baseline — test set  N={n:,}  churn_rate={cr:.2%}")
    _cmp = [
        ("ROC-AUC", "roc_auc", "higher=better"),
        ("PR-AUC", "pr_auc", "higher=better"),
        ("Brier", "brier_score", "lower=better"),
        ("Precision", "precision", "higher=better"),
        ("Recall", "recall", "higher=better"),
        ("F1", "f1", "higher=better"),
    ]
    for label, key, direction in _cmp:
        ml_v = metrics[key]
        bl_v = baseline_metrics[key]
        delta = ml_v - bl_v
        logger.info(
            f"  {label:<10}  ML={ml_v:.4f}  baseline={bl_v:.4f}  delta={delta:+.4f}  ({direction})"
        )

    predictions_stub = pd.DataFrame({"customer_id": features["customer_id"]})
    guardrails_results = run_all_guardrails(
        metrics,
        calibration,
        predictions_stub,
        customers,
        train_metrics=train_metrics,
        baseline_metrics=baseline_metrics,
        shift_metrics=shift_metrics,
    )
    for g in guardrails_results:
        status = "PASS" if g["passed"] else "FAIL"
        logger.info(f"  [{status}] {g['message']}")

    save_model(model)
    save_evaluation_artifacts(
        model,
        metrics,
        calibration,
        guardrails_results,
        X_test,
        y_test,
        feature_cols,
        baseline_metrics,
        shift_metrics,
    )

    # --- Category-level summary for dashboard (test-set actual vs predicted) ---
    test_df_cats = pd.DataFrame(
        {"customer_id": test_customer_ids, "actual": y_test.to_numpy(), "predicted": y_prob_test}
    )
    test_df_cats = test_df_cats.merge(
        customers[["customer_id", "segment", "marketing_channel", "country"]],
        on="customer_id",
        how="left",
    )
    category_summary: dict = {}
    for raw_col, key in [
        ("segment", "segment"),
        ("marketing_channel", "channel"),
        ("country", "country"),
    ]:
        grp = (
            test_df_cats.groupby(raw_col)
            .agg(
                actual_churn_rate=("actual", "mean"),
                predicted_churn_rate=("predicted", "mean"),
                count=("actual", "count"),
            )
            .reset_index()
            .sort_values(raw_col)
            .rename(columns={raw_col: "category"})
        )
        category_summary[key] = grp.to_dict("records")

    customer_counts: dict = {}
    for raw_col, key in [
        ("segment", "segment"),
        ("marketing_channel", "channel"),
        ("country", "country"),
    ]:
        customer_counts[key] = customers[raw_col].value_counts().to_dict()

    with open(ARTIFACTS_DIR / "category_summary.json", "w") as fh:
        json.dump(category_summary, fh, indent=2)
    with open(ARTIFACTS_DIR / "customer_stats.json", "w") as fh:
        json.dump(customer_counts, fh, indent=2)
    logger.info("Category summary artifacts saved")

    logger.info("=== Training complete ===")
