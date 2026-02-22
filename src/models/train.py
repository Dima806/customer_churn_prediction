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
    CHURN_PERIOD_DAYS,
    DATA_DIR,
    TEST_FRACTION,
    XGBOOST_PARAMS,
)
from src.evaluation.calibration import compute_calibration
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
    """Cohort-based temporal train / test split – no shuffle.

    Customers are sorted by ``registration_date`` (the natural cohort axis).
    The earliest-registered ``1 − test_fraction`` customers form the training
    set; the most-recently-registered form the test set.

    This mirrors real-world deployment: train on established cohorts,
    evaluate on incoming cohorts.  Sorting by last-order-date would place
    all recently-active (near-zero churn rate) customers in the test set,
    producing a degenerate single-class evaluation.

    Returns:
        X_train, y_train, X_test, y_test, feature_cols
    """
    df = features.merge(target, on="customer_id", how="inner")
    reg_dates = customers[["customer_id", "registration_date"]]
    df = df.merge(reg_dates, on="customer_id", how="left")
    df = df.sort_values("registration_date").reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    feature_cols = get_feature_cols(features)
    X_train = train_df[feature_cols]
    y_train = train_df["is_churned"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_churned"]

    logger.info(
        f"Split: train={len(X_train):,} (churn={y_train.mean():.2%})  "
        f"test={len(X_test):,} (churn={y_test.mean():.2%})"
    )
    return X_train, y_train, X_test, y_test, feature_cols


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


def load_best_params() -> dict | None:
    """Load tuned hyperparameters saved by ``tune.py``, if available."""
    params_path = ARTIFACTS_DIR / "best_params.json"
    if params_path.exists():
        with open(params_path) as fh:
            data = json.load(fh)
        logger.info(f"Loaded tuned params (CV ROC-AUC={data['score']:.4f})")
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
) -> None:
    """Write all evaluation artefacts consumed by the dashboard."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ARTIFACTS_DIR / "eval_metrics.json", "w") as fh:
        json.dump(
            {**metrics, "baseline_roc_auc": baseline_metrics.get("roc_auc")},
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

    X_train, y_train, X_test, y_test, feature_cols = time_based_split(
        features, target_df, customers
    )

    params = load_best_params() or XGBOOST_PARAMS.copy()
    model = train_xgboost(X_train, y_train, params)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_prob_test)
    calibration = compute_calibration(y_test.to_numpy(), y_prob_test)

    logger.info(
        f"ML  ROC-AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"Brier={metrics['brier_score']:.4f}"
    )

    baseline_metrics = evaluate_baseline(features, target_df)

    beat = metrics["roc_auc"] > baseline_metrics["roc_auc"]
    logger.info(f"ML vs Baseline: {'BETTER' if beat else 'WORSE'} by ROC-AUC")

    predictions_stub = pd.DataFrame({"customer_id": features["customer_id"]})
    guardrails_results = run_all_guardrails(metrics, calibration, predictions_stub, customers)
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
    )

    logger.info("=== Training complete ===")
