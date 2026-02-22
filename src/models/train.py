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

    logger.info("=== Training complete ===")
