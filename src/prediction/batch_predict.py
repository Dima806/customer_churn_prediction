"""Batch churn probability prediction for all active customers.

Run as a module:
    uv run python -m src.prediction.batch_predict
"""

import pandas as pd

from src.config import (
    CHURN_PERIOD_DAYS,
    DATA_DIR,
    PREDICTIONS_DIR,
    RISK_HIGH,
    RISK_MEDIUM,
)
from src.feature_engineering.pipeline import build_feature_matrix
from src.models.train import get_feature_cols, load_model
from src.preprocessing.clean import clean_customers, clean_orders
from src.target.churn_target import compute_churn_target
from src.utils.logger import get_logger

logger = get_logger(__name__)


def assign_risk_bucket(prob: float) -> str:
    """Map a churn probability to a risk segment label.

    Args:
        prob: Predicted churn probability in [0, 1].

    Returns:
        One of ``"high"``, ``"medium"``, or ``"low"``.
    """
    if prob >= RISK_HIGH:
        return "high"
    if prob >= RISK_MEDIUM:
        return "medium"
    return "low"


def run_batch_prediction(
    orders: pd.DataFrame | None = None,
    customers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate batch churn predictions for all eligible customers.

    The model is trained on the historical window ending at
    ``T_max − CHURN_PERIOD_DAYS``.  Predictions represent estimated churn
    probability over the *next* ``CHURN_PERIOD_DAYS`` days.

    Args:
        orders: Pre-loaded orders DataFrame (reads from disk if None).
        customers: Pre-loaded customers DataFrame (reads from disk if None).

    Returns:
        DataFrame with columns [customer_id, churn_probability, risk_bucket].
        Also persisted to ``predictions/churn_predictions.csv``.
    """
    if orders is None:
        orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date", "contract_date"])
    if customers is None:
        customers = pd.read_csv(
            DATA_DIR / "customers.csv",
            parse_dates=["registration_date", "birth_date", "last_profile_update"],
        )

    orders = clean_orders(orders)
    customers = clean_customers(customers)

    _, feature_end_date, t_max = compute_churn_target(orders, customers, CHURN_PERIOD_DAYS)
    logger.info(f"Building features up to {feature_end_date.date()} (T_max={t_max.date()})")

    features = build_feature_matrix(orders, customers, feature_end_date)

    model = load_model()
    feature_cols = get_feature_cols(features)
    X = features[feature_cols]

    probs = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame(
        {
            "customer_id": features["customer_id"].values,
            "churn_probability": probs.round(6),
            "risk_bucket": [assign_risk_bucket(p) for p in probs],
        }
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / "churn_predictions.csv"
    predictions.to_csv(output_path, index=False)

    dist = predictions["risk_bucket"].value_counts().to_dict()
    logger.info(
        f"Predictions saved → {output_path}  "
        f"high={dist.get('high', 0)}, medium={dist.get('medium', 0)}, low={dist.get('low', 0)}"
    )
    return predictions


if __name__ == "__main__":
    logger.info("=== Batch Prediction ===")
    preds = run_batch_prediction()
    logger.info(f"Generated {len(preds):,} predictions")
    logger.info("=== Done ===")
