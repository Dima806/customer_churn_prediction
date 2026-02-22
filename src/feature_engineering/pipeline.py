"""Feature engineering pipeline: assembles the full feature matrix.

Entry point: ``build_feature_matrix(orders, customers, reference_date)``.
The reference_date is the start of the churn window – all features are
computed on data STRICTLY BEFORE this date.
"""

import pandas as pd

from src.config import FEATURE_WINDOWS
from src.feature_engineering.customer_attributes import compute_customer_attributes
from src.feature_engineering.rfm import compute_rfm
from src.feature_engineering.time_features import compute_time_features
from src.feature_engineering.trend_features import compute_trend_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_feature_matrix(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    reference_date: pd.Timestamp,
    windows: list[int] = FEATURE_WINDOWS,
) -> pd.DataFrame:
    """Build the complete feature matrix for churn modelling.

    All feature sub-modules receive ``reference_date`` as the exclusive
    upper bound, so no information from the churn observation window can
    leak into the features.

    Args:
        orders: Cleaned orders fact table.
        customers: Cleaned customer dimension table.
        reference_date: Start of the churn window (= T_max − CHURN_PERIOD_DAYS).
                        Features are built on orders *before* this date.
        windows: Rolling-window sizes in days for time-based features.

    Returns:
        DataFrame with one row per eligible customer and all numeric features.
        ``customer_id`` is preserved as the join key.
    """
    logger.info(f"Building feature matrix (reference_date={reference_date.date()}) …")

    rfm = compute_rfm(orders, reference_date)
    time_feats = compute_time_features(orders, reference_date, windows)
    trend_feats = compute_trend_features(orders, reference_date)
    cust_attrs = compute_customer_attributes(customers, reference_date)

    features = rfm
    features = features.merge(time_feats, on="customer_id", how="left")
    features = features.merge(trend_feats, on="customer_id", how="left")
    features = features.merge(cust_attrs, on="customer_id", how="left")

    # Fill any residual NaN with 0 (customers with thin history)
    numeric_cols = features.select_dtypes(include="number").columns.difference(["customer_id"])
    features[numeric_cols] = features[numeric_cols].fillna(0)

    logger.info(f"Feature matrix: {len(features):,} rows × {len(features.columns) - 1} features")
    return features
