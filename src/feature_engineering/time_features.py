"""Time-based feature computation: rolling windows, inter-order stats."""

import pandas as pd

from src.config import FEATURE_WINDOWS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_time_features(
    orders: pd.DataFrame,
    reference_date: pd.Timestamp,
    windows: list[int] = FEATURE_WINDOWS,
) -> pd.DataFrame:
    """Compute time-based features per customer.

    Features produced:
    - avg_inter_order_days  – mean gap between consecutive orders
    - std_inter_order_days  – std dev of that gap (0 for single-order customers)
    - orders_last_Nd        – order count in the last N days (per ``windows``)
    - spend_last_Nd         – total spend in the last N days
    - ratio_recent_historical – orders_last_30d / (orders_last_90d + 1)

    Args:
        orders: Full orders fact table.
        reference_date: Features are computed on orders strictly before this date.
        windows: List of rolling-window sizes in days.

    Returns:
        DataFrame with one row per customer.
    """
    hist = orders[orders["order_date"] < reference_date].sort_values(["customer_id", "order_date"])

    # Inter-order gap stats
    hist = hist.copy()
    hist["prev_date"] = hist.groupby("customer_id")["order_date"].shift(1)
    hist["inter_days"] = (hist["order_date"] - hist["prev_date"]).dt.days

    inter_stats = (
        hist.groupby("customer_id")["inter_days"]
        .agg(avg_inter_order_days="mean", std_inter_order_days="std")
        .reset_index()
    )
    inter_stats["avg_inter_order_days"] = inter_stats["avg_inter_order_days"].fillna(0).round(2)
    inter_stats["std_inter_order_days"] = inter_stats["std_inter_order_days"].fillna(0).round(2)

    result = inter_stats.copy()

    # Rolling-window counts and spend
    for w in sorted(windows):
        w_start = reference_date - pd.Timedelta(days=w)
        window_orders = hist[hist["order_date"] >= w_start]

        cnt = window_orders.groupby("customer_id").size().reset_index(name=f"orders_last_{w}d")
        spend = (
            window_orders.groupby("customer_id")["total_value"]
            .sum()
            .reset_index()
            .rename(columns={"total_value": f"spend_last_{w}d"})
        )

        result = result.merge(cnt, on="customer_id", how="left")
        result[f"orders_last_{w}d"] = result[f"orders_last_{w}d"].fillna(0).astype(int)

        result = result.merge(spend, on="customer_id", how="left")
        result[f"spend_last_{w}d"] = result[f"spend_last_{w}d"].fillna(0.0).round(2)

    # Ratio recent vs historical
    if "orders_last_30d" in result.columns and "orders_last_90d" in result.columns:
        result["ratio_recent_historical"] = (
            result["orders_last_30d"] / (result["orders_last_90d"] + 1)
        ).round(4)
    else:
        result["ratio_recent_historical"] = 0.0

    logger.info(f"compute_time_features: {len(result):,} customers, windows={windows}")
    return result
