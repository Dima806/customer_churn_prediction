"""Trend features: slope of order frequency, rolling averages, spend change."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_N_TREND_PERIODS = 6  # months of history used for slope computation


def _ols_slope(values: np.ndarray) -> float:
    """Return the OLS slope of a 1-D time series (no intercept shift needed)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = values.astype(float) - values.mean()
    denom = float((x * x).sum())
    if denom == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def compute_trend_features(
    orders: pd.DataFrame,
    reference_date: pd.Timestamp,
    n_periods: int = _N_TREND_PERIODS,
) -> pd.DataFrame:
    """Compute trend-based features for each customer.

    Features produced:
    - order_frequency_slope  – OLS slope of monthly order counts (last n_periods months)
    - spend_slope            – OLS slope of monthly spend
    - rolling_avg_orders_3m  – mean monthly order count over last 3 months
    - spend_change_ratio     – rolling_avg_orders_3m / (avg over previous 3 months + 1)

    Args:
        orders: Full orders fact table.
        reference_date: Upper bound for feature computation (exclusive).
        n_periods: Number of monthly buckets used for slope computation.

    Returns:
        DataFrame with one row per customer.
    """
    hist = orders[orders["order_date"] < reference_date].copy()
    hist["year_month"] = hist["order_date"].dt.to_period("M")

    monthly = (
        hist.groupby(["customer_id", "year_month"])
        .agg(monthly_orders=("order_id", "count"), monthly_spend=("total_value", "sum"))
        .reset_index()
    )

    # Build the index of the last n_periods complete months
    all_months = pd.period_range(
        end=reference_date - pd.Timedelta(days=1), periods=n_periods, freq="M"
    )

    records: list[dict] = []
    for cust_id, grp in monthly.groupby("customer_id"):
        ts = grp.set_index("year_month").reindex(all_months, fill_value=0)

        order_vals = ts["monthly_orders"].to_numpy()
        spend_vals = ts["monthly_spend"].to_numpy()

        order_slope = _ols_slope(order_vals)
        spend_slope = _ols_slope(spend_vals)

        last_3 = float(order_vals[-3:].mean()) if len(order_vals) >= 3 else float(order_vals.mean())
        prev_3 = float(order_vals[-6:-3].mean()) if len(order_vals) >= 6 else 0.0
        spend_change_ratio = round(last_3 / (prev_3 + 1.0), 4)

        records.append(
            {
                "customer_id": cust_id,
                "order_frequency_slope": round(order_slope, 6),
                "spend_slope": round(spend_slope, 6),
                "rolling_avg_orders_3m": round(last_3, 4),
                "spend_change_ratio": spend_change_ratio,
            }
        )

    result = pd.DataFrame(records)
    logger.info(f"compute_trend_features: {len(result):,} customers")
    return result
