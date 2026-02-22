"""RFM (Recency, Frequency, Monetary) feature computation.

All aggregations are computed strictly on orders whose ``order_date``
is *before* ``reference_date`` (the start of the churn window).
This guarantees zero target leakage.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_rfm(
    orders: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute RFM features for every customer active before reference_date.

    Args:
        orders: Full orders fact table with columns
                [customer_id, order_id, order_date, total_value].
        reference_date: Exclusive upper bound – only orders *before* this
                        date are used.

    Returns:
        DataFrame indexed by ``customer_id`` with columns:
        recency, frequency, monetary, avg_order_value, max_order_value.
    """
    hist = orders[orders["order_date"] < reference_date].copy()

    if hist.empty:
        logger.warning("compute_rfm: no historical orders found before reference_date")
        return pd.DataFrame(
            columns=[
                "customer_id",
                "recency",
                "frequency",
                "monetary",
                "avg_order_value",
                "max_order_value",
            ]
        )

    rfm = (
        hist.groupby("customer_id")
        .agg(
            recency=("order_date", lambda x: (reference_date - x.max()).days),
            frequency=("order_id", "count"),
            monetary=("total_value", "sum"),
            avg_order_value=("total_value", "mean"),
            max_order_value=("total_value", "max"),
        )
        .reset_index()
    )

    rfm["monetary"] = rfm["monetary"].round(2)
    rfm["avg_order_value"] = rfm["avg_order_value"].round(2)

    logger.info(
        f"compute_rfm: {len(rfm):,} customers, recency median={rfm['recency'].median():.0f}d"
    )
    return rfm
