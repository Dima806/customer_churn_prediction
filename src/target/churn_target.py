"""Leakage-free churn target construction.

Strategy (from PRD §4–§5):
1. T_max  = last order date in the dataset.
2. Churn window  = (T_max − CHURN_PERIOD_DAYS, T_max].
3. A customer is churned (is_churned = 1) if they placed NO orders inside
   the churn window.
4. Only customers with at least one order BEFORE the churn window are
   eligible for the target (we cannot label new-to-churn-window customers).
5. Features must be computed on data strictly before churn_window_start.
"""

import pandas as pd

from src.config import CHURN_PERIOD_DAYS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_churn_target(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    churn_period_days: int = CHURN_PERIOD_DAYS,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Compute the binary churn label for every eligible customer.

    Args:
        orders: Cleaned orders fact table with ``order_date`` as Timestamp.
        customers: Customer dimension table with ``customer_id``.
        churn_period_days: Length of the churn observation window in days.

    Returns:
        A three-tuple:
        - ``target_df``          – DataFrame[customer_id, is_churned].
        - ``feature_end_date``   – Exclusive upper bound for feature computation
                                   (= churn_window_start).
        - ``t_max``              – Last date in the dataset.
    """
    t_max: pd.Timestamp = orders["order_date"].max()
    churn_window_start: pd.Timestamp = t_max - pd.Timedelta(days=churn_period_days)

    logger.info(f"T_max={t_max.date()}  churn_window=[{churn_window_start.date()}, {t_max.date()}]")

    # Customers who ordered at least once BEFORE the churn window
    pre_window_orders = orders[orders["order_date"] <= churn_window_start]
    eligible_ids = pre_window_orders["customer_id"].unique()

    # Customers who ordered INSIDE the churn window (= active = not churned)
    active_in_window = orders[orders["order_date"] > churn_window_start]["customer_id"].unique()

    target_df = pd.DataFrame({"customer_id": eligible_ids})
    target_df["is_churned"] = (~target_df["customer_id"].isin(active_in_window)).astype(int)

    churn_rate = target_df["is_churned"].mean()
    logger.info(f"Eligible customers: {len(target_df):,}  Churn rate: {churn_rate:.2%}")

    return target_df, churn_window_start, t_max
