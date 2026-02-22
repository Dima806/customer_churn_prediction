"""Data cleaning and schema validation for raw CSV inputs."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the orders fact table.

    Operations performed:
    - Drop rows with null ``order_date`` or ``customer_id`` (mandatory keys).
    - Coerce date columns to proper dtypes.
    - Clip ``total_value`` at 2× the 99th percentile (winsorisation).
    - Enforce minimum sensible values for ``quantity`` and ``price``.

    Args:
        orders: Raw orders DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = orders.copy()
    initial = len(df)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["contract_date"] = pd.to_datetime(df["contract_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "customer_id"])

    # Clip extreme monetary values (2× P99)
    p99 = df["total_value"].quantile(0.99)
    df["total_value"] = df["total_value"].clip(upper=p99 * 2.0)

    df["quantity"] = df["quantity"].clip(lower=1)
    df["price"] = df["price"].clip(lower=0.01)

    df["customer_id"] = df["customer_id"].astype(int)
    df["order_id"] = df["order_id"].astype(int)

    removed = initial - len(df)
    if removed:
        logger.warning(f"clean_orders: removed {removed} invalid rows")
    else:
        logger.info(f"clean_orders: {len(df):,} rows, no rows removed")

    return df.reset_index(drop=True)


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the customer dimension table.

    Args:
        customers: Raw customers DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = customers.copy()

    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["last_profile_update"] = pd.to_datetime(df["last_profile_update"], errors="coerce")
    df["customer_id"] = df["customer_id"].astype(int)

    missing_reg = df["registration_date"].isna().sum()
    if missing_reg:
        logger.warning(f"clean_customers: {missing_reg} missing registration_date – dropping")
        df = df.dropna(subset=["registration_date"])

    logger.info(f"clean_customers: {len(df):,} customers")
    return df.reset_index(drop=True)
