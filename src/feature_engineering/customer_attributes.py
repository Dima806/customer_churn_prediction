"""Customer attribute feature encoding.

Uses deterministic ordinal encoding so the mapping is reproducible
without persisting a fitted encoder object.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Fixed category orders guarantee consistent codes across runs
_SEGMENT_ORDER: list[str] = ["bronze", "silver", "gold", "platinum"]
_CHANNEL_ORDER: list[str] = ["organic", "social", "email", "paid_search"]
_COUNTRY_ORDER: list[str] = ["AU", "CA", "DE", "ES", "FR", "NL", "UK", "US"]


def _ordinal_encode(series: pd.Series, categories: list[str]) -> pd.Series:
    cat = pd.Categorical(series, categories=categories)
    return pd.Series(cat.codes, index=series.index, dtype=int)


def compute_customer_attributes(
    customers: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Encode customer metadata into numeric features.

    Args:
        customers: Customer dimension table with columns
                   [customer_id, registration_date, segment,
                    marketing_channel, country].
        reference_date: Used to compute tenure (days since registration).

    Returns:
        DataFrame with customer_id and encoded feature columns.
    """
    df = customers[
        ["customer_id", "registration_date", "segment", "marketing_channel", "country"]
    ].copy()

    df["tenure_days"] = (reference_date - df["registration_date"]).dt.days.clip(lower=0)
    df["segment_encoded"] = _ordinal_encode(df["segment"], _SEGMENT_ORDER)
    df["channel_encoded"] = _ordinal_encode(df["marketing_channel"], _CHANNEL_ORDER)
    df["country_encoded"] = _ordinal_encode(df["country"], _COUNTRY_ORDER)

    result = df[
        ["customer_id", "tenure_days", "segment_encoded", "channel_encoded", "country_encoded"]
    ]
    logger.info(f"compute_customer_attributes: {len(result):,} customers")
    return result
