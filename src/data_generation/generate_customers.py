"""Synthetic customer dimension table generator."""

import numpy as np
import pandas as pd

from src.config import END_DATE, N_CUSTOMERS, SEED, START_DATE

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
COUNTRIES: list[str] = ["US", "UK", "DE", "FR", "CA", "AU", "NL", "ES"]
COUNTRY_WEIGHTS: list[float] = [0.35, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05]

SEGMENTS: list[str] = ["bronze", "silver", "gold", "platinum"]
SEGMENT_WEIGHTS: list[float] = [0.40, 0.30, 0.20, 0.10]

MARKETING_CHANNELS: list[str] = ["organic", "email", "social", "paid_search"]
CHANNEL_WEIGHTS: list[float] = [0.30, 0.25, 0.25, 0.20]

LIFECYCLE_TYPES: list[str] = [
    "loyal",
    "seasonal",
    "early_churner",
    "churn_returner",
    "late_joiner",
]
LIFECYCLE_WEIGHTS: list[float] = [0.30, 0.20, 0.15, 0.15, 0.20]


def generate_customers(
    n: int = N_CUSTOMERS,
    seed: int = SEED,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Generate a synthetic customer dimension table.

    Args:
        n: Number of customers to generate.
        seed: Random seed for full reproducibility.
        start_date: Start of the data window (ISO format).
        end_date: End of the data window (ISO format).

    Returns:
        DataFrame with one row per customer.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Lifecycle type determines registration timing
    lifecycle_types = rng.choice(LIFECYCLE_TYPES, size=n, p=LIFECYCLE_WEIGHTS)

    registration_dates: list[pd.Timestamp] = []
    for lt in lifecycle_types:
        if lt == "late_joiner":
            reg_start = start + pd.Timedelta(days=365)
            reg_end = end - pd.Timedelta(days=180)
        else:
            # Spread registrations across the full period so that both
            # train and test cohorts share the same registration range.
            reg_start = start
            reg_end = end - pd.Timedelta(days=180)

        span = max(1, (reg_end - reg_start).days)
        offset = int(rng.integers(0, span))
        registration_dates.append(reg_start + pd.Timedelta(days=offset))

    # Demographics
    countries = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    segments = rng.choice(SEGMENTS, size=n, p=SEGMENT_WEIGHTS)
    channels = rng.choice(MARKETING_CHANNELS, size=n, p=CHANNEL_WEIGHTS)

    # Birth date – nullable (~15 % missing)
    birth_dates: list[pd.Timestamp | float] = []
    for _ in range(n):
        if rng.random() < 0.15:
            birth_dates.append(float("nan"))
        else:
            age_days = int(rng.integers(18 * 365, 70 * 365))
            birth_dates.append(end - pd.Timedelta(days=age_days))

    # Last profile update – nullable (~30 % missing)
    last_updates: list[pd.Timestamp | float] = []
    for reg_date in registration_dates:
        if rng.random() < 0.30:
            last_updates.append(float("nan"))
        else:
            span = max(1, (end - reg_date).days)
            offset = int(rng.integers(0, span))
            last_updates.append(reg_date + pd.Timedelta(days=offset))

    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "registration_date": pd.to_datetime(registration_dates),
            "country": countries,
            "segment": segments,
            "marketing_channel": channels,
            "birth_date": pd.to_datetime(birth_dates, errors="coerce"),
            "last_profile_update": pd.to_datetime(last_updates, errors="coerce"),
            # lifecycle_type is kept as a generation artifact; it is NOT a model feature
            "lifecycle_type": lifecycle_types,
        }
    )
    return df
