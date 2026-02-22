"""Synthetic orders fact table generator.

Design:
- Each customer is simulated through an activity lifecycle (loyal, seasonal,
  early_churner, churn_returner, late_joiner).
- Within each active period, inter-order times come from a mixture:
    70 % Exponential  (frequent buyers)
    30 % Pareto       (heavy-tail / infrequent buyers)
- Seasonal and holiday boosts reduce effective inter-order time.
"""

from __future__ import annotations

import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import END_DATE, SEED, START_DATE

# ---------------------------------------------------------------------------
# Holiday boost dates (fixed calendar events that lift ordering rates)
# ---------------------------------------------------------------------------

_HOLIDAY_MONTHS_DAYS: list[tuple[int, int]] = [
    (12, 20),
    (12, 21),
    (12, 22),
    (12, 23),
    (12, 24),
    (12, 25),
    (12, 26),
    (12, 27),
    (12, 28),
    (12, 29),
    (12, 30),
    (12, 31),  # Christmas–NY
    (11, 24),
    (11, 25),
    (11, 26),
    (11, 27),
    (11, 28),
    (11, 29),
    (11, 30),  # Black Fri
    (10, 30),
    (10, 31),  # Halloween
    (2, 13),
    (2, 14),  # Valentine's
]


def _build_holiday_set(start: pd.Timestamp, end: pd.Timestamp) -> set[datetime.date]:
    """Return all calendar holiday dates between start and end."""
    dates: set[datetime.date] = set()
    for year in range(start.year, end.year + 1):
        for month, day in _HOLIDAY_MONTHS_DAYS:
            try:
                dates.add(datetime.date(year, month, day))
            except ValueError:
                pass
    return dates


def _seasonal_multiplier(month: int) -> float:
    """Ordering-rate multiplier based on calendar month."""
    return {
        1: 0.80,
        2: 0.90,
        3: 1.00,
        4: 1.10,
        5: 1.10,
        6: 1.00,
        7: 0.90,
        8: 0.90,
        9: 1.00,
        10: 1.20,
        11: 1.40,
        12: 1.50,
    }.get(month, 1.0)


# ---------------------------------------------------------------------------
# Activity-period helpers
# ---------------------------------------------------------------------------


class Period(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp


def _activity_periods(
    lifecycle: str,
    reg: pd.Timestamp,
    end: pd.Timestamp,
    rng: np.random.Generator,
) -> list[Period]:
    """Return the list of (start, end) active intervals for a customer."""

    if lifecycle == "loyal":
        return [Period(reg, end)]

    if lifecycle == "late_joiner":
        return [Period(reg, end)]

    if lifecycle == "early_churner":
        churn_after = int(rng.integers(60, 270))  # 2–9 months
        churn_date = reg + pd.Timedelta(days=churn_after)
        return [Period(reg, min(churn_date, end))]

    if lifecycle == "churn_returner":
        first_active = int(rng.integers(90, 365))
        gap = int(rng.integers(180, 540))
        churn_date = reg + pd.Timedelta(days=first_active)
        return_date = churn_date + pd.Timedelta(days=gap)
        periods = [Period(reg, min(churn_date, end))]
        if return_date < end:
            periods.append(Period(return_date, end))
        return periods

    if lifecycle == "seasonal":
        # Active Q4 (Oct–Dec) and Q2 (Apr–Jun) each year
        periods: list[Period] = []
        for year in range(reg.year, end.year + 1):
            for m_start, m_end_day in [(10, (12, 31)), (4, (6, 30))]:
                p_start = pd.Timestamp(year, m_start, 1)
                p_end = pd.Timestamp(year, m_end_day[0], m_end_day[1])
                p_start = max(p_start, reg)
                p_end = min(p_end, end)
                if p_start <= p_end:
                    periods.append(Period(p_start, p_end))
        return sorted(periods, key=lambda p: p.start)

    return [Period(reg, end)]  # fallback


# ---------------------------------------------------------------------------
# Per-customer order simulation
# ---------------------------------------------------------------------------

_BASE_RATES: dict[str, float] = {
    "loyal": 1 / 10,
    "seasonal": 1 / 12,
    "early_churner": 1 / 8,
    "churn_returner": 1 / 11,
    "late_joiner": 1 / 13,
}

_CHANNEL_MULTIPLIERS: dict[str, float] = {
    "paid_search": 1.20,
    "email": 1.10,
    "social": 1.00,
    "organic": 0.90,
}


def _simulate_customer(
    customer_id: int,
    lifecycle: str,
    reg: pd.Timestamp,
    channel: str,
    products: pd.DataFrame,
    rng: np.random.Generator,
    end: pd.Timestamp,
    holidays: set[datetime.date],
) -> list[dict]:
    """Simulate all orders for a single customer."""
    base_rate = _BASE_RATES.get(lifecycle, 1 / 12)
    ch_mult = _CHANNEL_MULTIPLIERS.get(channel, 1.0)
    rate = base_rate * ch_mult

    n_products = len(products)
    records: list[dict] = []

    for period in _activity_periods(lifecycle, reg, end, rng):
        current = period.start

        while current <= period.end:
            # Effective rate adjusted for season + holiday
            s_mult = _seasonal_multiplier(current.month)
            h_mult = 1.5 if current.date() in holidays else 1.0
            eff_rate = rate * s_mult * h_mult

            # Inter-order time from mixture distribution
            if rng.random() < 0.70:
                inter = rng.exponential(1.0 / eff_rate)
            else:
                # Pareto heavy tail
                inter = (rng.pareto(1.5) + 1.0) * (1.5 / eff_rate)

            current = current + pd.Timedelta(days=max(1, int(inter)))
            if current > period.end:
                break

            # Product selection (5 % missing product_id)
            prod_idx = int(rng.integers(0, n_products))
            product = products.iloc[prod_idx]
            product_id: int | None = None if rng.random() < 0.05 else int(product["product_id"])

            # Quantity: power-law (Pareto + 1), clipped at 50
            quantity = min(50, max(1, int(rng.pareto(2.0) + 1)))

            # Price: base ± 10 % noise
            price = float(product["base_price"]) * (1.0 + rng.normal(0, 0.10))
            price = round(max(price, 0.01), 2)
            total_value = round(price * quantity, 2)

            # Contract date: nullable (~20 % missing)
            contract_date: pd.Timestamp | None = None
            if rng.random() > 0.20:
                contract_days = int(rng.integers(7, 91))
                contract_date = current + pd.Timedelta(days=contract_days)

            records.append(
                {
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "order_date": current,
                    "contract_date": contract_date,
                    "quantity": quantity,
                    "price": price,
                    "total_value": total_value,
                    "holiday_flag": int(h_mult > 1.0),
                    "seasonal_flag": int(s_mult > 1.10),
                }
            )

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_orders(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    seed: int = SEED,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Generate the synthetic orders fact table.

    Args:
        customers: Customer dimension DataFrame (output of ``generate_customers``).
        products: Product dimension DataFrame (output of ``generate_products``).
        seed: Random seed.
        start_date: Data window start.
        end_date: Data window end.

    Returns:
        Orders DataFrame sorted by ``order_date``.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date)
    holidays = _build_holiday_set(pd.Timestamp(start_date), end)

    all_records: list[dict] = []
    for _, cust in customers.iterrows():
        records = _simulate_customer(
            customer_id=int(cust["customer_id"]),
            lifecycle=str(cust["lifecycle_type"]),
            reg=pd.Timestamp(cust["registration_date"]),
            channel=str(cust["marketing_channel"]),
            products=products,
            rng=rng,
            end=end,
            holidays=holidays,
        )
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df = df.sort_values("order_date").reset_index(drop=True)
    df.insert(0, "order_id", range(1, len(df) + 1))
    return df
