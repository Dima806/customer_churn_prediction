"""Synthetic product dimension table generator."""

import numpy as np
import pandas as pd

from src.config import N_PRODUCTS, SEED

PRODUCT_TYPES: list[str] = [
    "electronics",
    "clothing",
    "food_beverage",
    "home_garden",
    "sports",
    "beauty",
    "books",
    "toys",
]
PRODUCT_TYPE_WEIGHTS: list[float] = [0.20, 0.18, 0.15, 0.13, 0.12, 0.10, 0.07, 0.05]

MARGIN_GROUPS: list[str] = ["low", "medium", "high"]
MARGIN_WEIGHTS: list[float] = [0.30, 0.50, 0.20]


def generate_products(
    n: int = N_PRODUCTS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate a synthetic product dimension table.

    Prices follow a power-law (Pareto) distribution to reflect real-world
    SKU catalogues where a few expensive items coexist with many cheap ones.

    Args:
        n: Number of products to generate.
        seed: Random seed.

    Returns:
        DataFrame with one row per product.
    """
    rng = np.random.default_rng(seed)

    product_types = rng.choice(PRODUCT_TYPES, size=n, p=PRODUCT_TYPE_WEIGHTS)
    margin_groups = rng.choice(MARGIN_GROUPS, size=n, p=MARGIN_WEIGHTS)

    # Power-law prices: Pareto shape=1.5, minimum=$5
    raw_prices = (rng.pareto(1.5, size=n) + 1) * 5.0
    # Cap at $2 000 to avoid extreme outliers ruining the scale
    base_prices = np.clip(raw_prices, 5.0, 2_000.0).round(2)

    df = pd.DataFrame(
        {
            "product_id": range(1, n + 1),
            "product_type": product_types,
            "base_price": base_prices,
            "margin_group": margin_groups,
        }
    )
    return df
