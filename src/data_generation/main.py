"""Orchestrates synthetic data generation and persists CSVs to data/."""

from src.config import DATA_DIR, END_DATE, N_CUSTOMERS, N_PRODUCTS, SEED, START_DATE
from src.data_generation.generate_customers import generate_customers
from src.data_generation.generate_orders import generate_orders
from src.data_generation.generate_products import generate_products
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_all(
    n_customers: int = N_CUSTOMERS,
    n_products: int = N_PRODUCTS,
    seed: int = SEED,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple:
    """Run the full synthetic data generation pipeline.

    Returns:
        Tuple of (customers, products, orders) DataFrames.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_customers:,} customers …")
    customers = generate_customers(
        n=n_customers, seed=seed, start_date=start_date, end_date=end_date
    )

    logger.info(f"Generating {n_products:,} products …")
    products = generate_products(n=n_products, seed=seed)

    logger.info("Generating orders (this may take a moment) …")
    orders = generate_orders(
        customers=customers,
        products=products,
        seed=seed,
        start_date=start_date,
        end_date=end_date,
    )

    # Strip the internal simulation label before persisting
    customers_out = customers.drop(columns=["lifecycle_type"])

    customers_out.to_csv(DATA_DIR / "customers.csv", index=False)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    logger.info(
        f"Data saved to {DATA_DIR}: "
        f"{len(customers_out):,} customers, "
        f"{len(products):,} products, "
        f"{len(orders):,} orders"
    )

    _log_summary(customers, orders)
    return customers, products, orders


def _log_summary(customers: "pd.DataFrame", orders: "pd.DataFrame") -> None:  # noqa: F821
    """Log descriptive statistics useful for churn-period calibration."""
    import numpy as np

    hist = orders.sort_values(["customer_id", "order_date"])
    hist["prev"] = hist.groupby("customer_id")["order_date"].shift(1)
    hist["inter_days"] = (hist["order_date"] - hist["prev"]).dt.days
    inter = hist["inter_days"].dropna()

    logger.info(
        f"Inter-order time  median={np.median(inter):.0f}d  "
        f"P75={np.percentile(inter, 75):.0f}d  "
        f"P80={np.percentile(inter, 80):.0f}d  "
        f"P90={np.percentile(inter, 90):.0f}d"
    )
    lc = customers.groupby("lifecycle_type").size().to_dict()
    logger.info(f"Lifecycle mix: {lc}")


if __name__ == "__main__":
    generate_all()
