"""File I/O helpers for data, artifacts, and predictions."""

import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR, PREDICTIONS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_orders() -> pd.DataFrame:
    """Load orders fact table from CSV."""
    path = DATA_DIR / "orders.csv"
    if not path.exists():
        raise FileNotFoundError(f"Orders file not found: {path}. Run `make data` first.")
    df = pd.read_csv(path, parse_dates=["order_date", "contract_date"])
    logger.info(f"Loaded {len(df):,} orders from {path}")
    return df


def load_customers() -> pd.DataFrame:
    """Load customer dimension table from CSV."""
    path = DATA_DIR / "customers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Customers file not found: {path}. Run `make data` first.")
    df = pd.read_csv(
        path,
        parse_dates=["registration_date", "birth_date", "last_profile_update"],
    )
    logger.info(f"Loaded {len(df):,} customers from {path}")
    return df


def load_products() -> pd.DataFrame:
    """Load product dimension table from CSV."""
    path = DATA_DIR / "products.csv"
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {path}. Run `make data` first.")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} products from {path}")
    return df


# ---------------------------------------------------------------------------
# JSON artifact helpers
# ---------------------------------------------------------------------------


def save_json(data: dict | list, filename: str) -> Path:
    """Persist a JSON-serialisable object to the artifacts directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    logger.info(f"Saved {filename} to {path}")
    return path


def load_json(filename: str) -> dict | list | None:
    """Load a JSON artifact; returns None if the file does not exist."""
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


def save_predictions(predictions: pd.DataFrame) -> Path:
    """Write batch predictions to the predictions directory."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / "churn_predictions.csv"
    predictions.to_csv(path, index=False)
    logger.info(f"Saved {len(predictions):,} predictions to {path}")
    return path
