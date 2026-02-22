"""Central configuration and constants for the churn prediction system."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
PREDICTIONS_DIR: Path = ROOT_DIR / "predictions"
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
SEED: int = 42
START_DATE: str = "2021-01-01"
END_DATE: str = "2023-12-31"
N_CUSTOMERS: int = 2_000
N_PRODUCTS: int = 200

# ---------------------------------------------------------------------------
# Churn definition
# ---------------------------------------------------------------------------
# Days of inactivity that define a churned customer.
# Derived from P85 of inter-order duration distribution (≈ 90 days).
CHURN_PERIOD_DAYS: int = 90

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Rolling windows (in days) used for recency-based feature counts.
FEATURE_WINDOWS: list[int] = [30, 60, 90]

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
TEST_FRACTION: float = 0.20

XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": SEED,
}

# ---------------------------------------------------------------------------
# Evaluation guardrails
# ---------------------------------------------------------------------------
MIN_ROC_AUC: float = 0.65
MAX_BRIER_SCORE: float = 0.25
MAX_ECE: float = 0.10  # Expected Calibration Error
MIN_COVERAGE: float = 0.50  # Fraction of customers that must get a prediction

# ---------------------------------------------------------------------------
# Prediction risk buckets
# ---------------------------------------------------------------------------
RISK_HIGH: float = 0.70
RISK_MEDIUM: float = 0.40
