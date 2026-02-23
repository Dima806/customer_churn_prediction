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

# Sliding-window training-data augmentation.
# Each training customer generates AUG_N_WINDOWS additional (features, label) rows by
# anchoring the churn observation window at earlier points in time, stepping back
# AUG_STEP_DAYS at a time.  Features for every window are computed strictly before
# that window's reference date, so there is no label leakage.
AUG_N_WINDOWS: int = 5  # additional temporal snapshots per training customer
AUG_STEP_DAYS: int = 30  # days between consecutive augmented windows

XGBOOST_PARAMS: dict = {
    "n_estimators": 200,  # was 300; fewer trees, less memorisation
    "max_depth": 3,  # was 5: shallower trees generalise better
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,  # min samples per leaf; prevents tiny splits
    "gamma": 0.1,  # min gain required to make a split
    "reg_alpha": 0.1,  # L1 regularisation on leaf weights
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

# --- ML model additional guardrails ---
MIN_PR_AUC: float = 0.20  # Precision-Recall AUC (harder bar under class imbalance)
MAX_TRAIN_TEST_ROC_GAP: float = 0.15  # Max allowed gap between train and test ROC-AUC

# --- ML vs baseline ---
ML_BASELINE_TOLERANCE: float = 0.05  # ML ROC-AUC may not lag baseline by more than this
ML_BASELINE_PR_AUC_TOLERANCE: float = 0.05  # ML PR-AUC lag tolerance

# --- Distribution shift (adversarial validation) ---
MAX_ADVERSARIAL_AUC: float = 0.70  # AUC > threshold signals train/test shift

# --- Baseline quality guardrails ---
BASELINE_MIN_ROC_AUC: float = 0.65
BASELINE_MIN_PR_AUC: float = 0.40
BASELINE_MIN_PRECISION: float = 0.55
BASELINE_MIN_RECALL: float = 0.55

# ---------------------------------------------------------------------------
# Prediction risk buckets
# ---------------------------------------------------------------------------
RISK_HIGH: float = 0.70
RISK_MEDIUM: float = 0.40
