# Customer Churn Prediction

A production-ready machine learning system that forecasts which customers are likely to churn within the next 90 days.  The project covers every stage of the ML lifecycle: synthetic data generation, leakage-free feature engineering, temporal cross-validation, XGBoost model training, rule-based baseline comparison, automated quality guardrails, batch prediction, and an interactive Plotly Dash dashboard.

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Generate synthetic data
make data

# 3. (Optional) Tune hyperparameters
make tune

# 4. Train the model and run all evaluations
make train

# 5. Score all customers
make predict

# 6. Launch the dashboard
make dashboard
```

Or run the full pipeline in one shot:

```bash
make all        # data → train → predict
```

---

## Repository Layout

```text
customer_churn_prediction/
├── Makefile                         # All pipeline targets
├── pyproject.toml                   # Dependencies (uv)
├── Dockerfile                       # HuggingFace Spaces container
├── src/
│   ├── config.py                    # Central constants and thresholds
│   ├── utils/
│   │   ├── logger.py                # Structured logging
│   │   └── io.py                    # CSV / JSON helpers
│   ├── data_generation/
│   │   ├── generate_customers.py    # Synthetic customer dimension
│   │   ├── generate_products.py     # Synthetic product catalogue
│   │   ├── generate_orders.py       # Behavioural order simulation
│   │   └── main.py                  # Generation orchestrator
│   ├── preprocessing/
│   │   └── clean.py                 # Null handling, winsorization
│   ├── target/
│   │   └── churn_target.py          # Leakage-free binary target
│   ├── feature_engineering/
│   │   ├── rfm.py                   # Recency / Frequency / Monetary
│   │   ├── time_features.py         # Rolling windows and inter-order gaps
│   │   ├── trend_features.py        # Monthly slope features
│   │   ├── customer_attributes.py   # Tenure and demographic encoding
│   │   └── pipeline.py              # Assembles full feature matrix
│   ├── baseline/
│   │   └── rule_based.py            # Recency + activity rule
│   ├── models/
│   │   ├── train.py                 # XGBoost training, evaluation, artifacts
│   │   └── tune.py                  # Walk-forward CV random search
│   ├── evaluation/
│   │   ├── metrics.py               # ROC-AUC, PR-AUC, Brier, F1, …
│   │   ├── calibration.py           # Expected Calibration Error (ECE)
│   │   ├── guardrails.py            # 13 automated quality checks
│   │   └── distribution_shift.py    # Adversarial validation (KS + AUC)
│   └── prediction/
│       └── batch_predict.py         # Score all customers → risk buckets
├── dashboard/
│   └── app.py                       # Plotly Dash on port 7860
├── tests/                           # pytest unit tests (44+ tests)
├── data/                            # Generated CSVs (gitignored)
├── artifacts/                       # Model + evaluation artifacts (gitignored)
└── predictions/                     # Batch prediction output (gitignored)
```

---

## Pipeline Steps

### Step 1 — Generate Data (`make data`)

`src/data_generation/` creates three relational CSV files under `data/`:

| File | Rows | Description |
| ---- | ---- | ----------- |
| `customers.csv` | 2,000 | Demographics, registration date, lifecycle type |
| `products.csv` | 200 | Product type, Pareto-distributed price, margin group |
| `orders.csv` | ~54,000 | Transaction history 2021-01-01 – 2023-12-31 |

**Customer lifecycle types** drive order behaviour:

| Lifecycle       | Weight | Registration window | Behaviour                  |
| --------------- | ------ | ------------------- | -------------------------- |
| loyal           | 30 %   | Full 3-year range   | High, stable order rate    |
| seasonal        | 20 %   | Full 3-year range   | Holiday-amplified peaks    |
| early_churner   | 15 %   | Full 3-year range   | Rapid frequency decay      |
| churn_returner  | 15 %   | Full 3-year range   | Periodic re-engagement     |
| late_joiner     | 20 %   | Year 2 onwards      | Newer customers            |

All non-late-joiner registrations are spread across the full three-year period so that both the train and test splits share the same registration-date range (no cohort confound).

Order inter-arrival times use a 70 % Exponential + 30 % Pareto mixture to reproduce the heavy tail seen in real e-commerce data.  Seasonal (Oct–Dec: 1.4×) and holiday (Christmas, Black Friday: 1.5×) multipliers are applied.

---

### Step 2 — Preprocess (`src/preprocessing/clean.py`)

Applied automatically before all downstream steps:

- Drop rows missing `order_date` or `customer_id`
- Winsorise `total_value` at 2× P99 to clip outlier orders
- Enforce `quantity ≥ 1`, `price ≥ $0.01`
- Drop customers missing `registration_date`

---

### Step 3 — Build Churn Target (`src/target/churn_target.py`)

The 90-day inactivity definition is derived from the P85 of the inter-order duration distribution.

```text
T_max            = last order date in dataset
churn_window     = (T_max − 90 d,  T_max]
eligible         = customers with ≥ 1 order BEFORE churn_window_start
is_churned = 1   ⟺  customer placed NO orders inside churn_window
feature_end_date = churn_window_start  (exclusive upper bound for features)
```

This construction guarantees **zero target leakage**: all features are computed on data strictly before `feature_end_date`.

---

### Step 4 — Feature Engineering (`src/feature_engineering/`)

All sub-modules accept a `reference_date` and filter `order_date < reference_date` internally, so they can be called safely during temporal cross-validation as well as final training.

| Module | Features |
| ------ | -------- |
| `rfm.py` | `recency`, `frequency`, `monetary`, `avg_order_value`, `max_order_value` |
| `time_features.py` | `avg_inter_order_days`, `std_inter_order_days`, `orders_last_{30,60,90}d`, `spend_last_{30,60,90}d`, `ratio_recent_historical` |
| `trend_features.py` | `order_frequency_slope`, `spend_slope`, `rolling_avg_orders_3m`, `spend_change_ratio` |
| `customer_attributes.py` | `tenure_days`, `segment_encoded`, `channel_encoded`, `country_encoded` |

`pipeline.py` joins all four blocks on `customer_id` and fills residual NaNs with 0.

---

### Step 5 — Train/Test Split (`src/models/train.py: time_based_split`)

Customers are sorted by `registration_date` and assigned to splits using **interleaved stratified sampling**: every 5th customer (by registration order) goes to the test set, the rest to train.

This ensures:

- Both splits cover the **same** registration-date range
- No lifecycle-type confound (avoids placing all late-joiners in test)
- ~80 / 20 train/test ratio maintained

---

### Step 6 — Tune Hyperparameters (`make tune`)

`src/models/tune.py` runs a **walk-forward temporal random search** over 30 parameter combinations with 3 CV folds.

```text
usable_start = t_min + 90 d
usable_end   = t_max_train − 90 d
step         = (usable_end − usable_start) / (n_splits + 1)

Fold k:
  feature_end  = usable_start + (k+1) × step
  fold_t_max   = feature_end + 90 d
  eligible     = customers with ≥1 order before feature_end
  train fold   = earliest 80 % (by registration_date)
  val fold     = latest  20 %
  Features     recomputed at feature_end   ← zero leakage
  Labels       recomputed at fold_t_max    ← zero leakage
```

Fold datasets are precomputed once; all 30 trials reuse them.  Best parameters are saved to `artifacts/best_params.json`.

**Search space:**

| Parameter | Values |
| --------- | ------ |
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 4, 5 |
| `learning_rate` | 0.01, 0.03, 0.05, 0.10 |
| `subsample` | 0.70, 0.80, 0.90 |
| `colsample_bytree` | 0.70, 0.80, 0.90 |
| `min_child_weight` | 3, 5, 10 |
| `gamma` | 0.0, 0.1, 0.3 |
| `reg_alpha` | 0.0, 0.1, 0.5 |

---

### Step 7 — Train Model (`make train`)

`src/models/train.py` executes the full training and evaluation pipeline:

1. Load best-tuned params from `artifacts/best_params.json` (or fall back to defaults)
2. Fit `XGBClassifier` on the training split
3. Compute test-set metrics: ROC-AUC, PR-AUC, Brier score, precision, recall, F1
4. Compute train-set metrics to detect overfitting (train − test gap)
5. Run **adversarial validation** to detect covariate shift between train and test
6. Evaluate the **rule-based baseline** on the same test set
7. Log a side-by-side ML vs baseline comparison across all 6 metrics
8. Run all 13 **automated guardrails** (see below)
9. Save model to `artifacts/xgb_churn_model.pkl`
10. Save evaluation artifacts for the dashboard

Default XGBoost parameters (before tuning):

```python
n_estimators=200, max_depth=3, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
min_child_weight=5, gamma=0.1, reg_alpha=0.1
```

---

### Step 8 — Rule-Based Baseline (`src/baseline/rule_based.py`)

A simple heuristic evaluated on the same test set as the ML model:

```text
predict churned  ⟺  recency > 90 days  OR  orders_last_60d == 0
```

A continuous score (normalised recency, boosted when `orders_last_60d == 0`) enables fair ROC-AUC and PR-AUC comparison.  The ML model must not lag the baseline by more than 0.05 on either metric (enforced by guardrails).

---

### Step 9 — Guardrails (`src/evaluation/guardrails.py`)

13 automated checks run after every training.  A failure is logged as `[FAIL]` but does not halt execution.

| Check                   | Threshold  | What it tests                         |
| ----------------------- | ---------- | ------------------------------------- |
| ROC-AUC                 | ≥ 0.65     | Basic discrimination                  |
| Brier score             | ≤ 0.25     | Probabilistic accuracy                |
| Calibration (ECE)       | ≤ 0.10     | Reliability of probabilities          |
| Prediction coverage     | ≥ 50 %     | Fraction of customers scored          |
| PR-AUC                  | ≥ 0.20     | Precision-recall under imbalance      |
| Train–test ROC-AUC gap  | ≤ 0.15     | Overfitting                           |
| ML vs baseline ROC-AUC  | gap ≤ 0.05 | ML at least as good as heuristic      |
| ML vs baseline PR-AUC   | gap ≤ 0.05 | Same for PR-AUC                       |
| Baseline ROC-AUC        | ≥ 0.65     | Baseline sanity                       |
| Baseline PR-AUC         | ≥ 0.40     | Baseline precision-recall quality     |
| Baseline precision      | ≥ 0.55     | Baseline precision floor              |
| Baseline recall         | ≥ 0.55     | Baseline recall floor                 |
| Adversarial AUC         | ≤ 0.70     | Train/test distribution shift         |

---

### Step 10 — Batch Prediction (`make predict`)

`src/prediction/batch_predict.py` scores every eligible customer:

1. Rebuild the feature matrix at the current `feature_end_date`
2. Load `artifacts/xgb_churn_model.pkl`
3. Predict churn probability for each customer
4. Assign a risk bucket:

| Bucket | Probability  |
| ------ | ------------ |
| high   | ≥ 0.70       |
| medium | 0.40 – 0.69  |
| low    | < 0.40       |

Output: `predictions/churn_predictions.csv` (`customer_id`, `churn_probability`, `risk_bucket`)

---

### Step 11 — Dashboard (`make dashboard`)

A Plotly Dash application on `http://localhost:7860` visualises:

- Churn probability distribution with risk-bucket breakdown
- Feature importance bar chart
- ROC curve
- Calibration plot (reliability diagram)
- Guardrail check results
- ML vs baseline metric comparison
- Distribution shift report (adversarial AUC + per-feature KS statistics)

---

## Testing

```bash
make test             # Run all tests with coverage
make test-verbose     # Verbose output
```

Test files and coverage:

| File | Tests | Covers |
| ---- | ----- | ------ |
| `test_data_generation.py` | 21 | Customers, products, orders generation |
| `test_features.py` | 17 | RFM, time, trend, attribute, pipeline |
| `test_target.py` | 7 | Churn label construction + leakage cases |
| `test_baseline.py` | 9 | Rule predictions and evaluation metrics |
| `test_guardrails.py` | 44 | All 13 guardrail checks + orchestration |

---

## Code Quality

```bash
make lint             # Ruff linter
make format           # Auto-format with Ruff
make security         # Bandit static analysis (medium+ severity)
make complexity       # Radon cyclomatic complexity (B-grade minimum)
make audit            # pip-audit dependency vulnerability scan
make ci               # All of the above (no file writes)
```

Pre-commit hooks (`.pre-commit-config.yaml`) run Ruff check + format on every commit.

---

## Configuration

All tuneable constants live in `src/config.py`:

```python
CHURN_PERIOD_DAYS = 90             # Inactivity window (days)
FEATURE_WINDOWS   = [30, 60, 90]   # Rolling aggregation windows
TEST_FRACTION     = 0.20           # Train/test split ratio
SEED              = 42

# Guardrail thresholds
MIN_ROC_AUC              = 0.65
MAX_BRIER_SCORE          = 0.25
MAX_ECE                  = 0.10
MIN_PR_AUC               = 0.20
MAX_TRAIN_TEST_ROC_GAP   = 0.15
MAX_ADVERSARIAL_AUC      = 0.70

# Risk buckets
RISK_HIGH   = 0.70
RISK_MEDIUM = 0.40
```

---

## Deployment (HuggingFace Spaces)

Build and run the Docker image locally:

```bash
make docker-build
make docker-run       # serves on http://localhost:7860
```

The `Dockerfile` runs the Dash app as a non-root user (`appuser`, uid 1000) to satisfy HuggingFace Spaces requirements.  Generate model artifacts locally with `make all` before building the image; `predictions/` and `artifacts/` are copied into the container.

---

## Key Design Decisions

**Zero temporal leakage** — Feature sub-modules filter `order_date < reference_date` internally, so the same functions are safe to call during both CV folds and final training without any risk of future data contaminating the feature values.

**Walk-forward CV instead of k-fold** — Hyperparameter tuning uses expanding-window temporal folds rather than shuffled k-fold, preserving the temporal ordering that governs real churn dynamics.

**Cohort-aware train/test split** — The split interleaves customers sorted by `registration_date` instead of cutting at a fixed date.  This prevents the cohort confound where a hard temporal cut places all late-joiners (with fewer orders and different churn patterns) exclusively in the test set.

**Shallow, regularised XGBoost** — `max_depth=3`, `min_child_weight=5`, `gamma=0.1`, `reg_alpha=0.1` reduce the risk of memorising noisy customer histories on a dataset of 2,000 customers.

**Dual-track evaluation** — Every training run compares the ML model against the rule-based baseline on the identical test set.  A guardrail enforces that the ML model does not regress below the heuristic.
