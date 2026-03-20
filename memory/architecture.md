# Architecture: Zero-Leakage Churn Pipeline

## Core Invariant

Features must **never** see data from after the churn observation window. This is enforced at every layer.

## Churn Definition

```
T_max            = last order date in the dataset
churn_window     = (T_max − 90d, T_max]
feature_end_date = T_max − 90d   ← all features computed BEFORE this date
is_churned       = 1 if no orders in churn_window, else 0
```

Only customers with at least one order **before** the churn window are eligible (new-to-churn-window customers cannot be labelled).

Key function: `compute_churn_target(orders, customers)` in [src/target/churn_target.py](../src/target/churn_target.py)
Returns: `(target_df, feature_end_date, t_max)`

## Feature Engineering

Entry point: `build_feature_matrix(orders, customers, reference_date)` in [src/feature_engineering/pipeline.py](../src/feature_engineering/pipeline.py)

Four sub-modules, each filtering `order_date < reference_date` internally:
- `rfm.py` — Recency, Frequency, Monetary features
- `time_features.py` — Rolling window counts (30/60/90-day windows from `FEATURE_WINDOWS`)
- `trend_features.py` — Order trend/velocity signals
- `customer_attributes.py` — Static customer demographics

All sub-modules return `DataFrame` with `customer_id` as first column, one row per customer. `pipeline.py` left-merges them and fills NaN with 0.

## Train/Test Split

`time_based_split` in [src/models/train.py](../src/models/train.py):
- Sorts customers by `registration_date`
- Every `round(1/test_fraction)`-th customer (index `% 5 == 4`) → test set
- **Interleaved** — not a hard temporal cut — both splits cover the full registration range
- Hard temporal cut is explicitly banned: it causes cohort confound where late-joiners all end up in test set, creating artificially low test churn rates

## Data Augmentation (Sliding Window)

`build_augmented_training_set` in [src/models/train.py](../src/models/train.py):
- Generates `AUG_N_WINDOWS=5` extra `(features, label)` rows per training customer
- Each window steps back `AUG_STEP_DAYS=30` days from `feature_end_date`
- Features computed strictly before `aug_feature_end`, labels from `(aug_feature_end, aug_t_max]`
- **Test-set customers are never included** in augmentation
- Empty DataFrames are returned gracefully if no eligible customers

## Hyperparameter Tuning CV

`tune_hyperparameters` in [src/models/tune.py](../src/models/tune.py):
- Walk-forward temporal folds (NOT StratifiedKFold / shuffled CV)
- For each fold: features AND labels are recomputed at that fold's temporal cutpoints
- Fold datasets pre-computed once, reused across all trials for efficiency
- Results saved to `artifacts/best_params.json` with `{"params": ..., "score": ...}`

## Key Config Constants

All in [src/config.py](../src/config.py):

| Constant | Value | Purpose |
|---|---|---|
| `CHURN_PERIOD_DAYS` | 90 | Churn observation window length |
| `FEATURE_WINDOWS` | [30, 60, 90] | Rolling window sizes for time features |
| `AUG_N_WINDOWS` | 5 | Augmentation windows per training customer |
| `AUG_STEP_DAYS` | 30 | Days between augmented windows |
| `TEST_FRACTION` | 0.20 | Fraction of customers for test set |
| `SEED` | 42 | Global random seed |
| `RISK_HIGH` | 0.70 | High-risk probability threshold |
| `RISK_MEDIUM` | 0.40 | Medium-risk probability threshold |
