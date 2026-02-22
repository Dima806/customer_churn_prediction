# CLAUDE.md — Development Guide

## Essential Commands

```bash
# Setup
make install           # uv sync --all-groups

# Data
make data              # generate synthetic CSVs into data/

# ML pipeline
make tune              # walk-forward CV hyperparameter search → artifacts/best_params.json
make train             # train XGBoost, evaluate, save model + artifacts
make predict           # batch-score all customers → predictions/churn_predictions.csv
make all               # data → train → predict

# Tests
make test              # pytest with coverage (quiet)
make test-verbose      # pytest with coverage (verbose)

# Code quality (run before committing)
make lint              # ruff check
make format            # ruff format (writes files)
make format-check      # ruff format --check (read-only, used in CI)
make security          # bandit -r src/ --severity-level medium
make ci                # lint + format-check + security + test

# Dashboard
make dashboard         # Plotly Dash on http://localhost:7860
```

All commands use `uv run`; never call `python` directly.

## Architecture: Zero-Leakage Pipeline

The entire system is built around one invariant: **features must never see data from after the churn observation window.**

### How it works

```
T_max            = last order date in the dataset
churn_window     = (T_max − 90 d, T_max]
feature_end_date = T_max − 90 d        ← all features computed BEFORE this date
is_churned       = no orders in churn_window
```

Every feature sub-module (`rfm.py`, `time_features.py`, `trend_features.py`, `customer_attributes.py`) filters `order_date < reference_date` internally. This makes them safe to call during CV folds and final training using the same code path.

**Never** pass a `reference_date` that is later than the intended feature cutoff. **Never** compute features on the full orders table without a reference date.

### Train/test split

`time_based_split` in `src/models/train.py` uses **interleaved stratified sampling**: customers are sorted by `registration_date`, and every 5th one (index `% 5 == 4`) is assigned to test. This ensures both splits cover the same registration-date range. Do not revert to a hard temporal cut — it causes a cohort confound where all late-joiners end up in the test set, producing artificially low test churn rates.

### Hyperparameter tuning CV

`tune.py` uses walk-forward temporal folds, not `StratifiedKFold`. For each fold, both features **and** labels are recomputed at that fold's temporal cutpoints (`fold_feature_end` and `fold_t_max`). Do not replace this with any form of shuffled cross-validation.

## Code Style

- **Line length**: 100 characters (enforced by ruff)
- **Python**: 3.12+, use modern syntax (`X | Y` unions, `match`, `list[int]` lowercase generics)
- **Imports**: isort-ordered by ruff (`I` rules); stdlib → third-party → local
- **ML variable names**: `X_train`, `X_test`, `y_train`, `y_test` are intentionally uppercase (N803/N806 are ignored in ruff config)
- **Logging**: always use `get_logger(__name__)` from `src/utils/logger.py`; never `print()`
- **No asserts in production code**: use explicit `raise ValueError` / `raise RuntimeError` with clear messages; `assert` is only allowed in `tests/`

## Project Conventions

### Configuration
All constants live in `src/config.py`. Never hardcode thresholds, paths, or seeds in individual modules. When adding a new guardrail threshold or model parameter, add it to `config.py` first.

### Feature engineering
Each sub-module returns a `pd.DataFrame` with `customer_id` as the first column, one row per customer. `pipeline.py` left-merges them in order and fills NaN with 0. New features go in the most appropriate existing module or a new module that follows the same `(orders, reference_date) → DataFrame` signature.

### Guardrails
`src/evaluation/guardrails.py` returns `list[dict]` with keys `check`, `value`, `threshold`, `passed`, `message`. Failures are logged but do not raise. When adding a new guardrail:
1. Add the threshold constant to `config.py`
2. Write a `check_*` function that returns the standard dict
3. Add it to `run_all_guardrails`
4. Add corresponding tests in `tests/test_guardrails.py`

### Artifacts
Files written during training (model, metrics, calibration, etc.) go under `artifacts/`. Batch predictions go under `predictions/`. Both directories are gitignored. The dashboard reads exclusively from these two directories.

### Data generation
`lifecycle_type` is a generation artifact used to drive behavioural simulation. It is **not** a model feature and must not appear in the feature matrix. It is stripped from the CSV in `main.py`.

## Testing

Tests live in `tests/` and mirror the `src/` module structure. Run a single test file:

```bash
uv run pytest tests/test_guardrails.py -v
```

Run a single test by name:

```bash
uv run pytest tests/test_guardrails.py::TestRunAllGuardrails::test_all_checks_present -v
```

Key test patterns:
- Leakage tests in `test_features.py` verify that orders at or after `reference_date` do not affect feature values
- Guardrail tests use explicit numeric inputs to verify pass/fail boundaries, not model fixtures
- Data generation tests check for nullable columns using `isna().mean()` bounds, not exact counts (the counts are stochastic)

## Dependencies

Managed by `uv` with `pyproject.toml`. To add a dependency:

```bash
uv add <package>           # runtime
uv add --dev <package>     # dev-only
```

Runtime: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `plotly`, `dash`, `gunicorn`
Dev: `pytest`, `pytest-cov`, `ruff`, `bandit`, `radon`, `pip-audit`, `pre-commit`

## Pre-commit Hooks

`.pre-commit-config.yaml` runs ruff check + format on every commit. To install:

```bash
uv run pre-commit install
```

The CI target (`make ci`) runs the same checks without writing files, suitable for automated pipelines.

## Common Pitfalls

- **Forgetting `reference_date`**: `build_feature_matrix` requires a `reference_date`. In the `__main__` block this is `feature_end_date` returned by `compute_churn_target`.
- **Using the full orders table for labels**: `compute_churn_target` must receive the full orders table so it can determine `T_max` correctly. During CV, labels are anchored by passing a filtered table (`orders[order_date <= fold_t_max]`).
- **Checking only `scale_pos_weight`**: class imbalance is deliberately not corrected via `scale_pos_weight` (it is left at 1.0 in both training and CV). Do not add automatic `scale_pos_weight` computation.
- **Hard temporal cut in `time_based_split`**: see Architecture section above. The interleaved split is intentional.
- **Writing to `data/`, `artifacts/`, or `predictions/` from tests**: these dirs are gitignored runtime outputs; tests must not write there.
