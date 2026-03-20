# Project Conventions

## Code Style

- **Line length**: 100 characters (enforced by ruff)
- **Python**: 3.12+, modern syntax: `X | Y` unions, `match`, `list[int]` lowercase generics
- **Imports**: isort-ordered (`I` rules): stdlib → third-party → local
- **ML variable names**: `X_train`, `X_test`, `y_train`, `y_test` intentionally uppercase (N803/N806 ignored)
- **Logging**: always `get_logger(__name__)` from `src/utils/logger.py`; never `print()`
- **No asserts in production**: use `raise ValueError` / `raise RuntimeError` with clear messages; `assert` only in `tests/`

## Configuration Pattern

All constants in `src/config.py`. Never hardcode thresholds, paths, or seeds in individual modules. When adding a guardrail threshold or model parameter, add to `config.py` first.

## Feature Engineering Pattern

Each sub-module signature: `(orders, reference_date) → DataFrame` with `customer_id` as first column, one row per customer. `pipeline.py` left-merges in order and fills NaN with 0.

New features go in the most appropriate existing module or a new module following the same signature.

## Guardrail Pattern

See [evaluation.md](./evaluation.md#adding-a-new-guardrail) for the full workflow.

## Artifact Storage

- Training outputs (model, metrics, calibration, etc.) → `artifacts/`
- Batch predictions → `predictions/`
- Both directories are gitignored runtime outputs
- Dashboard reads **exclusively** from these two directories

## Data Generation Note

`lifecycle_type` is a generation artifact used to drive behavioural simulation. It is **not** a model feature and must not appear in the feature matrix. It is stripped from the CSV in `src/data_generation/main.py`.

## Dependency Management

Use `uv`, never call `python` directly — always `uv run`. Add dependencies:
```bash
uv add <package>        # runtime
uv add --dev <package>  # dev-only
```

## Testing Patterns

- Tests mirror `src/` module structure under `tests/`
- Leakage tests in `test_features.py`: verify orders at/after `reference_date` don't affect features
- Guardrail tests use explicit numeric inputs, not model fixtures
- Data generation tests use `isna().mean()` bounds, not exact counts (counts are stochastic)
- **Never write to `data/`, `artifacts/`, or `predictions/` from tests**

## Common Pitfalls to Avoid

| Pitfall | Correct Approach |
|---|---|
| Forgetting `reference_date` | `build_feature_matrix` always requires it |
| Using full orders table for labels | Pass full orders to `compute_churn_target` so T_max is correct; during CV filter orders to `fold_t_max` |
| Adding `scale_pos_weight` | Deliberately left at 1.0, do not add |
| Hard temporal cut in `time_based_split` | Use interleaved stratified sampling by registration_date |
| Shuffled CV in tuning | Use walk-forward temporal folds only |
