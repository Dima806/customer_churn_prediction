# Commands Reference

## Make Targets

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
make format-check      # ruff format --check (read-only, CI)
make security          # bandit -r src/ --severity-level medium
make ci                # lint + format-check + security + test

# Dashboard
make dashboard         # Plotly Dash on http://localhost:7860
```

## Running Individual Modules

```bash
uv run python -m src.models.train
uv run python -m src.models.tune
uv run python -m src.data_generation.main
uv run python -m src.prediction.batch_predict
```

## Running Individual Tests

```bash
uv run pytest tests/test_guardrails.py -v
uv run pytest tests/test_guardrails.py::TestRunAllGuardrails::test_all_checks_present -v
```

## Module Entry Points

| Module | Path |
|---|---|
| Config constants | [src/config.py](../src/config.py) |
| Data generation | [src/data_generation/main.py](../src/data_generation/main.py) |
| Churn target | [src/target/churn_target.py](../src/target/churn_target.py) |
| Feature pipeline | [src/feature_engineering/pipeline.py](../src/feature_engineering/pipeline.py) |
| Training | [src/models/train.py](../src/models/train.py) |
| Tuning | [src/models/tune.py](../src/models/tune.py) |
| Prediction | [src/prediction/batch_predict.py](../src/prediction/batch_predict.py) |
| Guardrails | [src/evaluation/guardrails.py](../src/evaluation/guardrails.py) |
| Baseline | [src/baseline/rule_based.py](../src/baseline/rule_based.py) |
| Dashboard | [app.py](../app.py) |
| Logger utility | [src/utils/logger.py](../src/utils/logger.py) |
