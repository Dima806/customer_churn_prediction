# ===========================================================================
# Customer Churn Prediction – Makefile
# All commands use `uv run` to stay inside the project's virtual environment.
# ===========================================================================

.PHONY: install data lint format audit security complexity maintainability \
        test tune train predict dashboard docker-build all help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:          ## Install all dependencies (incl. dev group)
	uv sync --all-groups

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

data:             ## Generate synthetic star-schema CSVs into data/
	uv run python -m src.data_generation.main

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:             ## Run ruff linter
	uv run ruff check src/ tests/ dashboard/

format:           ## Auto-format code with ruff
	uv run ruff format src/ tests/ dashboard/

format-check:     ## Check formatting without writing changes
	uv run ruff format --check src/ tests/ dashboard/

audit:            ## Dependency vulnerability audit
	uv run pip-audit

security:         ## Static security analysis with bandit
	uv run bandit -r src/ --severity-level medium -q

complexity:       ## Cyclomatic complexity report (only B-grade and below)
	uv run radon cc src/ -a -s -nb

maintainability:  ## Maintainability index report
	uv run radon mi src/ -s

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:             ## Run full test suite with coverage
	uv run pytest --cov=src --cov-report=term-missing -q

test-verbose:     ## Run tests with verbose output
	uv run pytest --cov=src --cov-report=term-missing -v

# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------

tune:             ## Hyperparameter search (writes best_params.json)
	uv run python -m src.models.tune

train:            ## Train XGBoost model and save all evaluation artefacts
	uv run python -m src.models.train

predict:          ## Batch-predict churn for all customers
	uv run python -m src.prediction.batch_predict

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

dashboard:        ## Launch the Plotly Dash dashboard on port 7860
	uv run python dashboard/app.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:     ## Build the Docker image for HuggingFace Spaces
	docker build -t customer-churn-dashboard .

docker-run:       ## Run the Docker container locally (port 7860)
	docker run --rm -p 7860:7860 customer-churn-dashboard

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

all: data train predict   ## Run data generation → training → prediction

# ---------------------------------------------------------------------------
# CI helper
# ---------------------------------------------------------------------------

ci: lint format-check security test   ## Run all CI checks

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
