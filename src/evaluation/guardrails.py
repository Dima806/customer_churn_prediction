"""Production guardrails: automated quality checks on model outputs.

Each check returns a standardised dict:
  { check, value, threshold, passed, message }
"""

import pandas as pd

from src.config import MAX_BRIER_SCORE, MAX_ECE, MIN_COVERAGE, MIN_ROC_AUC
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_result(
    check: str,
    value: float,
    threshold: float,
    passed: bool,
    op: str = ">=",
) -> dict:
    op_str = op if not passed else op
    msg = f"{check}: {value:.4f} {op_str} {threshold} → {'PASS' if passed else 'FAIL'}"
    return {
        "check": check,
        "value": value,
        "threshold": threshold,
        "passed": passed,
        "message": msg,
    }


def check_roc_auc(roc_auc: float, threshold: float = MIN_ROC_AUC) -> dict:
    """Minimum acceptable ROC-AUC."""
    return _make_result("min_roc_auc", roc_auc, threshold, roc_auc >= threshold, ">=")


def check_brier_score(brier_score: float, max_score: float = MAX_BRIER_SCORE) -> dict:
    """Maximum acceptable Brier score (lower is better)."""
    return _make_result("max_brier_score", brier_score, max_score, brier_score <= max_score, "<=")


def check_calibration(ece: float, max_ece: float = MAX_ECE) -> dict:
    """Maximum acceptable Expected Calibration Error."""
    return _make_result("max_ece", ece, max_ece, ece <= max_ece, "<=")


def check_prediction_coverage(
    predictions: pd.DataFrame,
    customers: pd.DataFrame,
    min_coverage: float = MIN_COVERAGE,
) -> dict:
    """Fraction of customers with a prediction must meet the minimum."""
    predicted = len(predictions["customer_id"].unique())
    total = len(customers["customer_id"].unique())
    coverage = predicted / max(total, 1)
    return _make_result("min_coverage", coverage, min_coverage, coverage >= min_coverage, ">=")


def run_all_guardrails(
    metrics: dict,
    calibration: dict,
    predictions: pd.DataFrame,
    customers: pd.DataFrame,
) -> list[dict]:
    """Execute all guardrail checks and return a consolidated list.

    Args:
        metrics: Output of :func:`~src.evaluation.metrics.compute_metrics`.
        calibration: Output of :func:`~src.evaluation.calibration.compute_calibration`.
        predictions: DataFrame with ``customer_id`` column.
        customers: Full customer dimension table.

    Returns:
        List of guardrail result dicts.
    """
    results = [
        check_roc_auc(metrics["roc_auc"]),
        check_brier_score(metrics["brier_score"]),
        check_calibration(calibration["ece"]),
        check_prediction_coverage(predictions, customers),
    ]

    n_failed = sum(1 for r in results if not r["passed"])
    if n_failed:
        failed_names = [r["check"] for r in results if not r["passed"]]
        logger.warning(f"Guardrail failures ({n_failed}): {failed_names}")
    else:
        logger.info("All guardrails passed")

    return results
