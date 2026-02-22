"""Production guardrails: automated quality checks on model outputs.

Each check returns a standardised dict:
  { check, value, threshold, passed, message }
"""

import pandas as pd

from src.config import (
    BASELINE_MIN_PR_AUC,
    BASELINE_MIN_PRECISION,
    BASELINE_MIN_RECALL,
    BASELINE_MIN_ROC_AUC,
    MAX_ADVERSARIAL_AUC,
    MAX_BRIER_SCORE,
    MAX_ECE,
    MAX_TRAIN_TEST_ROC_GAP,
    MIN_COVERAGE,
    MIN_PR_AUC,
    MIN_ROC_AUC,
    ML_BASELINE_PR_AUC_TOLERANCE,
    ML_BASELINE_TOLERANCE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_result(
    check: str,
    value: float,
    threshold: float,
    passed: bool,
    op: str = ">=",
) -> dict:
    msg = f"{check}: {value:.4f} {op} {threshold} → {'PASS' if passed else 'FAIL'}"
    return {
        "check": check,
        "value": value,
        "threshold": threshold,
        "passed": passed,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# ML model checks
# ---------------------------------------------------------------------------


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


def check_min_pr_auc(pr_auc: float, threshold: float = MIN_PR_AUC) -> dict:
    """Minimum acceptable PR-AUC (informative under class imbalance)."""
    return _make_result("min_pr_auc", pr_auc, threshold, pr_auc >= threshold, ">=")


def check_no_overfitting(
    train_roc_auc: float,
    test_roc_auc: float,
    max_gap: float = MAX_TRAIN_TEST_ROC_GAP,
) -> dict:
    """Train-test ROC-AUC gap must not exceed the maximum allowed gap."""
    gap = train_roc_auc - test_roc_auc
    passed = gap <= max_gap
    msg = (
        f"train_test_roc_gap: train={train_roc_auc:.4f} test={test_roc_auc:.4f} "
        f"gap={gap:.4f} <= {max_gap} → {'PASS' if passed else 'FAIL'}"
    )
    return {
        "check": "max_train_test_roc_gap",
        "value": gap,
        "threshold": max_gap,
        "passed": passed,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# ML vs baseline checks
# ---------------------------------------------------------------------------


def check_ml_beats_baseline(
    ml_roc_auc: float,
    baseline_roc_auc: float,
    tolerance: float = ML_BASELINE_TOLERANCE,
) -> dict:
    """ML ROC-AUC must not lag the baseline by more than the tolerance."""
    diff = ml_roc_auc - baseline_roc_auc
    passed = diff >= -tolerance
    msg = (
        f"ml_vs_baseline_roc_auc: ml={ml_roc_auc:.4f} baseline={baseline_roc_auc:.4f} "
        f"diff={diff:+.4f} >= -{tolerance} → {'PASS' if passed else 'FAIL'}"
    )
    return {
        "check": "ml_vs_baseline_roc_auc",
        "value": diff,
        "threshold": -tolerance,
        "passed": passed,
        "message": msg,
    }


def check_ml_beats_baseline_pr_auc(
    ml_pr_auc: float,
    baseline_pr_auc: float,
    tolerance: float = ML_BASELINE_PR_AUC_TOLERANCE,
) -> dict:
    """ML PR-AUC must not lag the baseline by more than the tolerance."""
    diff = ml_pr_auc - baseline_pr_auc
    passed = diff >= -tolerance
    msg = (
        f"ml_vs_baseline_pr_auc: ml={ml_pr_auc:.4f}"
        f" baseline={baseline_pr_auc:.4f}"
        f" diff={diff:+.4f} >= -{tolerance}"
        f" → {'PASS' if passed else 'FAIL'}"
    )
    return {
        "check": "ml_vs_baseline_pr_auc",
        "value": diff,
        "threshold": -tolerance,
        "passed": passed,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Baseline quality checks
# ---------------------------------------------------------------------------


def check_baseline_roc_auc(roc_auc: float, threshold: float = BASELINE_MIN_ROC_AUC) -> dict:
    """Baseline minimum ROC-AUC sanity check."""
    return _make_result("baseline_min_roc_auc", roc_auc, threshold, roc_auc >= threshold, ">=")


def check_baseline_pr_auc(pr_auc: float, threshold: float = BASELINE_MIN_PR_AUC) -> dict:
    """Baseline minimum PR-AUC sanity check."""
    return _make_result("baseline_min_pr_auc", pr_auc, threshold, pr_auc >= threshold, ">=")


def check_baseline_precision(precision: float, threshold: float = BASELINE_MIN_PRECISION) -> dict:
    """Baseline minimum precision sanity check."""
    return _make_result(
        "baseline_min_precision", precision, threshold, precision >= threshold, ">="
    )


def check_baseline_recall(recall: float, threshold: float = BASELINE_MIN_RECALL) -> dict:
    """Baseline minimum recall sanity check."""
    return _make_result("baseline_min_recall", recall, threshold, recall >= threshold, ">=")


# ---------------------------------------------------------------------------
# Distribution-shift check
# ---------------------------------------------------------------------------


def check_distribution_shift(
    adversarial_auc: float,
    max_auc: float = MAX_ADVERSARIAL_AUC,
) -> dict:
    """Adversarial train/test AUC must not exceed the threshold.

    A random-forest trained to separate train from test samples should
    perform near chance (AUC ≈ 0.5).  High AUC indicates covariate shift
    that can make hold-out evaluation unreliable.
    """
    return _make_result(
        "max_adversarial_auc", adversarial_auc, max_auc, adversarial_auc <= max_auc, "<="
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_guardrails(
    metrics: dict,
    calibration: dict,
    predictions: pd.DataFrame,
    customers: pd.DataFrame,
    train_metrics: dict | None = None,
    baseline_metrics: dict | None = None,
    shift_metrics: dict | None = None,
) -> list[dict]:
    """Execute all guardrail checks and return a consolidated list.

    Args:
        metrics: Output of :func:`~src.evaluation.metrics.compute_metrics` on
                 the **test** set.
        calibration: Output of :func:`~src.evaluation.calibration.compute_calibration`.
        predictions: DataFrame with ``customer_id`` column.
        customers: Full customer dimension table.
        train_metrics: Optional output of ``compute_metrics`` on the **train**
                       set; enables PR-AUC and overfitting checks.
        baseline_metrics: Optional output of
                          :func:`~src.baseline.rule_based.evaluate_baseline`
                          on the same test set; enables ML-vs-baseline and
                          baseline quality checks.
        shift_metrics: Optional output of
                       :func:`~src.evaluation.distribution_shift.compute_adversarial_auc`;
                       enables the distribution-shift check.

    Returns:
        List of guardrail result dicts.
    """
    results: list[dict] = [
        check_roc_auc(metrics["roc_auc"]),
        check_brier_score(metrics["brier_score"]),
        check_calibration(calibration["ece"]),
        check_prediction_coverage(predictions, customers),
    ]

    if train_metrics is not None:
        results.append(check_min_pr_auc(metrics["pr_auc"]))
        results.append(check_no_overfitting(train_metrics["roc_auc"], metrics["roc_auc"]))

    if baseline_metrics is not None:
        results.append(check_ml_beats_baseline(metrics["roc_auc"], baseline_metrics["roc_auc"]))
        results.append(
            check_ml_beats_baseline_pr_auc(metrics["pr_auc"], baseline_metrics["pr_auc"])
        )
        results.append(check_baseline_roc_auc(baseline_metrics["roc_auc"]))
        results.append(check_baseline_pr_auc(baseline_metrics["pr_auc"]))
        results.append(check_baseline_precision(baseline_metrics["precision"]))
        results.append(check_baseline_recall(baseline_metrics["recall"]))

    if shift_metrics is not None:
        results.append(check_distribution_shift(shift_metrics["adversarial_auc"]))

    n_failed = sum(1 for r in results if not r["passed"])
    if n_failed:
        failed_names = [r["check"] for r in results if not r["passed"]]
        logger.warning(f"Guardrail failures ({n_failed}): {failed_names}")
    else:
        logger.info("All guardrails passed")

    return results
