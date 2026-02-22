"""Unit tests for guardrail checks."""

import pandas as pd
import pytest

from src.evaluation.guardrails import (
    check_baseline_pr_auc,
    check_baseline_precision,
    check_baseline_recall,
    check_baseline_roc_auc,
    check_brier_score,
    check_calibration,
    check_distribution_shift,
    check_min_pr_auc,
    check_ml_beats_baseline,
    check_ml_beats_baseline_pr_auc,
    check_no_overfitting,
    check_prediction_coverage,
    check_roc_auc,
    run_all_guardrails,
)

# ---------------------------------------------------------------------------
# ML model checks
# ---------------------------------------------------------------------------


class TestCheckRocAuc:
    def test_passes_at_threshold(self):
        result = check_roc_auc(0.65, threshold=0.65)
        assert result["passed"] is True

    def test_passes_above_threshold(self):
        result = check_roc_auc(0.80, threshold=0.65)
        assert result["passed"] is True

    def test_fails_below_threshold(self):
        result = check_roc_auc(0.60, threshold=0.65)
        assert result["passed"] is False

    def test_result_keys(self):
        result = check_roc_auc(0.70)
        assert {"check", "value", "threshold", "passed", "message"}.issubset(result)

    def test_message_contains_value(self):
        result = check_roc_auc(0.72, threshold=0.65)
        assert "0.7200" in result["message"]


class TestCheckBrierScore:
    def test_passes_low_score(self):
        result = check_brier_score(0.10, max_score=0.25)
        assert result["passed"] is True

    def test_fails_high_score(self):
        result = check_brier_score(0.30, max_score=0.25)
        assert result["passed"] is False

    def test_passes_at_threshold(self):
        result = check_brier_score(0.25, max_score=0.25)
        assert result["passed"] is True


class TestCheckCalibration:
    def test_passes_low_ece(self):
        result = check_calibration(0.05, max_ece=0.10)
        assert result["passed"] is True

    def test_fails_high_ece(self):
        result = check_calibration(0.15, max_ece=0.10)
        assert result["passed"] is False


class TestCheckPredictionCoverage:
    @pytest.fixture
    def customers(self):
        return pd.DataFrame({"customer_id": range(1, 101)})

    def test_passes_full_coverage(self, customers):
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        result = check_prediction_coverage(predictions, customers, min_coverage=0.50)
        assert result["passed"] is True

    def test_fails_low_coverage(self, customers):
        predictions = pd.DataFrame({"customer_id": range(1, 11)})  # only 10 out of 100
        result = check_prediction_coverage(predictions, customers, min_coverage=0.50)
        assert result["passed"] is False

    def test_value_is_fraction(self, customers):
        predictions = pd.DataFrame({"customer_id": range(1, 76)})
        result = check_prediction_coverage(predictions, customers, min_coverage=0.50)
        assert abs(result["value"] - 0.75) < 1e-6


class TestCheckMinPrAuc:
    def test_passes_above_threshold(self):
        result = check_min_pr_auc(0.35, threshold=0.20)
        assert result["passed"] is True

    def test_fails_below_threshold(self):
        result = check_min_pr_auc(0.10, threshold=0.20)
        assert result["passed"] is False

    def test_passes_at_threshold(self):
        result = check_min_pr_auc(0.20, threshold=0.20)
        assert result["passed"] is True

    def test_check_name(self):
        assert check_min_pr_auc(0.30)["check"] == "min_pr_auc"


class TestCheckNoOverfitting:
    def test_passes_small_gap(self):
        result = check_no_overfitting(0.85, 0.80, max_gap=0.15)
        assert result["passed"] is True

    def test_fails_large_gap(self):
        result = check_no_overfitting(0.98, 0.70, max_gap=0.15)
        assert result["passed"] is False

    def test_passes_below_threshold(self):
        result = check_no_overfitting(0.80, 0.70, max_gap=0.15)  # gap=0.10 < 0.15
        assert result["passed"] is True

    def test_gap_value_is_difference(self):
        result = check_no_overfitting(0.90, 0.75, max_gap=0.20)
        assert abs(result["value"] - 0.15) < 1e-9

    def test_check_name(self):
        assert check_no_overfitting(0.80, 0.75)["check"] == "max_train_test_roc_gap"

    def test_negative_gap_passes(self):
        # test ROC slightly above train ROC — valid, should pass
        result = check_no_overfitting(0.78, 0.80, max_gap=0.15)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# ML vs baseline checks
# ---------------------------------------------------------------------------


class TestCheckMlBeatsBaseline:
    def test_passes_when_ml_better(self):
        result = check_ml_beats_baseline(0.80, 0.75, tolerance=0.05)
        assert result["passed"] is True

    def test_passes_within_tolerance(self):
        result = check_ml_beats_baseline(0.73, 0.75, tolerance=0.05)
        assert result["passed"] is True  # lag is 0.02 < tolerance 0.05

    def test_fails_outside_tolerance(self):
        result = check_ml_beats_baseline(0.65, 0.75, tolerance=0.05)
        assert result["passed"] is False  # lag is 0.10 > tolerance 0.05

    def test_passes_within_tolerance_boundary(self):
        result = check_ml_beats_baseline(0.70, 0.75, tolerance=0.06)
        assert result["passed"] is True  # lag=0.05 < tolerance=0.06

    def test_check_name(self):
        assert check_ml_beats_baseline(0.80, 0.75)["check"] == "ml_vs_baseline_roc_auc"

    def test_value_is_signed_diff(self):
        result = check_ml_beats_baseline(0.72, 0.80, tolerance=0.10)
        assert abs(result["value"] - (-0.08)) < 1e-9


class TestCheckMlBeatsBaselinePrAuc:
    def test_passes_when_ml_better(self):
        result = check_ml_beats_baseline_pr_auc(0.60, 0.50, tolerance=0.05)
        assert result["passed"] is True

    def test_passes_within_tolerance(self):
        result = check_ml_beats_baseline_pr_auc(0.47, 0.50, tolerance=0.05)
        assert result["passed"] is True  # lag 0.03 < tolerance 0.05

    def test_fails_outside_tolerance(self):
        result = check_ml_beats_baseline_pr_auc(0.40, 0.50, tolerance=0.05)
        assert result["passed"] is False  # lag 0.10 > tolerance 0.05

    def test_passes_at_tolerance_boundary(self):
        result = check_ml_beats_baseline_pr_auc(0.45, 0.50, tolerance=0.05)
        assert result["passed"] is True  # lag == tolerance → boundary passes

    def test_check_name(self):
        result = check_ml_beats_baseline_pr_auc(0.60, 0.50)
        assert result["check"] == "ml_vs_baseline_pr_auc"

    def test_value_is_signed_diff(self):
        result = check_ml_beats_baseline_pr_auc(0.42, 0.50, tolerance=0.10)
        assert abs(result["value"] - (-0.08)) < 1e-9


# ---------------------------------------------------------------------------
# Baseline quality checks
# ---------------------------------------------------------------------------


class TestCheckBaselineRocAuc:
    def test_passes_good_baseline(self):
        assert check_baseline_roc_auc(0.75, threshold=0.65)["passed"] is True

    def test_fails_weak_baseline(self):
        assert check_baseline_roc_auc(0.55, threshold=0.65)["passed"] is False

    def test_check_name(self):
        assert check_baseline_roc_auc(0.70)["check"] == "baseline_min_roc_auc"


class TestCheckBaselinePrAuc:
    def test_passes_good_pr_auc(self):
        assert check_baseline_pr_auc(0.50, threshold=0.40)["passed"] is True

    def test_fails_low_pr_auc(self):
        assert check_baseline_pr_auc(0.30, threshold=0.40)["passed"] is False

    def test_check_name(self):
        assert check_baseline_pr_auc(0.50)["check"] == "baseline_min_pr_auc"


class TestCheckBaselinePrecision:
    def test_passes_good_precision(self):
        assert check_baseline_precision(0.70, threshold=0.55)["passed"] is True

    def test_fails_low_precision(self):
        assert check_baseline_precision(0.40, threshold=0.55)["passed"] is False

    def test_check_name(self):
        assert check_baseline_precision(0.60)["check"] == "baseline_min_precision"


class TestCheckBaselineRecall:
    def test_passes_good_recall(self):
        assert check_baseline_recall(0.70, threshold=0.55)["passed"] is True

    def test_fails_low_recall(self):
        assert check_baseline_recall(0.40, threshold=0.55)["passed"] is False

    def test_check_name(self):
        assert check_baseline_recall(0.60)["check"] == "baseline_min_recall"


# ---------------------------------------------------------------------------
# Distribution-shift check
# ---------------------------------------------------------------------------


class TestCheckDistributionShift:
    def test_passes_low_auc(self):
        result = check_distribution_shift(0.55, max_auc=0.70)
        assert result["passed"] is True

    def test_passes_at_threshold(self):
        result = check_distribution_shift(0.70, max_auc=0.70)
        assert result["passed"] is True

    def test_fails_above_threshold(self):
        result = check_distribution_shift(0.80, max_auc=0.70)
        assert result["passed"] is False

    def test_check_name(self):
        assert check_distribution_shift(0.60)["check"] == "max_adversarial_auc"

    def test_value_stored(self):
        result = check_distribution_shift(0.65, max_auc=0.70)
        assert abs(result["value"] - 0.65) < 1e-9

    def test_operator_is_leq(self):
        assert "<=" in check_distribution_shift(0.60)["message"]


# ---------------------------------------------------------------------------
# run_all_guardrails orchestrator
# ---------------------------------------------------------------------------


class TestRunAllGuardrails:
    def _make_metrics(self, roc=0.80, brier=0.15, pr_auc=0.50):
        return {
            "roc_auc": roc,
            "brier_score": brier,
            "pr_auc": pr_auc,
        }

    def _make_calibration(self, ece=0.05):
        return {
            "fraction_of_positives": [0.1, 0.3, 0.5, 0.7, 0.9],
            "mean_predicted_value": [0.1, 0.3, 0.5, 0.7, 0.9],
            "ece": ece,
        }

    def _make_baseline(self, roc=0.75, pr_auc=0.55, precision=0.70, recall=0.65):
        return {"roc_auc": roc, "pr_auc": pr_auc, "precision": precision, "recall": recall}

    def _make_shift(self, adversarial_auc=0.60):
        return {"adversarial_auc": adversarial_auc, "feature_ks": {"recency": 0.10}}

    # ---- base (4 checks) ----

    def test_base_returns_four_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        results = run_all_guardrails(metrics, calibration, predictions, customers)
        assert len(results) == 4

    def test_all_pass(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        results = run_all_guardrails(metrics, calibration, predictions, customers)
        assert all(r["passed"] for r in results)

    def test_roc_fail_surfaces(self):
        metrics = self._make_metrics(roc=0.55)  # below MIN_ROC_AUC
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        results = run_all_guardrails(metrics, calibration, predictions, customers)
        roc_check = next(r for r in results if r["check"] == "min_roc_auc")
        assert not roc_check["passed"]

    # ---- with train_metrics (6 checks) ----

    def test_with_train_metrics_returns_six_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        train_metrics = self._make_metrics(roc=0.82)
        results = run_all_guardrails(
            metrics, calibration, predictions, customers, train_metrics=train_metrics
        )
        assert len(results) == 6

    def test_overfitting_check_included_with_train_metrics(self):
        metrics = self._make_metrics(roc=0.70)
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        train_metrics = self._make_metrics(roc=0.99)  # huge gap → should fail
        results = run_all_guardrails(
            metrics, calibration, predictions, customers, train_metrics=train_metrics
        )
        overfitting_check = next(r for r in results if r["check"] == "max_train_test_roc_gap")
        assert not overfitting_check["passed"]

    # ---- with baseline_metrics (10 checks) ----

    def test_with_baseline_metrics_returns_ten_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        baseline = self._make_baseline()
        results = run_all_guardrails(
            metrics, calibration, predictions, customers, baseline_metrics=baseline
        )
        assert len(results) == 10

    def test_ml_vs_baseline_check_included(self):
        metrics = self._make_metrics(roc=0.60)
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        baseline = self._make_baseline(roc=0.80)  # ML lags by 0.20 > tolerance
        results = run_all_guardrails(
            metrics, calibration, predictions, customers, baseline_metrics=baseline
        )
        ml_check = next(r for r in results if r["check"] == "ml_vs_baseline_roc_auc")
        assert not ml_check["passed"]

    # ---- with both optional args (12 checks) ----

    def test_with_all_args_returns_twelve_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        train_metrics = self._make_metrics(roc=0.82)
        baseline = self._make_baseline()
        results = run_all_guardrails(
            metrics,
            calibration,
            predictions,
            customers,
            train_metrics=train_metrics,
            baseline_metrics=baseline,
        )
        assert len(results) == 12

    def test_all_pass_with_all_args(self):
        metrics = self._make_metrics(roc=0.80, pr_auc=0.50)
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        train_metrics = self._make_metrics(roc=0.82)
        # pr_auc=0.45: ML (0.50) clearly exceeds baseline (0.45), avoids fp boundary
        baseline = self._make_baseline(roc=0.75, pr_auc=0.45)
        results = run_all_guardrails(
            metrics,
            calibration,
            predictions,
            customers,
            train_metrics=train_metrics,
            baseline_metrics=baseline,
        )
        assert all(r["passed"] for r in results), [r["message"] for r in results if not r["passed"]]

    # ---- with shift_metrics only (5 checks) ----

    def test_with_shift_metrics_returns_five_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        results = run_all_guardrails(
            metrics,
            calibration,
            predictions,
            customers,
            shift_metrics=self._make_shift(),
        )
        assert len(results) == 5

    def test_shift_check_included(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        shift = self._make_shift(adversarial_auc=0.85)  # above threshold → fail
        results = run_all_guardrails(
            metrics,
            calibration,
            predictions,
            customers,
            shift_metrics=shift,
        )
        shift_check = next(r for r in results if r["check"] == "max_adversarial_auc")
        assert not shift_check["passed"]

    # ---- with all three optional args (13 checks) ----

    def test_with_all_three_args_returns_thirteen_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        results = run_all_guardrails(
            metrics,
            calibration,
            predictions,
            customers,
            train_metrics=self._make_metrics(roc=0.82),
            baseline_metrics=self._make_baseline(),
            shift_metrics=self._make_shift(),
        )
        assert len(results) == 13
