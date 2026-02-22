"""Unit tests for guardrail checks."""

import pandas as pd
import pytest

from src.evaluation.guardrails import (
    check_brier_score,
    check_calibration,
    check_prediction_coverage,
    check_roc_auc,
    run_all_guardrails,
)


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


class TestRunAllGuardrails:
    def _make_metrics(self, roc=0.80, brier=0.15):
        return {
            "roc_auc": roc,
            "brier_score": brier,
            "pr_auc": 0.70,
        }

    def _make_calibration(self, ece=0.05):
        return {
            "fraction_of_positives": [0.1, 0.3, 0.5, 0.7, 0.9],
            "mean_predicted_value": [0.1, 0.3, 0.5, 0.7, 0.9],
            "ece": ece,
        }

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

    def test_returns_four_checks(self):
        metrics = self._make_metrics()
        calibration = self._make_calibration()
        predictions = pd.DataFrame({"customer_id": range(1, 101)})
        customers = pd.DataFrame({"customer_id": range(1, 101)})
        results = run_all_guardrails(metrics, calibration, predictions, customers)
        assert len(results) == 4
