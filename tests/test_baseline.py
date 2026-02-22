"""Unit tests for the rule-based baseline model."""

import pandas as pd
import pytest

from src.baseline.rule_based import evaluate_baseline, predict_churn_rule_based


@pytest.fixture
def sample_features():
    """Feature rows with known recency / order_count patterns."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "recency": [5, 100, 45, 200, 30],
            "orders_last_60d": [3, 0, 2, 0, 5],
            "frequency": [20, 2, 8, 1, 15],
            "monetary": [500.0, 30.0, 200.0, 10.0, 700.0],
        }
    )


@pytest.fixture
def sample_target():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "is_churned": [0, 1, 0, 1, 0],
        }
    )


class TestPredictChurnRuleBased:
    def test_output_shape(self, sample_features):
        y_pred, y_score = predict_churn_rule_based(sample_features, churn_period_days=90)
        assert len(y_pred) == len(sample_features)
        assert len(y_score) == len(sample_features)

    def test_high_recency_predicts_churn(self, sample_features):
        """Recency > CHURN_PERIOD must be predicted churned."""
        y_pred, _ = predict_churn_rule_based(sample_features, churn_period_days=90)
        # Customer 2 (recency=100) and 4 (recency=200) should be churned
        preds = y_pred.to_numpy()
        assert preds[1] == 1  # customer 2
        assert preds[3] == 1  # customer 4

    def test_zero_60d_orders_predicts_churn(self, sample_features):
        y_pred, _ = predict_churn_rule_based(sample_features, churn_period_days=90)
        # customers 2 and 4 have orders_last_60d == 0
        assert y_pred.iloc[1] == 1
        assert y_pred.iloc[3] == 1

    def test_recent_active_predicts_not_churned(self, sample_features):
        y_pred, _ = predict_churn_rule_based(sample_features, churn_period_days=90)
        # Customer 1: recency=5, orders_last_60d=3 → active
        assert y_pred.iloc[0] == 0

    def test_binary_output(self, sample_features):
        y_pred, _ = predict_churn_rule_based(sample_features, churn_period_days=90)
        assert set(y_pred.unique()).issubset({0, 1})

    def test_score_in_unit_interval(self, sample_features):
        _, y_score = predict_churn_rule_based(sample_features, churn_period_days=90)
        assert (y_score >= 0).all() and (y_score <= 1).all()

    def test_higher_recency_higher_score(self, sample_features):
        """Score should be monotone with recency (all else equal)."""
        _, y_score = predict_churn_rule_based(sample_features, churn_period_days=90)
        scores = y_score.to_numpy()
        # Customer 4 (recency=200) should score higher than customer 1 (recency=5)
        assert scores[3] > scores[0]


class TestEvaluateBaseline:
    def test_metric_keys_present(self, sample_features, sample_target):
        metrics = evaluate_baseline(sample_features, sample_target, churn_period_days=90)
        assert {"precision", "recall", "roc_auc", "pr_auc"}.issubset(metrics)

    def test_metrics_in_valid_range(self, sample_features, sample_target):
        metrics = evaluate_baseline(sample_features, sample_target, churn_period_days=90)
        for key in ["precision", "recall", "roc_auc", "pr_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"
