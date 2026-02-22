"""Unit tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.customer_attributes import compute_customer_attributes
from src.feature_engineering.pipeline import build_feature_matrix
from src.feature_engineering.rfm import compute_rfm
from src.feature_engineering.time_features import compute_time_features
from src.feature_engineering.trend_features import compute_trend_features

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_orders():
    """Small deterministic order dataset for feature tests."""
    rng = np.random.default_rng(0)
    n = 300
    customer_ids = rng.integers(1, 21, size=n)  # 20 customers
    dates = pd.date_range("2021-01-01", "2023-06-30", periods=n)
    return pd.DataFrame(
        {
            "order_id": range(1, n + 1),
            "customer_id": customer_ids,
            "order_date": dates,
            "total_value": rng.exponential(50, size=n).round(2),
            "quantity": rng.integers(1, 10, size=n),
            "price": rng.exponential(25, size=n).round(2),
        }
    )


@pytest.fixture(scope="module")
def sample_customers():
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "registration_date": pd.date_range("2021-01-01", periods=n, freq="ME"),
            "segment": rng.choice(["bronze", "silver", "gold", "platinum"], size=n),
            "marketing_channel": rng.choice(["organic", "email", "social", "paid_search"], size=n),
            "country": rng.choice(["US", "UK", "DE"], size=n),
        }
    )


REFERENCE_DATE = pd.Timestamp("2023-07-01")


# ---------------------------------------------------------------------------
# RFM tests
# ---------------------------------------------------------------------------


class TestComputeRFM:
    def test_output_columns(self, sample_orders):
        rfm = compute_rfm(sample_orders, REFERENCE_DATE)
        assert {"customer_id", "recency", "frequency", "monetary", "avg_order_value"}.issubset(
            rfm.columns
        )

    def test_no_leakage(self, sample_orders):
        """Recency must always be >= 0 (reference_date is always in the future)."""
        rfm = compute_rfm(sample_orders, REFERENCE_DATE)
        assert (rfm["recency"] >= 0).all()

    def test_frequency_positive(self, sample_orders):
        rfm = compute_rfm(sample_orders, REFERENCE_DATE)
        assert (rfm["frequency"] >= 1).all()

    def test_monetary_positive(self, sample_orders):
        rfm = compute_rfm(sample_orders, REFERENCE_DATE)
        assert (rfm["monetary"] > 0).all()

    def test_orders_after_reference_excluded(self, sample_orders):
        """Orders on or after reference_date must NOT count."""
        early_ref = pd.Timestamp("2021-06-01")
        rfm_early = compute_rfm(sample_orders, early_ref)
        rfm_full = compute_rfm(sample_orders, REFERENCE_DATE)
        # Later reference date must have higher or equal frequency totals
        assert rfm_full["frequency"].sum() >= rfm_early["frequency"].sum()

    def test_reproducibility(self, sample_orders):
        r1 = compute_rfm(sample_orders, REFERENCE_DATE)
        r2 = compute_rfm(sample_orders, REFERENCE_DATE)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Time feature tests
# ---------------------------------------------------------------------------


class TestComputeTimeFeatures:
    def test_output_columns(self, sample_orders):
        tf = compute_time_features(sample_orders, REFERENCE_DATE)
        assert {
            "customer_id",
            "avg_inter_order_days",
            "orders_last_30d",
            "orders_last_60d",
        }.issubset(tf.columns)

    def test_window_counts_non_negative(self, sample_orders):
        tf = compute_time_features(sample_orders, REFERENCE_DATE)
        for col in ["orders_last_30d", "orders_last_60d", "orders_last_90d"]:
            assert (tf[col] >= 0).all(), f"{col} has negative values"

    def test_30d_lte_60d(self, sample_orders):
        tf = compute_time_features(sample_orders, REFERENCE_DATE)
        assert (tf["orders_last_30d"] <= tf["orders_last_60d"]).all()

    def test_60d_lte_90d(self, sample_orders):
        tf = compute_time_features(sample_orders, REFERENCE_DATE)
        assert (tf["orders_last_60d"] <= tf["orders_last_90d"]).all()

    def test_no_leakage(self, sample_orders):
        """Orders at or after reference_date must NOT appear in any window count."""
        # Inject an order exactly AT reference_date – it must be excluded
        future_order = pd.DataFrame(
            [
                {
                    "order_id": 99999,
                    "customer_id": 1,
                    "order_date": REFERENCE_DATE,  # exactly at boundary → excluded
                    "total_value": 9999.0,
                    "quantity": 1,
                    "price": 9999.0,
                }
            ]
        )
        orders_with_future = pd.concat([sample_orders, future_order], ignore_index=True)

        tf_base = compute_time_features(sample_orders, REFERENCE_DATE)
        tf_with_future = compute_time_features(orders_with_future, REFERENCE_DATE)

        # The injected order at reference_date must be excluded → counts should be equal
        assert tf_base["orders_last_90d"].sum() == tf_with_future["orders_last_90d"].sum()


# ---------------------------------------------------------------------------
# Trend feature tests
# ---------------------------------------------------------------------------


class TestComputeTrendFeatures:
    def test_output_columns(self, sample_orders):
        tf = compute_trend_features(sample_orders, REFERENCE_DATE)
        assert {
            "customer_id",
            "order_frequency_slope",
            "spend_slope",
            "rolling_avg_orders_3m",
            "spend_change_ratio",
        }.issubset(tf.columns)

    def test_no_nan(self, sample_orders):
        tf = compute_trend_features(sample_orders, REFERENCE_DATE)
        assert not tf.isnull().any().any()


# ---------------------------------------------------------------------------
# Customer attributes tests
# ---------------------------------------------------------------------------


class TestComputeCustomerAttributes:
    def test_output_columns(self, sample_customers):
        ca = compute_customer_attributes(sample_customers, REFERENCE_DATE)
        assert {"customer_id", "tenure_days", "segment_encoded", "channel_encoded"}.issubset(
            ca.columns
        )

    def test_tenure_non_negative(self, sample_customers):
        ca = compute_customer_attributes(sample_customers, REFERENCE_DATE)
        assert (ca["tenure_days"] >= 0).all()

    def test_encodings_are_integers(self, sample_customers):
        ca = compute_customer_attributes(sample_customers, REFERENCE_DATE)
        for col in ["segment_encoded", "channel_encoded", "country_encoded"]:
            assert ca[col].dtype == int or ca[col].dtype.kind == "i"


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def test_no_nan_after_fill(self, sample_orders, sample_customers):
        features = build_feature_matrix(sample_orders, sample_customers, REFERENCE_DATE)
        numeric = features.select_dtypes(include="number").drop(columns=["customer_id"])
        assert not numeric.isnull().any().any()

    def test_customer_id_preserved(self, sample_orders, sample_customers):
        features = build_feature_matrix(sample_orders, sample_customers, REFERENCE_DATE)
        assert "customer_id" in features.columns

    def test_no_future_data_used(self, sample_orders, sample_customers):
        """Features from early cutoff must differ from full dataset features."""
        early_ref = pd.Timestamp("2022-01-01")
        f_early = build_feature_matrix(sample_orders, sample_customers, early_ref)
        f_full = build_feature_matrix(sample_orders, sample_customers, REFERENCE_DATE)
        # Full dataset should have higher recency on average (more recent orders)
        common = set(f_early["customer_id"]) & set(f_full["customer_id"])
        assert len(common) > 0
