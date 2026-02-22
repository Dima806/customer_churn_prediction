"""Unit tests for churn target construction."""

import pandas as pd
import pytest

from src.target.churn_target import compute_churn_target


@pytest.fixture
def order_data():
    """Minimal order dataset with known churn structure."""
    records = []

    # Customer 1: ordered frequently, including near the end → NOT churned
    for day_offset in range(0, 900, 10):
        records.append(
            {
                "order_id": len(records) + 1,
                "customer_id": 1,
                "order_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=day_offset),
                "total_value": 100.0,
            }
        )

    # Customer 2: ordered only in the first 12 months → churned
    for day_offset in range(0, 365, 20):
        records.append(
            {
                "order_id": len(records) + 1,
                "customer_id": 2,
                "order_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=day_offset),
                "total_value": 50.0,
            }
        )

    # Customer 3: ordered up to 180 days before end → churned (with 90-day window)
    t_max_approx = pd.Timestamp("2021-01-01") + pd.Timedelta(days=899)
    records.append(
        {
            "order_id": len(records) + 1,
            "customer_id": 3,
            "order_date": t_max_approx - pd.Timedelta(days=180),
            "total_value": 75.0,
        }
    )

    return pd.DataFrame(records)


@pytest.fixture
def customer_data():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "registration_date": pd.to_datetime(["2021-01-01"] * 3),
        }
    )


class TestComputeChurnTarget:
    def test_binary_target(self, order_data, customer_data):
        target, _, _ = compute_churn_target(order_data, customer_data, churn_period_days=90)
        assert target["is_churned"].isin([0, 1]).all()

    def test_no_nan_in_target(self, order_data, customer_data):
        target, _, _ = compute_churn_target(order_data, customer_data, churn_period_days=90)
        assert not target["is_churned"].isna().any()

    def test_feature_window_excludes_churn_window(self, order_data, customer_data):
        target, feature_end_date, t_max = compute_churn_target(order_data, customer_data, 90)
        assert feature_end_date < t_max
        assert (t_max - feature_end_date).days == 90

    def test_customer1_not_churned(self, order_data, customer_data):
        """Customer 1 ordered recently – must be labelled 0."""
        target, _, _ = compute_churn_target(order_data, customer_data, 90)
        row = target[target["customer_id"] == 1]
        assert not row.empty
        assert row["is_churned"].iloc[0] == 0

    def test_customer2_churned(self, order_data, customer_data):
        """Customer 2 stopped 12 months ago – must be labelled 1."""
        target, _, _ = compute_churn_target(order_data, customer_data, 90)
        row = target[target["customer_id"] == 2]
        assert not row.empty
        assert row["is_churned"].iloc[0] == 1

    def test_customer3_churned(self, order_data, customer_data):
        """Customer 3 last ordered 180 days ago – must be labelled 1 with 90-day window."""
        target, _, _ = compute_churn_target(order_data, customer_data, 90)
        row = target[target["customer_id"] == 3]
        assert not row.empty
        assert row["is_churned"].iloc[0] == 1

    def test_t_max_is_last_order_date(self, order_data, customer_data):
        _, _, t_max = compute_churn_target(order_data, customer_data, 90)
        assert t_max == order_data["order_date"].max()

    def test_only_pre_window_customers_in_target(self, order_data, customer_data):
        """Customers with orders only inside the churn window are excluded."""
        # Add a customer who only ordered inside the churn window
        new_order = pd.DataFrame(
            [
                {
                    "order_id": 9999,
                    "customer_id": 99,
                    "order_date": order_data["order_date"].max() - pd.Timedelta(days=5),
                    "total_value": 10.0,
                }
            ]
        )
        new_customer = pd.DataFrame(
            [{"customer_id": 99, "registration_date": pd.Timestamp("2023-01-01")}]
        )
        combined_orders = pd.concat([order_data, new_order], ignore_index=True)
        combined_customers = pd.concat([customer_data, new_customer], ignore_index=True)

        target, _, _ = compute_churn_target(combined_orders, combined_customers, 90)
        # Customer 99 has no pre-window orders, so should NOT appear in target
        assert 99 not in target["customer_id"].values
