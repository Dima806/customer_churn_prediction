"""Unit tests for synthetic data generation."""

import pandas as pd
import pytest

from src.data_generation.generate_customers import generate_customers
from src.data_generation.generate_orders import generate_orders
from src.data_generation.generate_products import generate_products

# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


class TestGenerateCustomers:
    def test_row_count(self):
        df = generate_customers(n=100, seed=0)
        assert len(df) == 100

    def test_required_columns(self):
        df = generate_customers(n=50, seed=0)
        required = {"customer_id", "registration_date", "country", "segment", "marketing_channel"}
        assert required.issubset(df.columns)

    def test_unique_customer_ids(self):
        df = generate_customers(n=100, seed=0)
        assert df["customer_id"].nunique() == 100

    def test_registration_dates_within_range(self):
        df = generate_customers(n=200, seed=0, start_date="2021-01-01", end_date="2023-12-31")
        assert df["registration_date"].min() >= pd.Timestamp("2021-01-01")
        assert df["registration_date"].max() <= pd.Timestamp("2023-12-31")

    def test_segment_values(self):
        df = generate_customers(n=100, seed=0)
        valid = {"bronze", "silver", "gold", "platinum"}
        assert set(df["segment"].unique()).issubset(valid)

    def test_marketing_channel_values(self):
        df = generate_customers(n=100, seed=0)
        valid = {"organic", "email", "social", "paid_search"}
        assert set(df["marketing_channel"].unique()).issubset(valid)

    def test_birth_date_nullable(self):
        df = generate_customers(n=200, seed=0)
        # Should have some nulls (~15 %)
        null_rate = df["birth_date"].isna().mean()
        assert 0.02 < null_rate < 0.50

    def test_last_profile_update_nullable(self):
        df = generate_customers(n=200, seed=0)
        null_rate = df["last_profile_update"].isna().mean()
        assert 0.05 < null_rate < 0.70

    def test_reproducibility(self):
        df1 = generate_customers(n=50, seed=42)
        df2 = generate_customers(n=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_customers(n=50, seed=1)
        df2 = generate_customers(n=50, seed=2)
        assert not df1["registration_date"].equals(df2["registration_date"])


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


class TestGenerateProducts:
    def test_row_count(self):
        df = generate_products(n=50, seed=0)
        assert len(df) == 50

    def test_required_columns(self):
        df = generate_products(n=50, seed=0)
        assert {"product_id", "product_type", "base_price", "margin_group"}.issubset(df.columns)

    def test_prices_positive(self):
        df = generate_products(n=100, seed=0)
        assert (df["base_price"] > 0).all()

    def test_price_power_law(self):
        """Most prices should be low, a few should be high (right-skewed)."""
        df = generate_products(n=1000, seed=0)
        assert df["base_price"].median() < df["base_price"].mean()

    def test_margin_groups_valid(self):
        df = generate_products(n=100, seed=0)
        assert set(df["margin_group"].unique()).issubset({"low", "medium", "high"})


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    customers = generate_customers(n=50, seed=7, start_date="2021-01-01", end_date="2023-12-31")
    products = generate_products(n=20, seed=7)
    orders = generate_orders(
        customers=customers,
        products=products,
        seed=7,
        start_date="2021-01-01",
        end_date="2023-12-31",
    )
    return customers, products, orders


class TestGenerateOrders:
    def test_required_columns(self, small_dataset):
        _, _, orders = small_dataset
        required = {
            "order_id",
            "customer_id",
            "order_date",
            "quantity",
            "price",
            "total_value",
            "holiday_flag",
            "seasonal_flag",
        }
        assert required.issubset(orders.columns)

    def test_no_future_orders(self, small_dataset):
        _, _, orders = small_dataset
        assert orders["order_date"].max() <= pd.Timestamp("2023-12-31")

    def test_order_dates_after_registration(self, small_dataset):
        customers, _, orders = small_dataset
        merged = orders.merge(customers[["customer_id", "registration_date"]], on="customer_id")
        assert (merged["order_date"] >= merged["registration_date"]).all()

    def test_positive_values(self, small_dataset):
        _, _, orders = small_dataset
        assert (orders["total_value"] > 0).all()
        assert (orders["quantity"] >= 1).all()
        assert (orders["price"] > 0).all()

    def test_product_id_nullable(self, small_dataset):
        _, _, orders = small_dataset
        # Should have some nulls (~5 %)
        null_rate = orders["product_id"].isna().mean()
        assert null_rate < 0.30  # reasonable upper bound

    def test_valid_customer_ids(self, small_dataset):
        customers, _, orders = small_dataset
        valid_ids = set(customers["customer_id"])
        assert set(orders["customer_id"]).issubset(valid_ids)

    def test_order_ids_unique(self, small_dataset):
        _, _, orders = small_dataset
        assert orders["order_id"].nunique() == len(orders)

    def test_reproducibility(self):
        customers = generate_customers(n=20, seed=99)
        products = generate_products(n=10, seed=99)
        o1 = generate_orders(customers, products, seed=99)
        o2 = generate_orders(customers, products, seed=99)
        pd.testing.assert_frame_equal(o1, o2)

    def test_holiday_flag_binary(self, small_dataset):
        _, _, orders = small_dataset
        assert orders["holiday_flag"].isin([0, 1]).all()

    def test_seasonal_flag_binary(self, small_dataset):
        _, _, orders = small_dataset
        assert orders["seasonal_flag"].isin([0, 1]).all()
