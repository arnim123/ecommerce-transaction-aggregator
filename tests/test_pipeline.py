"""Tests for cleaning rules and for the full Olist dataset."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import analyze_transactions
from src.config import DATASET_DIR
from src.data_loader import load_tables_from_directory, match_table_key, read_table
from src.pipeline import build_from_tables, run_from_directory
from src.plotting import save_all_figures
from src.preprocessing import clean_city, normalize_zip


def sample_tables() -> dict[str, pd.DataFrame]:
    """A tiny store with the same columns as the Olist files, plus a few bad rows."""
    return {
        "orders": pd.DataFrame(
            {
                "order_id": ["o1", "o2", "o1"],
                "customer_id": ["c1", "c2", "c1"],
                "order_status": ["delivered", "canceled", "delivered"],
                "order_purchase_timestamp": [
                    "2018-01-01 10:00:00",
                    "2018-01-02 11:00:00",
                    "2018-01-01 10:00:00",
                ],
                "order_approved_at": ["2018-01-01 12:00:00", None, "2018-01-01 18:00:00"],
                "order_delivered_carrier_date": ["2018-01-03 00:00:00", None, "2018-01-03 00:00:00"],
                "order_delivered_customer_date": ["2018-01-05 00:00:00", None, "2018-01-05 00:00:00"],
                "order_estimated_delivery_date": ["2018-01-08 00:00:00", "2018-01-09 00:00:00", "2018-01-08 00:00:00"],
            }
        ),
        "order_items": pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o2", "o1", "o1"],
                "order_item_id": [1, 2, 1, 1, 3],
                "product_id": ["p1", "p2", "p1", "p1", "p1"],
                "seller_id": ["s1", "s1", "s2", "s1", "s1"],
                "price": [10, 30, 50, 10, -5],
                "freight_value": [2, None, 5, 9, 1],
            }
        ),
        "payments": pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o2", "o1"],
                "payment_sequential": [1, 2, 1, 1],
                "payment_type": ["credit_card", "voucher", "boleto", "credit_card"],
                "payment_installments": [2, 1, 1, 2],
                "payment_value": [80, 20, 40, 80],
            }
        ),
        "reviews": pd.DataFrame(
            {
                "review_id": ["r1", "r2", "r3", "r4"],
                "order_id": ["o1", "o1", "o2", "o1"],
                "review_score": [5, 3, 1, 9],
            }
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["c1", "c2"],
                "customer_unique_id": ["u1", "u2"],
                "customer_zip_code_prefix": ["01037", 9790],
                "customer_city": ["São Paulo", "rio de janeiro"],
                "customer_state": ["sp", "rj"],
            }
        ),
        "products": pd.DataFrame(
            {
                "product_id": ["p1", "p2"],
                "product_category_name": ["beleza_saude", None],
                "product_weight_g": [100, None],
                "product_photos_qty": [1, 2],
            }
        ),
        "category_translation": pd.DataFrame(
            {
                "product_category_name": ["beleza_saude"],
                "product_category_name_english": ["health_beauty"],
            }
        ),
        "sellers": pd.DataFrame(
            {
                "seller_id": ["s1"],
                "seller_zip_code_prefix": ["01037"],
                "seller_city": ["campinas"],
                "seller_state": ["SP"],
            }
        ),
    }


class TextNormalizationTests(unittest.TestCase):
    def test_zip_prefixes_keep_leading_zeros(self) -> None:
        normalized = normalize_zip(pd.Series([9790, "01037", None]))
        self.assertEqual(list(normalized.dropna().astype(str)), ["09790", "01037"])

    def test_city_names_drop_accents(self) -> None:
        cities = clean_city(pd.Series(["São Paulo", "rio de janeiro"]))
        self.assertEqual(list(cities.astype(str)), ["Sao Paulo", "Rio de Janeiro"])


class LoaderTests(unittest.TestCase):
    def test_file_names_map_to_the_right_table(self) -> None:
        self.assertEqual(match_table_key("olist_order_items_dataset.csv"), "order_items")
        self.assertEqual(match_table_key("olist_orders_dataset.csv"), "orders")
        self.assertEqual(match_table_key("olist_order_payments_dataset.csv"), "payments")
        self.assertEqual(match_table_key("olist_order_reviews_dataset.csv"), "reviews")
        self.assertEqual(match_table_key("product_category_name_translation.csv"), "category_translation")
        self.assertEqual(match_table_key("olist_products_dataset.csv"), "products")
        self.assertIsNone(match_table_key("notes.csv"))

    def test_missing_folder_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_tables_from_directory(Path("/tmp/olist-dataset-does-not-exist"), require_all=True)

    def test_missing_columns_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "orders.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                read_table(path, "orders", display_name="orders.csv")
        self.assertIn("order_id", str(caught.exception))


class SyntheticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = build_from_tables(sample_tables())

    def test_cleaning_counts(self) -> None:
        overview = self.result.table_overview.set_index("table")
        self.assertEqual(int(overview.loc["orders", "key_duplicates_removed"]), 1)
        self.assertEqual(int(overview.loc["orders", "rows_after_cleaning"]), 2)
        self.assertEqual(int(overview.loc["order_items", "key_duplicates_removed"]), 1)
        self.assertEqual(int(overview.loc["order_items", "rows_removed_invalid"]), 1)
        self.assertEqual(int(overview.loc["order_items", "rows_after_cleaning"]), 3)
        self.assertEqual(int(overview.loc["payments", "exact_duplicates_removed"]), 1)

    def test_revenue_identity_and_payment_aggregation(self) -> None:
        transactions = self.result.transactions
        self.assertEqual(len(transactions), 3)
        actual = sorted(np.round(transactions["line_revenue"].to_numpy(dtype=float), 2).tolist())
        self.assertEqual(actual, [12.0, 30.0, 55.0])
        order_one = transactions.loc[transactions["order_id"] == "o1"]
        self.assertTrue(np.allclose(order_one["order_payment_total"], 100))
        self.assertTrue((order_one["payment_type"] == "credit_card").all())
        self.assertTrue(np.allclose(order_one["review_score"], 4))
        self.assertTrue((order_one["review_count"] == 2).all())
        self.assertAlmostEqual(float(self.result.headline["item_revenue_all"]), 97.0, places=2)
        self.assertAlmostEqual(float(self.result.headline["item_revenue_eligible"]), 42.0, places=2)
        self.assertAlmostEqual(float(self.result.headline["payment_revenue_all"]), 140.0, places=2)
        self.assertFalse(transactions.loc[transactions["order_status"] == "canceled", "counts_as_revenue"].any())

    def test_categories_places_and_timing(self) -> None:
        transactions = self.result.transactions
        labels = set(transactions["product_category_label"].astype(str))
        self.assertIn("Health Beauty", labels)
        self.assertIn("Unknown", labels)
        self.assertTrue((transactions["product_weight_g"] == 100).all())
        cities = set(transactions["customer_city"].astype(str))
        self.assertIn("Sao Paulo", cities)
        self.assertIn("Rio de Janeiro", cities)
        zips = set(transactions["customer_zip_code_prefix"].astype(str))
        self.assertIn("09790", zips)
        order_one = transactions.loc[transactions["order_id"] == "o1"].iloc[0]
        self.assertAlmostEqual(float(order_one["delivery_days"]), 3.58, places=2)
        self.assertAlmostEqual(float(order_one["delivery_delay_days"]), -3.0, places=2)
        self.assertTrue(bool(order_one["same_state"]))

    def test_one_hot_and_quality_checks(self) -> None:
        encoded = self.result.encoded
        self.assertEqual(len(encoded), len(self.result.transactions))
        for column in (
            "order_status_delivered",
            "order_status_canceled",
            "payment_type_credit_card",
            "product_category_health_beauty",
            "product_category_unknown",
            "customer_state_sp",
        ):
            self.assertIn(column, encoded.columns)
        self.assertTrue((self.result.sanity_checks["status"] == "Pass").all())

    def test_default_analysis_excludes_canceled_orders(self) -> None:
        kpis = self.result.default_analysis.kpis
        self.assertAlmostEqual(float(kpis["item_revenue"]), 42.0, places=2)
        self.assertEqual(int(kpis["orders"]), 1)
        self.assertEqual(int(kpis["units"]), 2)
        self.assertEqual(int(kpis["customers"]), 1)
        self.assertAlmostEqual(float(kpis["average_review"]), 4.0, places=2)
        self.assertAlmostEqual(float(kpis["payment_revenue"]), 100.0, places=2)
        all_rows = analyze_transactions(self.result.transactions)
        self.assertAlmostEqual(float(all_rows.kpis["item_revenue"]), 97.0, places=2)
        self.assertAlmostEqual(float(all_rows.kpis["payment_revenue"]), 140.0, places=2)

    def test_matplotlib_figures_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            saved = save_all_figures(self.result.default_analysis, Path(temporary))
            self.assertEqual(len(saved), 7)
            for path in saved:
                self.assertEqual(path.suffix, ".png")
                self.assertGreater(path.stat().st_size, 1000)

    def test_location_join_is_skipped_when_the_file_is_absent(self) -> None:
        step = self.result.merge_report.loc[
            self.result.merge_report["step"] == "Join customer location"
        ].iloc[0]
        self.assertIn("Skipped", step["detail"])


@unittest.skipUnless(DATASET_DIR.exists(), "Olist dataset folder is not present")
class OlistDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_from_directory(DATASET_DIR)

    def test_real_dataset_shape_and_revenue(self) -> None:
        transactions = self.result.transactions
        self.assertGreater(len(transactions), 100_000)
        self.assertLess(len(transactions), 130_000)
        self.assertFalse(transactions.duplicated(["order_id", "order_item_id"]).any())
        self.assertGreater(float(self.result.headline["item_revenue_eligible"]), 5_000_000)
        self.assertLess(float(self.result.headline["item_revenue_eligible"]), 50_000_000)
        self.assertEqual(int(transactions["product_category"].isna().sum()), 0)
        self.assertEqual(int(transactions["freight_value"].isna().sum()), 0)
        self.assertTrue((self.result.sanity_checks["status"] == "Pass").all())
        categories = set(transactions["product_category"].astype(str))
        self.assertIn("health_beauty", categories)
        self.assertIn("bed_bath_table", categories)

    def test_real_joins_match(self) -> None:
        report = self.result.merge_report.set_index("step")
        for step, minimum in (
            ("Join orders", 0.99),
            ("Join customers", 0.99),
            ("Join products", 0.99),
            ("Join category translation", 0.90),
            ("Join payments", 0.99),
            ("Join reviews", 0.90),
            ("Join customer location", 0.80),
        ):
            self.assertGreaterEqual(float(report.loc[step, "match_rate"]), minimum, step)
        top_state = self.result.default_analysis.by_state.iloc[0]["customer_state"]
        self.assertEqual(top_state, "SP")


if __name__ == "__main__":
    unittest.main()
