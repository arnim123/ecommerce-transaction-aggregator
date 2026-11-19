"""Paths, table names, and column roles shared by the experiments and the app."""

from __future__ import annotations

from pathlib import Path

# src/config.py -> project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Orders in these statuses are kept in the merged file for inspection, but they
# are left out of the default revenue totals. "unknown" means the status was
# blank, so we do not treat it as a sale.
REVENUE_EXCLUDED_STATUSES = frozenset({"canceled", "unavailable", "unknown"})

# The official translation file is missing English names for these two
# categories. They are real Olist categories, so a short supplement covers them.
EXTRA_CATEGORY_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers",
}

# Categorical columns used in Experiment 2. City and id columns are excluded
# on purpose; see src/encoding.py for the explanation shown in the app.
CATEGORICAL_COLUMNS = (
    "order_status",
    "payment_type",
    "customer_state",
    "seller_state",
    "product_category",
)

# Kept so an encoded row can be traced back to the cleaned table.
# They are identifiers, not model features.
ENCODED_ID_COLUMNS = ("order_id", "order_item_id")

# Numbers are already usable by a model, so they are not one-hot encoded.
ENCODED_NUMERIC_COLUMNS = (
    "price",
    "freight_value",
    "line_revenue",
    "payment_installments",
    "payment_count",
    "order_payment_total",
    "review_score",
    "review_count",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "purchase_year",
    "purchase_month",
    "delivery_days",
    "delivery_delay_days",
    "same_state",
)

NUMERIC_PRODUCT_COLUMNS = (
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
)

# Display names for the two-letter Brazilian state codes on the charts.
BRAZIL_STATE_NAMES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}

COLOR_TEAL = "#0F766E"
COLOR_CLAY = "#C2410C"
COLOR_INK = "#1C1917"
COLOR_MUTED = "#57534E"
COLOR_GRID = "#E7E5E4"

# filename is matched exactly when reading the bundled dataset folder.
# required=True means the merge cannot run without that file.
TABLES: dict[str, dict[str, object]] = {
    "customers": {
        "label": "Customers",
        "filename": "olist_customers_dataset.csv",
        "required": False,
        "required_columns": [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ],
        "key_columns": ["customer_id"],
    },
    "geolocation": {
        "label": "Geolocation",
        "filename": "olist_geolocation_dataset.csv",
        "required": False,
        "required_columns": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
        ],
        "key_columns": [],
    },
    "sellers": {
        "label": "Sellers",
        "filename": "olist_sellers_dataset.csv",
        "required": False,
        "required_columns": [
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ],
        "key_columns": ["seller_id"],
    },
    "products": {
        "label": "Products",
        "filename": "olist_products_dataset.csv",
        "required": False,
        "required_columns": ["product_id", "product_category_name"],
        "key_columns": ["product_id"],
    },
    "category_translation": {
        "label": "Category translation",
        "filename": "product_category_name_translation.csv",
        "required": False,
        "required_columns": [
            "product_category_name",
            "product_category_name_english",
        ],
        "key_columns": ["product_category_name"],
    },
    "orders": {
        "label": "Orders",
        "filename": "olist_orders_dataset.csv",
        "required": True,
        "required_columns": [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ],
        "key_columns": ["order_id"],
    },
    "order_items": {
        "label": "Order items",
        "filename": "olist_order_items_dataset.csv",
        "required": True,
        "required_columns": [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ],
        "key_columns": ["order_id", "order_item_id"],
    },
    "payments": {
        "label": "Payments",
        "filename": "olist_order_payments_dataset.csv",
        "required": False,
        "required_columns": [
            "order_id",
            "payment_type",
            "payment_installments",
            "payment_value",
        ],
        "key_columns": [],
    },
    "reviews": {
        "label": "Reviews",
        "filename": "olist_order_reviews_dataset.csv",
        "required": False,
        "required_columns": ["order_id", "review_score"],
        "key_columns": [],
    },
}

# Longer aliases are listed first so "order_items" is not confused with "orders".
FILE_MATCH_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("category_translation", ("product_category_name_translation", "category_translation")),
    ("order_items", ("order_items",)),
    ("payments", ("order_payments", "payments")),
    ("reviews", ("order_reviews", "reviews")),
    ("geolocation", ("geolocation",)),
    ("customers", ("customers",)),
    ("products", ("products",)),
    ("sellers", ("sellers",)),
    ("orders", ("orders",)),
)

DATA_DICTIONARY: tuple[tuple[str, str, str], ...] = (
    ("order_id", "Order", "Id of the purchase. Several item rows can share one order id."),
    ("order_item_id", "Item", "Unit number inside the order. One row means one unit."),
    ("order_status", "Order", "Where the order is in the pipeline, such as delivered or canceled."),
    ("order_purchase_timestamp", "Order", "When the customer placed the order."),
    ("customer_id", "Customer", "Id created for a single order. It is not the person."),
    ("customer_unique_id", "Customer", "Id of the person. Use this to count customers."),
    ("customer_city", "Customer", "Customer city, with accents removed so spellings group together."),
    ("customer_state", "Customer", "Two-letter Brazilian state code of the customer."),
    ("product_id", "Product", "Id of the product that was sold."),
    ("product_category", "Product", "English category slug used for one-hot encoding."),
    ("product_category_label", "Product", "Readable category name used on the charts."),
    ("product_category_name", "Product", "Original Portuguese category name from the product file."),
    ("seller_id", "Seller", "Id of the seller who fulfilled this item."),
    ("seller_state", "Seller", "Two-letter state code of the seller."),
    ("price", "Revenue", "Price of this one unit, in Brazilian reals."),
    ("freight_value", "Revenue", "Freight charged for this one unit. Blank freight becomes 0."),
    ("line_revenue", "Revenue", "Price plus freight for this unit. Add this column to get revenue."),
    ("payment_type", "Payment", "Method that paid the largest share of the order."),
    ("payment_installments", "Payment", "Largest installment count among the order's payments."),
    ("payment_count", "Payment", "How many payment rows the order had."),
    ("order_payment_total", "Payment", "Sum of payments for the whole order. Do not add it across items."),
    ("review_score", "Review", "Average review score for the order, from 1 to 5."),
    ("review_count", "Review", "How many reviews the order received."),
    ("purchase_year", "Time", "Year of the purchase. It is a number, so it is not one-hot encoded."),
    ("purchase_month", "Time", "Month number of the purchase, from 1 to 12."),
    ("delivery_days", "Delivery", "Days from purchase to delivery. Blank when the order was not delivered."),
    ("delivery_delay_days", "Delivery", "Delivered date minus the estimate. Negative means early."),
    ("same_state", "Delivery", "True when the customer and the seller are in the same state."),
    ("counts_as_revenue", "Revenue", "False for canceled, unavailable, and unknown statuses."),
    ("customer_latitude", "Location", "Rough center of the customer zip prefix, averaged from the location file."),
    ("customer_longitude", "Location", "Rough center of the customer zip prefix."),
)
