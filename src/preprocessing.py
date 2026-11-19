"""Experiment 1: clean every table, merge to one row per item, and compute revenue.

The grain of the result is one product unit. Olist does not store a quantity
column. Two units of the same product are two rows, so the row count is the
unit count. Revenue for a row is that unit's price plus its freight.

Payments, reviews, and zip-code coordinates are collapsed before the join.
Joining them raw would repeat item rows and inflate revenue.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    EXTRA_CATEGORY_TRANSLATIONS,
    NUMERIC_PRODUCT_COLUMNS,
    REVENUE_EXCLUDED_STATUSES,
    TABLES,
)

_MISSING_TEXT = {"", "nan", "none", "null", "<na>", "n/a", "na"}
_SMALL_WORDS = {"and", "of", "the", "de", "da", "do", "das", "dos", "e", "for"}
_ORDER_DATES = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)


@dataclass
class TableCleanResult:
    """What happened to one CSV while it was cleaned."""

    name: str
    frame: pd.DataFrame
    rows_raw: int
    rows_clean: int
    columns: int
    exact_duplicates_removed: int
    key_duplicates_removed: int
    rows_removed_invalid: int
    actions: list[str] = field(default_factory=list)
    missing_rows: list[dict[str, object]] = field(default_factory=list)


def clean_label(series: pd.Series, case: str = "lower") -> pd.Series:
    """Trim text, turn blanks into missing values, and normalize case."""
    labels = series.astype("string").str.strip()
    labels = labels.str.replace(r"\s+", " ", regex=True)
    labels = labels.mask(labels.str.lower().isin(_MISSING_TEXT))
    if case == "lower":
        labels = labels.str.lower()
    elif case == "upper":
        labels = labels.str.upper()
    return labels


def strip_accents(series: pd.Series) -> pd.Series:
    """Remove accents so 'São Paulo' and 'Sao Paulo' can be grouped."""

    def _one(value: object) -> object:
        if value is None or pd.isna(value):
            return pd.NA
        text = unicodedata.normalize("NFKD", str(value))
        return "".join(char for char in text if not unicodedata.combining(char))

    return series.map(_one).astype("string")


def pretty_words(value: object) -> object:
    """Capitalize a city or category without capitalizing small words like 'de'."""
    if value is None or pd.isna(value):
        return pd.NA
    words = str(value).split(" ")
    pretty: list[str] = []
    for index, word in enumerate(words):
        lower = word.lower()
        if index > 0 and lower in _SMALL_WORDS:
            pretty.append(lower)
        else:
            pretty.append(lower.capitalize())
    return " ".join(pretty)


def clean_city(series: pd.Series) -> pd.Series:
    """Normalize a city name for grouping and for display."""
    normalized = clean_label(strip_accents(series), case="lower")
    return normalized.map(pretty_words).astype("string")


def clean_category_slug(series: pd.Series) -> pd.Series:
    """Normalize a category into a lowercase slug with underscores."""
    labels = clean_label(strip_accents(series), case="lower")
    return labels.str.replace(" ", "_", regex=False)


def category_label(series: pd.Series) -> pd.Series:
    """Turn a category slug into a readable label."""
    text = series.astype("string").str.replace("_", " ", regex=False)
    return text.map(pretty_words).astype("string")


def normalize_zip(series: pd.Series) -> pd.Series:
    """Store zip prefixes as 5-character text so leading zeros survive.

    Pandas may read 09790 as the integer 9790. Padding restores the prefix
    used by the location file, which is what the join matches on.
    """
    text = series.astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D", "", regex=True)
    digits = digits.mask(digits.isna() | (digits.str.len().fillna(0) == 0))
    return digits.str.zfill(5)


def as_id(series: pd.Series) -> pd.Series:
    """Clean an identifier so joins are not broken by spaces or a trailing .0."""
    text = series.astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.mask(text.str.lower().isin(_MISSING_TEXT))


def _coerce_number(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert a column to numbers. The count is values that were not numeric."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = int((series.notna() & numeric.isna()).sum())
    return numeric, invalid


def _coerce_datetime(series: pd.Series) -> tuple[pd.Series, int]:
    parsed = pd.to_datetime(series, errors="coerce")
    invalid = int((series.notna() & parsed.isna()).sum())
    return parsed, invalid


def _begin(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, int]]:
    work = frame.copy()
    missing_before = {column: int(work[column].isna().sum()) for column in work.columns}
    return work, len(frame), missing_before


def _finish(
    name: str,
    raw_rows: int,
    raw_columns: int,
    work: pd.DataFrame,
    missing_before: dict[str, int],
    exact_duplicates: int,
    key_duplicates: int,
    invalid_rows: int,
    actions: list[str],
    treatments: dict[str, str],
) -> TableCleanResult:
    work = work.reset_index(drop=True)
    accounted = len(work) + exact_duplicates + key_duplicates + invalid_rows
    if accounted != raw_rows:
        raise ValueError(
            f"{name} row counts do not add up: loaded {raw_rows}, "
            f"clean {len(work)}, exact duplicates {exact_duplicates}, "
            f"key duplicates {key_duplicates}, invalid {invalid_rows}."
        )
    missing_rows: list[dict[str, object]] = []
    for column in work.columns:
        before = int(missing_before.get(column, 0))
        after = int(work[column].isna().sum())
        treatment = treatments.get(column, "Left unchanged")
        if before == 0 and after == 0 and treatment == "Left unchanged":
            continue
        missing_rows.append(
            {
                "table": name,
                "column": column,
                "missing_before": before,
                "missing_after": after,
                "treatment": treatment,
            }
        )
    # Columns that were dropped (for example review rows) still matter.
    for column, before in missing_before.items():
        if column in work.columns or before == 0:
            continue
        missing_rows.append(
            {
                "table": name,
                "column": column,
                "missing_before": before,
                "missing_after": 0,
                "treatment": treatments.get(column, "Column not kept after cleaning"),
            }
        )
    return TableCleanResult(
        name=name,
        frame=work,
        rows_raw=raw_rows,
        rows_clean=len(work),
        columns=raw_columns,
        exact_duplicates_removed=exact_duplicates,
        key_duplicates_removed=key_duplicates,
        rows_removed_invalid=invalid_rows,
        actions=actions,
        missing_rows=missing_rows,
    )


def _drop_exact(work: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(work)
    work = work.drop_duplicates()
    return work, before - len(work)


def _drop_key(work: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, int]:
    if not keys or any(key not in work.columns for key in keys):
        return work, 0
    before = len(work)
    work = work.drop_duplicates(subset=keys, keep="first")
    return work, before - len(work)


def clean_customers(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "customer_id is a new id for every order. customer_unique_id identifies the person, so customer counts use that column."
    ]
    treatments: dict[str, str] = {}
    work["customer_id"] = as_id(work["customer_id"])
    work["customer_unique_id"] = as_id(work["customer_unique_id"])
    work["customer_zip_code_prefix"] = normalize_zip(work["customer_zip_code_prefix"])
    work["customer_city"] = clean_city(work["customer_city"])
    work["customer_state"] = clean_label(work["customer_state"], case="upper")
    treatments["customer_city"] = "Accents removed and capitalization standardized so the same city groups together."
    treatments["customer_state"] = "State codes were uppercased."
    treatments["customer_zip_code_prefix"] = "Zip prefixes were stored as 5-character text so leading zeros are kept."
    work, exact = _drop_exact(work)
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate customer rows.")
    invalid = work["customer_id"].isna()
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} customers with no customer_id.")
        treatments["customer_id"] = "Rows with a missing customer id were removed."
    work, key_dups = _drop_key(work, ["customer_id"])
    if key_dups:
        actions.append(f"Kept the first row for {key_dups:,} repeated customer ids.")
    return _finish(
        "customers", raw_rows, frame.shape[1], work, missing_before,
        exact, key_dups, invalid_count, actions, treatments,
    )


def clean_sellers(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions: list[str] = []
    treatments: dict[str, str] = {}
    work["seller_id"] = as_id(work["seller_id"])
    work["seller_zip_code_prefix"] = normalize_zip(work["seller_zip_code_prefix"])
    work["seller_city"] = clean_city(work["seller_city"])
    work["seller_state"] = clean_label(work["seller_state"], case="upper")
    treatments["seller_city"] = "Accents removed and capitalization standardized."
    treatments["seller_state"] = "State codes were uppercased."
    treatments["seller_zip_code_prefix"] = "Zip prefixes were stored as 5-character text."
    work, exact = _drop_exact(work)
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate seller rows.")
    invalid = work["seller_id"].isna()
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} sellers with no seller_id.")
    work, key_dups = _drop_key(work, ["seller_id"])
    if key_dups:
        actions.append(f"Kept the first row for {key_dups:,} repeated seller ids.")
    if not actions:
        actions.append("Seller ids, cities, states, and zip prefixes were standardized.")
    return _finish(
        "sellers", raw_rows, frame.shape[1], work, missing_before,
        exact, key_dups, invalid_count, actions, treatments,
    )


def clean_products(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "The product file spells length as 'lenght'. That original column name is kept so it still matches the source."
    ]
    treatments: dict[str, str] = {}
    work["product_id"] = as_id(work["product_id"])
    work["product_category_name"] = clean_category_slug(work["product_category_name"])
    missing_category = int(work["product_category_name"].isna().sum())
    work["product_category_name"] = work["product_category_name"].fillna("unknown")
    treatments["product_category_name"] = "Blank categories were filled with 'unknown'."
    if missing_category:
        actions.append(f"Filled {missing_category:,} blank product categories with 'unknown'.")
    for column in NUMERIC_PRODUCT_COLUMNS:
        if column not in work.columns:
            continue
        converted, invalid_numbers = _coerce_number(work[column])
        work[column] = converted
        if invalid_numbers:
            actions.append(f"{column} had {invalid_numbers:,} values that were not numbers. They were treated as missing.")
        median = work[column].median()
        missing = int(work[column].isna().sum())
        if missing and pd.notna(median):
            work[column] = work[column].fillna(median)
            treatments[column] = f"Filled {missing:,} missing values with the median ({median:g})."
            actions.append(f"{column}: filled {missing:,} missing values with the median {median:g}.")
        elif missing:
            treatments[column] = "The whole column was missing, so there was no median to use."
    work, exact = _drop_exact(work)
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate product rows.")
    invalid = work["product_id"].isna()
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} products with no product_id.")
    work, key_dups = _drop_key(work, ["product_id"])
    if key_dups:
        actions.append(f"Kept the first row for {key_dups:,} repeated product ids.")
    return _finish(
        "products", raw_rows, frame.shape[1], work, missing_before,
        exact, key_dups, invalid_count, actions, treatments,
    )


def clean_translation(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = ["Category names were normalized to the same slug style as the product file."]
    treatments: dict[str, str] = {}
    work["product_category_name"] = clean_category_slug(work["product_category_name"])
    work["product_category_name_english"] = clean_category_slug(work["product_category_name_english"])
    treatments["product_category_name"] = "Slugs were lowercased so they match the product table."
    treatments["product_category_name_english"] = "English names were stored as slugs. The charts use a readable label."
    work, exact = _drop_exact(work)
    invalid = work["product_category_name"].isna() | work["product_category_name_english"].isna()
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    work, key_dups = _drop_key(work, ["product_category_name"])
    if exact or key_dups or invalid_count:
        actions.append(
            f"Removed {exact:,} exact duplicates, {key_dups:,} repeated categories, and {invalid_count:,} blank translation rows."
        )
    return _finish(
        "category_translation", raw_rows, frame.shape[1], work, missing_before,
        exact, key_dups, invalid_count, actions, treatments,
    )


def clean_orders(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "Missing delivery timestamps were kept. A blank delivery date usually means the event has not happened, not that the date was forgotten."
    ]
    treatments: dict[str, str] = {}
    work["order_id"] = as_id(work["order_id"])
    work["customer_id"] = as_id(work["customer_id"])
    work["order_status"] = clean_label(work["order_status"], case="lower").fillna("unknown")
    treatments["order_status"] = "Blank statuses were labeled 'unknown' and are excluded from revenue."
    for column in _ORDER_DATES:
        if column not in work.columns:
            actions.append(f"{column} is not in the file, so that point in the timeline cannot be measured.")
            continue
        parsed, invalid_dates = _coerce_datetime(work[column])
        work[column] = parsed
        missing = int(work[column].isna().sum())
        treatments[column] = (
            "Left missing on purpose. Inventing a delivery date would create orders that were never delivered."
        )
        if invalid_dates:
            actions.append(f"{column} had {invalid_dates:,} values that were not dates. They were set to missing.")
        if missing and column != "order_purchase_timestamp":
            actions.append(f"{column} is missing on {missing:,} orders and was left missing.")
    work, exact = _drop_exact(work)
    invalid = work["order_id"].isna()
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} orders with no order_id.")
        treatments["order_id"] = "Rows with a missing order id were removed."
    work, key_dups = _drop_key(work, ["order_id"])
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate order rows.")
    if key_dups:
        actions.append(f"Kept the first row for {key_dups:,} repeated order ids.")
    return _finish(
        "orders", raw_rows, frame.shape[1], work, missing_before,
        exact, key_dups, invalid_count, actions, treatments,
    )


def clean_order_items(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "Each row is one unit. Line revenue is calculated later as price plus freight."
    ]
    treatments: dict[str, str] = {}
    work["order_id"] = as_id(work["order_id"])
    work["product_id"] = as_id(work["product_id"])
    work["seller_id"] = as_id(work["seller_id"])
    item_number = pd.to_numeric(work["order_item_id"], errors="coerce").round()
    work["order_item_id"] = item_number.astype("Int64")
    if "shipping_limit_date" in work.columns:
        parsed, invalid_dates = _coerce_datetime(work["shipping_limit_date"])
        work["shipping_limit_date"] = parsed
        treatments["shipping_limit_date"] = "Unreadable dates were set to missing and kept."
        if invalid_dates:
            actions.append(f"shipping_limit_date had {invalid_dates:,} unreadable values.")
    work, exact = _drop_exact(work)
    price, bad_price_text = _coerce_number(work["price"])
    freight, bad_freight_text = _coerce_number(work["freight_value"])
    work["price"] = price
    work["freight_value"] = freight
    if bad_price_text:
        actions.append(f"{bad_price_text:,} price values were not numbers and were treated as missing.")
    if bad_freight_text:
        actions.append(f"{bad_freight_text:,} freight values were not numbers and were treated as missing.")
    # Prefer a usable price when the same item was loaded more than once.
    work = work.assign(_bad_price=work["price"].isna() | (work["price"] < 0))
    work = work.sort_values("_bad_price", kind="mergesort").drop(columns="_bad_price")
    work, key_dups = _drop_key(work, ["order_id", "order_item_id"])
    missing_keys = work["order_id"].isna() | work["order_item_id"].isna()
    invalid_key_count = int(missing_keys.sum())
    work = work.loc[~missing_keys].copy()
    bad_price = work["price"].isna() | (work["price"] < 0)
    bad_price_count = int(bad_price.sum())
    work = work.loc[~bad_price].copy()
    if bad_price_count:
        actions.append(f"Removed {bad_price_count:,} items with a missing or negative price, because revenue would be undefined.")
        treatments["price"] = "Rows with a missing or negative price were removed."
    negative_freight = work["freight_value"].notna() & (work["freight_value"] < 0)
    if int(negative_freight.sum()):
        actions.append(f"Replaced {int(negative_freight.sum()):,} negative freight values with 0.")
        work.loc[negative_freight, "freight_value"] = 0
    missing_freight = int(work["freight_value"].isna().sum())
    work["freight_value"] = work["freight_value"].fillna(0)
    treatments["freight_value"] = "Missing freight was filled with 0, meaning no freight was charged."
    if missing_freight:
        actions.append(f"Filled {missing_freight:,} missing freight values with 0.")
    zero_prices = int((work["price"] == 0).sum())
    if zero_prices:
        actions.append(f"Kept {zero_prices:,} items with a price of 0. A zero price can be a promotion or a gift.")
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate item rows.")
    if key_dups:
        actions.append(f"Kept the first copy of {key_dups:,} repeated order-item keys.")
    invalid_count = invalid_key_count + bad_price_count
    if work.empty:
        raise ValueError("No order items were left after cleaning. Check the price column.")
    return _finish(
        "order_items", raw_rows, frame.shape[1], work, missing_before,
        exact, key_dups, invalid_count, actions, treatments,
    )


def clean_payments(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "An order can have several payment rows, for example a voucher plus a credit card. Those rows are summed later, once per order."
    ]
    treatments: dict[str, str] = {}
    work["order_id"] = as_id(work["order_id"])
    work["payment_type"] = clean_label(work["payment_type"], case="lower").fillna("unknown")
    treatments["payment_type"] = "Blank payment types were labeled 'unknown'."
    value, bad_values = _coerce_number(work["payment_value"])
    installments, bad_installments = _coerce_number(work["payment_installments"])
    work["payment_value"] = value
    work["payment_installments"] = installments
    if bad_values:
        actions.append(f"{bad_values:,} payment amounts were not numbers.")
    if bad_installments:
        actions.append(f"{bad_installments:,} installment counts were not numbers.")
    if "payment_sequential" in work.columns:
        sequential, _ = _coerce_number(work["payment_sequential"])
        work["payment_sequential"] = sequential
    work, exact = _drop_exact(work)
    missing = work["order_id"].isna() | work["payment_value"].isna() | (work["payment_value"] < 0)
    invalid_count = int(missing.sum())
    work = work.loc[~missing].copy()
    treatments["payment_value"] = "Rows with a missing or negative payment amount were removed."
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate payment rows so the same payment is not summed twice.")
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} payment rows with no order id or no usable amount.")
    # Repeated order ids are valid here. Do not drop them.
    return _finish(
        "payments", raw_rows, frame.shape[1], work, missing_before,
        exact, 0, invalid_count, actions, treatments,
    )


def clean_reviews(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "Review comments are mostly blank. They stay in this table for preview, but they are not copied into the revenue table because this project studies numbers and categories."
    ]
    treatments: dict[str, str] = {
        "review_comment_title": "Blank comments were left blank. Comment text is not imputed.",
        "review_comment_message": "Blank comments were left blank. Comment text is not imputed.",
    }
    work["order_id"] = as_id(work["order_id"])
    score, bad_score = _coerce_number(work["review_score"])
    out_of_range = score.notna() & ~score.between(1, 5)
    if bad_score or int(out_of_range.sum()):
        actions.append(
            f"Ignored {bad_score + int(out_of_range.sum()):,} review scores that were not a number from 1 to 5."
        )
    score = score.where(score.between(1, 5))
    work["review_score"] = score
    treatments["review_score"] = "Scores outside 1 to 5 were removed with their rows. Remaining scores were not imputed."
    work, exact = _drop_exact(work)
    invalid = work["order_id"].isna() | work["review_score"].isna()
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate review rows.")
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} reviews with no order id or no usable score.")
    return _finish(
        "reviews", raw_rows, frame.shape[1], work, missing_before,
        exact, 0, invalid_count, actions, treatments,
    )


def clean_geolocation(frame: pd.DataFrame) -> TableCleanResult:
    work, raw_rows, missing_before = _begin(frame)
    actions = [
        "Each zip prefix has many coordinate rows. They are averaged later so joining locations cannot duplicate orders and inflate revenue."
    ]
    treatments: dict[str, str] = {}
    work["geolocation_zip_code_prefix"] = normalize_zip(work["geolocation_zip_code_prefix"])
    latitude, bad_lat = _coerce_number(work["geolocation_lat"])
    longitude, bad_lng = _coerce_number(work["geolocation_lng"])
    work["geolocation_lat"] = latitude
    work["geolocation_lng"] = longitude
    treatments["geolocation_zip_code_prefix"] = "Zip prefixes were stored as 5-character text."
    treatments["geolocation_lat"] = "Coordinates outside the valid latitude range were removed."
    treatments["geolocation_lng"] = "Coordinates outside the valid longitude range were removed."
    if bad_lat or bad_lng:
        actions.append(f"Ignored {bad_lat + bad_lng:,} coordinate values that were not numbers.")
    work, exact = _drop_exact(work)
    invalid = (
        work["geolocation_zip_code_prefix"].isna()
        | work["geolocation_lat"].isna()
        | work["geolocation_lng"].isna()
        | ~work["geolocation_lat"].between(-90, 90)
        | ~work["geolocation_lng"].between(-180, 180)
    )
    invalid_count = int(invalid.sum())
    work = work.loc[~invalid].copy()
    if exact:
        actions.append(f"Removed {exact:,} exact duplicate location rows.")
    if invalid_count:
        actions.append(f"Removed {invalid_count:,} location rows with a missing zip or an impossible coordinate.")
    # Keep only the columns the join needs, but do that after the duplicate
    # count so "exact duplicate" means a duplicate of the original row.
    keep = ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
    work = work[keep].copy()
    return _finish(
        "geolocation", raw_rows, frame.shape[1], work, missing_before,
        exact, 0, invalid_count, actions, treatments,
    )


CLEANERS = {
    "customers": clean_customers,
    "sellers": clean_sellers,
    "products": clean_products,
    "category_translation": clean_translation,
    "orders": clean_orders,
    "order_items": clean_order_items,
    "payments": clean_payments,
    "reviews": clean_reviews,
    "geolocation": clean_geolocation,
}


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Collapse payment rows to one row per order.

    The payment type that is kept is the method with the largest amount.
    The total is the sum of every method, so a voucher plus a card is one total.
    """
    columns = [
        "order_id",
        "payment_type",
        "payment_installments",
        "payment_count",
        "order_payment_total",
    ]
    if payments.empty:
        return pd.DataFrame(columns=columns)
    sort_columns = ["order_id", "payment_value"]
    ascending = [True, False]
    if "payment_sequential" in payments.columns:
        sort_columns.append("payment_sequential")
        ascending.append(True)
    ordered = payments.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    primary = ordered.drop_duplicates("order_id", keep="first")[["order_id", "payment_type"]]
    totals = payments.groupby("order_id", as_index=False).agg(
        order_payment_total=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_count=("order_id", "size"),
    )
    totals["order_payment_total"] = np.round(totals["order_payment_total"].to_numpy(dtype=float), 2)
    return totals.merge(primary, on="order_id", how="left")[columns]


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Collapse reviews to one average score per order."""
    columns = ["order_id", "review_score", "review_count"]
    if reviews.empty:
        return pd.DataFrame(columns=columns)
    grouped = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("order_id", "size"),
    )
    grouped["review_score"] = grouped["review_score"].round(2)
    return grouped


def aggregate_geolocation(locations: pd.DataFrame) -> pd.DataFrame:
    """Average many coordinates into one center per zip prefix."""
    columns = ["zip_prefix", "latitude", "longitude"]
    if locations.empty:
        return pd.DataFrame(columns=columns)
    grouped = locations.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        latitude=("geolocation_lat", "mean"),
        longitude=("geolocation_lng", "mean"),
    )
    grouped["latitude"] = grouped["latitude"].round(5)
    grouped["longitude"] = grouped["longitude"].round(5)
    return grouped.rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})


def _take(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    return frame[present].copy()


def _left_join(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    on: str,
    step: str,
    steps: list[dict[str, object]],
    detail: str,
) -> pd.DataFrame:
    """Left-join a lookup that must have one row per key.

    A left join keeps every item even when the lookup has no match.
    validate='m:1' stops the join if the lookup still has duplicate keys,
    which would silently multiply revenue.
    """
    if right is None or right.empty or on not in right.columns:
        steps.append(
            {
                "step": step,
                "rows": len(left),
                "matched_rows": 0,
                "match_rate": 0.0,
                "detail": "Skipped because that table was not loaded.",
            }
        )
        return left
    overlap = [column for column in right.columns if column in left.columns and column != on]
    if overlap:
        right = right.drop(columns=overlap)
    try:
        merged = left.merge(right, how="left", on=on, validate="m:1", indicator=True)
    except pd.errors.MergeError as exc:
        raise ValueError(
            f"{step} failed because the lookup table has duplicate keys. {exc}"
        ) from exc
    matched = int((merged["_merge"] == "both").sum())
    rows = len(merged)
    merged = merged.drop(columns="_merge")
    steps.append(
        {
            "step": step,
            "rows": rows,
            "matched_rows": matched,
            "match_rate": (matched / rows) if rows else 0.0,
            "detail": detail,
        }
    )
    return merged


def assemble_transactions(
    frames: dict[str, pd.DataFrame],
    payments: pd.DataFrame | None,
    reviews: pd.DataFrame | None,
    locations: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Join the cleaned tables onto order items and add revenue columns."""
    if "order_items" not in frames or "orders" not in frames:
        raise ValueError("Orders and order items are required before the tables can be merged.")

    steps: list[dict[str, object]] = []
    items = frames["order_items"]
    steps.append(
        {
            "step": "Order items (starting point)",
            "rows": len(items),
            "matched_rows": len(items),
            "match_rate": 1.0,
            "detail": "One row per unit. This is the grain of the merged table, so summing line revenue does not double-count.",
        }
    )
    orders = _take(
        frames["orders"],
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    merged = _left_join(
        items,
        orders,
        "order_id",
        "Join orders",
        steps,
        "Left join on order_id. Every item should find its order.",
    )
    if "customers" in frames:
        customers = _take(
            frames["customers"],
            [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
            ],
        )
        merged = _left_join(
            merged,
            customers,
            "customer_id",
            "Join customers",
            steps,
            "Left join on customer_id.",
        )
    else:
        _left_join(merged, None, "customer_id", "Join customers", steps, "")

    if "products" in frames:
        products = _take(
            frames["products"],
            ["product_id", "product_category_name", *NUMERIC_PRODUCT_COLUMNS],
        )
        merged = _left_join(
            merged,
            products,
            "product_id",
            "Join products",
            steps,
            "Left join on product_id.",
        )
    else:
        _left_join(merged, None, "product_id", "Join products", steps, "")
        merged["product_category_name"] = pd.Series(pd.NA, index=merged.index, dtype="string")

    if "product_category_name" not in merged.columns:
        merged["product_category_name"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged["product_category_name"] = clean_category_slug(merged["product_category_name"]).fillna("unknown")

    translation = frames.get("category_translation")
    if translation is not None:
        translation = _take(
            translation,
            ["product_category_name", "product_category_name_english"],
        )
    merged = _left_join(
        merged,
        translation,
        "product_category_name",
        "Join category translation",
        steps,
        "Left join on the Portuguese category slug.",
    )
    english = (
        merged["product_category_name_english"]
        if "product_category_name_english" in merged.columns
        else pd.Series(pd.NA, index=merged.index, dtype="string")
    )
    supplement = merged["product_category_name"].map(EXTRA_CATEGORY_TRANSLATIONS).astype("string")
    used_supplement = sorted(
        set(merged.loc[english.isna() & supplement.notna(), "product_category_name"].dropna().astype(str))
    )
    filled = english.fillna(supplement)
    untranslated_mask = filled.isna() & ~merged["product_category_name"].isin(["unknown"])
    untranslated = sorted(merged.loc[untranslated_mask, "product_category_name"].dropna().astype(str).unique())
    merged["product_category"] = filled.fillna(merged["product_category_name"]).fillna("unknown")
    merged["product_category_label"] = category_label(merged["product_category"])
    if "product_category_name_english" in merged.columns:
        merged = merged.drop(columns=["product_category_name_english"])

    if "sellers" in frames:
        sellers = _take(
            frames["sellers"],
            ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
        )
        merged = _left_join(
            merged,
            sellers,
            "seller_id",
            "Join sellers",
            steps,
            "Left join on seller_id. An order can contain items from more than one seller.",
        )
    else:
        _left_join(merged, None, "seller_id", "Join sellers", steps, "")

    merged = _left_join(
        merged,
        payments,
        "order_id",
        "Join payments",
        steps,
        "Left join on order_id after payments were summed to one row per order.",
    )
    merged = _left_join(
        merged,
        reviews,
        "order_id",
        "Join reviews",
        steps,
        "Left join on order_id after reviews were averaged to one score per order.",
    )

    if locations is not None and not locations.empty:
        customer_locations = locations.rename(
            columns={
                "zip_prefix": "customer_zip_code_prefix",
                "latitude": "customer_latitude",
                "longitude": "customer_longitude",
            }
        )
        seller_locations = locations.rename(
            columns={
                "zip_prefix": "seller_zip_code_prefix",
                "latitude": "seller_latitude",
                "longitude": "seller_longitude",
            }
        )
        if "customer_zip_code_prefix" in merged.columns:
            merged = _left_join(
                merged,
                customer_locations,
                "customer_zip_code_prefix",
                "Join customer location",
                steps,
                "Left join on the customer zip prefix. The coordinate is the average of that prefix, not a street address.",
            )
        else:
            _left_join(merged, None, "customer_zip_code_prefix", "Join customer location", steps, "")
        if "seller_zip_code_prefix" in merged.columns:
            merged = _left_join(
                merged,
                seller_locations,
                "seller_zip_code_prefix",
                "Join seller location",
                steps,
                "Left join on the seller zip prefix, using the same averaged coordinates.",
            )
        else:
            _left_join(merged, None, "seller_zip_code_prefix", "Join seller location", steps, "")
    else:
        _left_join(merged, None, "customer_zip_code_prefix", "Join customer location", steps, "")
        _left_join(merged, None, "seller_zip_code_prefix", "Join seller location", steps, "")

    merged = _add_revenue_and_timing(merged)
    sort_columns = [
        column
        for column in ("order_purchase_timestamp", "order_id", "order_item_id")
        if column in merged.columns
    ]
    merged = merged.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    merged = _order_columns(merged)

    notes = _build_notes(merged, used_supplement, untranslated)
    merge_report = pd.DataFrame(steps)
    return merged, merge_report, notes, untranslated


def _add_revenue_and_timing(frame: pd.DataFrame) -> pd.DataFrame:
    """Add line revenue and the timing columns used by the charts."""
    df = frame.copy()
    if "freight_value" not in df.columns:
        df["freight_value"] = 0.0
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if df["price"].isna().any():
        raise ValueError("Some items still have no price after cleaning, so revenue is undefined.")

    # Round each money column to the nearest cent, then add the cents.
    # That keeps line revenue equal to price plus freight at currency precision.
    price_cents = np.round(df["price"].to_numpy(dtype=float) * 100).astype(np.int64)
    freight_cents = np.round(df["freight_value"].to_numpy(dtype=float) * 100).astype(np.int64)
    df["price"] = price_cents / 100
    df["freight_value"] = freight_cents / 100
    df["line_revenue"] = (price_cents + freight_cents) / 100

    status = df["order_status"] if "order_status" in df.columns else pd.Series("unknown", index=df.index)
    status = status.astype("string").fillna("unknown")
    df["order_status"] = status
    df["counts_as_revenue"] = ~status.isin(REVENUE_EXCLUDED_STATUSES)

    if "order_purchase_timestamp" in df.columns:
        purchased = df["order_purchase_timestamp"]
        df["purchase_year"] = purchased.dt.year.astype("Int16")
        df["purchase_month"] = purchased.dt.month.astype("Int8")
        df["purchase_month_start"] = purchased.dt.to_period("M").dt.to_timestamp()
    else:
        df["purchase_year"] = pd.Series(pd.NA, index=df.index, dtype="Int16")
        df["purchase_month"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        df["purchase_month_start"] = pd.NaT

    if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(df.columns):
        seconds = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds()
        df["delivery_days"] = (seconds / 86400).round(2)
    else:
        df["delivery_days"] = np.nan

    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(df.columns):
        delay = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.total_seconds()
        df["delivery_delay_days"] = (delay / 86400).round(2)
    else:
        df["delivery_delay_days"] = np.nan

    if {"customer_state", "seller_state"}.issubset(df.columns):
        both = df["customer_state"].notna() & df["seller_state"].notna()
        same = pd.Series(pd.NA, index=df.index, dtype="boolean")
        comparison = df.loc[both, "customer_state"].eq(df.loc[both, "seller_state"])
        same.loc[both] = comparison.astype("boolean")
        df["same_state"] = same
    else:
        df["same_state"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    return df


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "order_id",
        "order_item_id",
        "order_status",
        "counts_as_revenue",
        "order_purchase_timestamp",
        "purchase_year",
        "purchase_month",
        "purchase_month_start",
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
        "customer_zip_code_prefix",
        "customer_latitude",
        "customer_longitude",
        "product_id",
        "product_category",
        "product_category_label",
        "product_category_name",
        "seller_id",
        "seller_city",
        "seller_state",
        "seller_zip_code_prefix",
        "seller_latitude",
        "seller_longitude",
        "price",
        "freight_value",
        "line_revenue",
        "payment_type",
        "payment_installments",
        "payment_count",
        "order_payment_total",
        "review_score",
        "review_count",
        "delivery_days",
        "delivery_delay_days",
        "same_state",
        "shipping_limit_date",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        *NUMERIC_PRODUCT_COLUMNS,
    ]
    front = [column for column in preferred if column in frame.columns]
    rest = [column for column in frame.columns if column not in front]
    return frame[front + rest]


def _build_notes(
    frame: pd.DataFrame,
    used_supplement: list[str],
    untranslated: list[str],
) -> list[str]:
    notes = [
        "customer_id changes on every order. customer_unique_id is the person, and customer counts use that column.",
        "Line revenue is the item price plus freight, rounded to the cent. Add line_revenue. Do not add order_payment_total across item rows, because that payment belongs to the whole order.",
        "Canceled, unavailable, and unknown statuses stay in the merged file but counts_as_revenue is False for them.",
    ]
    if used_supplement:
        notes.append(
            "The official translation file has no English name for: "
            + ", ".join(used_supplement)
            + ". A small built-in map filled those names."
        )
    if untranslated:
        notes.append(
            "These categories still have no English name and kept the original slug: "
            + ", ".join(untranslated)
            + "."
        )
    if "delivery_days" in frame.columns:
        negative = int((pd.to_numeric(frame["delivery_days"], errors="coerce") < 0).sum())
        if negative:
            notes.append(
                f"{negative:,} items have a delivery date before the purchase date. Those rows are kept, but they are left out of the average delivery time."
            )
    return notes


def overview_from_results(results: list[TableCleanResult]) -> pd.DataFrame:
    """Build the cleaning summary table shown in the report and the dashboard."""
    rows = []
    for result in results:
        spec = TABLES.get(result.name, {})
        rows.append(
            {
                "table": result.name,
                "label": spec.get("label", result.name),
                "rows_loaded": result.rows_raw,
                "columns": result.columns,
                "exact_duplicates_removed": result.exact_duplicates_removed,
                "key_duplicates_removed": result.key_duplicates_removed,
                "rows_removed_invalid": result.rows_removed_invalid,
                "rows_after_cleaning": result.rows_clean,
            }
        )
    return pd.DataFrame(rows)
