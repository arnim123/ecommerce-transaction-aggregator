"""Experiment 2: one-hot encode the categorical columns.

One-hot encoding turns a category into several 0/1 columns. A payment type of
"credit_card" becomes payment_type_credit_card = 1 and payment_type_boleto = 0.
The encoded table has the same number of rows as the cleaned transactions.

City names and ids are not encoded. A column per city, or a column per
customer, would be huge and would mostly be zeros.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLUMNS, ENCODED_ID_COLUMNS, ENCODED_NUMERIC_COLUMNS

_NOT_ONE_HOT = (
    ("customer_city", "Thousands of city names. One column per city would be huge and almost entirely zeros. Customer state is encoded instead."),
    ("seller_city", "Same reason as customer city. Seller state is encoded instead."),
    ("customer_unique_id", "This identifies a person. A model should not memorize one column per person."),
    ("customer_id", "Olist creates a new customer id for every order, so it cannot be reused for a new order."),
    ("order_id", "An identifier. It is kept in the encoded file only so a row can be traced back, and it should be dropped before training."),
    ("order_item_id", "An identifier for the unit inside the order, not a category."),
    ("product_id", "Tens of thousands of products. The category is the useful grouping."),
    ("seller_id", "An identifier. Seller state is the categorical summary."),
    ("product_category_label", "Readable version of product_category. Encoding both would duplicate the same information."),
    ("product_category_name", "Portuguese category name. The English slug product_category is encoded instead."),
    ("price", "Already a number. If the target is line revenue, remove price and freight first, because they add up to the target."),
    ("freight_value", "Already a number, and it is part of line revenue."),
    ("line_revenue", "This is a possible target. Keep it out of the feature set when the model is trying to predict it."),
    ("order_payment_total", "This is the whole order, repeated on every item. Summing it across items double-counts the payment."),
    ("review_score", "A number collected after delivery. Do not use it to predict something that is only known at purchase time."),
    ("purchase_year", "Already a number, so one-hot encoding is unnecessary."),
    ("purchase_month", "Already a month number from 1 to 12."),
)


def not_encoded_table() -> pd.DataFrame:
    """Explain columns that are intentionally not one-hot encoded."""
    return pd.DataFrame(_NOT_ONE_HOT, columns=["column", "why_it_is_not_one_hot"])


def one_hot_encode(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return the encoded table, a summary of new columns, and notes.

    Missing categories are filled with "unknown" first. After that, every row
    has exactly one 1 in each group of dummy columns.
    """
    notes: list[str] = [
        "One-hot encoding replaces each selected category with 0/1 columns, one column per value.",
        "Year, month, prices, and scores stay as numbers because they already have a numeric meaning.",
    ]
    present = [column for column in CATEGORICAL_COLUMNS if column in frame.columns]
    missing = [column for column in CATEGORICAL_COLUMNS if column not in frame.columns]
    if missing:
        notes.append("These categorical columns were not in the table, so they were not encoded: " + ", ".join(missing) + ".")

    work = frame.reset_index(drop=True)
    prepared = pd.DataFrame(index=work.index)
    summary_rows: list[dict[str, object]] = []
    for column in present:
        labels = work[column].astype("string").str.strip()
        labels = labels.mask(labels.str.lower().isin(["", "nan", "none", "null", "<na>"]))
        labels = labels.fillna("unknown")
        safe = labels.map(_safe_token).astype("string")
        if int(safe.nunique(dropna=True)) < int(labels.nunique(dropna=True)):
            notes.append(f"{column}: some labels were combined so the new column names stay simple.")
        prepared[column] = safe
        examples = ", ".join(sorted(safe.dropna().unique().astype(str))[:8])
        summary_rows.append(
            {
                "original_column": column,
                "unique_values": int(safe.nunique(dropna=True)),
                "new_columns": int(safe.nunique(dropna=True)),
                "example_values": examples,
            }
        )

    if present:
        dummies = pd.get_dummies(prepared, columns=present, dtype=np.uint8)
    else:
        dummies = pd.DataFrame(index=work.index)

    id_columns = [column for column in ENCODED_ID_COLUMNS if column in work.columns]
    numeric_columns = [column for column in ENCODED_NUMERIC_COLUMNS if column in work.columns]
    numeric = work[id_columns + numeric_columns].copy()
    for column in numeric.columns:
        if column in id_columns:
            continue
        if pd.api.types.is_bool_dtype(numeric[column]) or str(numeric[column].dtype) == "boolean":
            numeric[column] = numeric[column].astype("float")

    encoded = pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    encoded.columns = _unique_names([str(column) for column in encoded.columns])

    for row in summary_rows:
        prefix = str(row["original_column"]) + "_"
        row["new_columns"] = sum(column.startswith(prefix) for column in encoded.columns)
    summary = pd.DataFrame(summary_rows)
    notes.append(
        f"The encoded table has {encoded.shape[0]:,} rows and {encoded.shape[1]:,} columns, "
        f"of which {dummies.shape[1]:,} are one-hot flags."
    )
    return encoded, summary, notes


def _safe_token(value: object) -> str:
    """Make a category safe to use as part of a column name."""
    if value is None or pd.isna(value):
        return "unknown"
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return token or "unknown"


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count}")
    return unique
