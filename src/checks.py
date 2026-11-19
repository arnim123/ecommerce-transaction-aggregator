"""Quality checks for the merged table and the one-hot encoded table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLUMNS, REVENUE_EXCLUDED_STATUSES


def sanity_checks(transactions: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
    """Return a pass/fail table a beginner can read after the pipeline runs.

    These checks protect the definition of revenue and the definition of
    one-hot encoding. They do not check that the business totals match a
    published Olist figure, because a small cleaning change would make that
    brittle.
    """
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "check": name,
                "status": "Pass" if passed else "Fail",
                "detail": detail,
            }
        )

    add(
        "Merged table has transactions",
        len(transactions) > 0,
        f"{len(transactions):,} rows",
    )

    if {"order_id", "order_item_id"}.issubset(transactions.columns):
        duplicate_keys = transactions.duplicated(["order_id", "order_item_id"]).any()
        add(
            "Each order item appears once",
            not bool(duplicate_keys),
            "Key is order_id + order_item_id",
        )
    else:
        add("Each order item appears once", False, "Key columns are missing.")

    money = {"price", "freight_value", "line_revenue"}
    if money.issubset(transactions.columns) and len(transactions) > 0:
        price = pd.to_numeric(transactions["price"], errors="coerce").to_numpy(dtype=float)
        freight = pd.to_numeric(transactions["freight_value"], errors="coerce").to_numpy(dtype=float)
        revenue = pd.to_numeric(transactions["line_revenue"], errors="coerce").to_numpy(dtype=float)
        # One tenth of a cent absorbs binary floating-point dust after rounding.
        identity = np.allclose(price + freight, revenue, rtol=0, atol=0.001, equal_nan=True)
        add(
            "Line revenue equals price plus freight",
            bool(identity),
            "Checked with NumPy",
        )
        finite = np.isfinite(revenue)
        non_negative = bool(np.all(finite & (revenue >= -0.01)))
        add(
            "Line revenue is present and not negative",
            non_negative and bool(finite.all()),
            f"Minimum {np.nanmin(revenue):,.2f}" if finite.any() else "No values",
        )
    else:
        add("Line revenue equals price plus freight", False, "Money columns are missing.")

    if {"order_status", "counts_as_revenue"}.issubset(transactions.columns):
        excluded = transactions["order_status"].isin(REVENUE_EXCLUDED_STATUSES)
        flags = transactions.loc[excluded, "counts_as_revenue"].fillna(False).astype(bool)
        add(
            "Canceled and unavailable orders are excluded from revenue",
            not bool(flags.any()),
            f"{int(excluded.sum()):,} rows have an excluded status",
        )

    if len(encoded) != len(transactions):
        add(
            "Encoded table keeps every transaction",
            False,
            f"Encoded rows {len(encoded):,}, transaction rows {len(transactions):,}",
        )
    else:
        add(
            "Encoded table keeps every transaction",
            True,
            f"{len(encoded):,} rows",
        )

    for column in CATEGORICAL_COLUMNS:
        if column not in transactions.columns:
            continue
        dummy_columns = [name for name in encoded.columns if name.startswith(f"{column}_")]
        if not dummy_columns or encoded.empty:
            add(
                f"One-hot columns exist for {column}",
                False,
                "No dummy columns were created.",
            )
            continue
        totals = encoded[dummy_columns].to_numpy(dtype=np.int16).sum(axis=1)
        one_each = bool(np.all(totals == 1))
        add(
            f"One-hot {column} has one active column per row",
            one_each,
            f"{len(dummy_columns)} columns",
        )

    return pd.DataFrame(rows)
