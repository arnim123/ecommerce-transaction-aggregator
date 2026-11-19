"""Run Experiment 1 and Experiment 2 as one pipeline.

The command-line scripts and the Streamlit app both call run_from_directory
or build_from_tables, so a number in the lab report is the same number as
the number on the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.analysis import AnalysisResult, analyze_transactions, revenue_by_status
from src.checks import sanity_checks
from src.config import DATASET_DIR, TABLES
from src.data_loader import load_tables_from_directory
from src.encoding import not_encoded_table, one_hot_encode
from src.formatters import format_brl
from src.preprocessing import (
    CLEANERS,
    TableCleanResult,
    aggregate_geolocation,
    aggregate_payments,
    aggregate_reviews,
    assemble_transactions,
    overview_from_results,
)


@dataclass
class PipelineResult:
    """Everything the report and the dashboard need after a run."""

    previews: dict[str, pd.DataFrame]
    table_overview: pd.DataFrame
    missing_report: pd.DataFrame
    actions: dict[str, list[str]]
    merge_report: pd.DataFrame
    transactions: pd.DataFrame
    encoded: pd.DataFrame
    encoding_summary: pd.DataFrame
    not_encoded: pd.DataFrame
    status_totals: pd.DataFrame
    default_analysis: AnalysisResult
    sanity_checks: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    untranslated_categories: list[str] = field(default_factory=list)
    headline: dict[str, object] = field(default_factory=dict)


def run_from_directory(
    directory: Path | str = DATASET_DIR,
    progress: Callable[[str], None] | None = None,
    require_all: bool = True,
) -> PipelineResult:
    """Load a dataset folder and run both experiments."""
    folder = Path(directory)
    _say(progress, f"Loading CSV files from {folder}")
    tables, warnings = load_tables_from_directory(folder, require_all=require_all)
    result = build_from_tables(tables, progress=progress)
    result.warnings = warnings + result.warnings
    return result


def build_from_tables(
    tables: dict[str, pd.DataFrame],
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Clean, merge, encode, and summarize tables that are already in memory.

    The input dictionary is emptied as each file is cleaned so the large
    location file is not held twice.
    """
    loaded_keys = [key for key in tables if key in CLEANERS]
    previews: dict[str, pd.DataFrame] = {}
    results: list[TableCleanResult] = []
    frames: dict[str, pd.DataFrame] = {}

    for key in loaded_keys:
        raw = tables.pop(key)
        label = str(TABLES[key]["label"])
        _say(progress, f"Cleaning {label.lower()} ({len(raw):,} rows)")
        previews[key] = raw.head(100).copy()
        cleaned = CLEANERS[key](raw)
        frames[key] = cleaned.frame
        results.append(cleaned)
        del raw

    if "orders" not in frames or "order_items" not in frames:
        raise ValueError("Orders and order items are required.")
    if frames["orders"].empty or frames["order_items"].empty:
        raise ValueError("Orders or order items were empty after cleaning.")

    _say(progress, "Combining payments, reviews, and zip-code coordinates")
    payments = _aggregate_optional(frames, results, "payments", aggregate_payments)
    reviews = _aggregate_optional(frames, results, "reviews", aggregate_reviews)
    locations = _aggregate_optional(frames, results, "geolocation", aggregate_geolocation)

    _say(progress, "Merging the tables and calculating line revenue")
    transactions, merge_report, notes, untranslated = assemble_transactions(
        frames, payments, reviews, locations
    )
    for result in results:
        result.frame = pd.DataFrame()
    frames.clear()

    overview = overview_from_results(results)
    actions = {result.name: list(result.actions) for result in results}
    _append_aggregation_notes(actions, overview, payments, reviews, locations)
    missing_report = _missing_report(results)

    for key, spec in TABLES.items():
        if not spec["required"] and key not in previews:
            notes.append(f"The {spec['label']} table was not loaded, so its columns are absent.")

    _say(progress, "One-hot encoding status, payment type, state, and category")
    encoded, encoding_summary, encoding_notes = one_hot_encode(transactions)
    notes.extend(encoding_notes)

    eligible = transactions.loc[transactions["counts_as_revenue"]].copy()
    _say(progress, "Summarizing revenue")
    default_analysis = analyze_transactions(eligible)
    status_totals = revenue_by_status(transactions)
    checks = sanity_checks(transactions, encoded)
    headline = _headline(transactions, default_analysis)
    notes.extend(_revenue_notes(headline))

    return PipelineResult(
        previews=previews,
        table_overview=overview,
        missing_report=missing_report,
        actions=actions,
        merge_report=merge_report,
        transactions=transactions,
        encoded=encoded,
        encoding_summary=encoding_summary,
        not_encoded=not_encoded_table(),
        status_totals=status_totals,
        default_analysis=default_analysis,
        sanity_checks=checks,
        notes=notes,
        untranslated_categories=untranslated,
        headline=headline,
    )


def _aggregate_optional(frames, results, key, aggregator):
    if key not in frames:
        return None
    aggregated = aggregator(frames.pop(key))
    for result in results:
        if result.name == key:
            result.frame = pd.DataFrame()
    return aggregated


def _append_aggregation_notes(actions, overview, payments, reviews, locations) -> None:
    if payments is not None and "payments" in actions:
        source_rows = _clean_rows(overview, "payments")
        actions["payments"].append(
            f"Combined {source_rows:,} payment rows into {len(payments):,} orders. "
            "The payment type kept for an order is the method with the largest amount."
        )
    if reviews is not None and "reviews" in actions:
        source_rows = _clean_rows(overview, "reviews")
        actions["reviews"].append(
            f"Averaged {source_rows:,} reviews down to {len(reviews):,} orders."
        )
    if locations is not None and "geolocation" in actions:
        actions["geolocation"].append(
            f"Averaged coordinates down to {len(locations):,} zip prefixes. "
            "Joining the raw location file would copy each order many times and inflate revenue."
        )


def _clean_rows(overview: pd.DataFrame, table: str) -> int:
    matched = overview.loc[overview["table"] == table, "rows_after_cleaning"]
    if matched.empty:
        return 0
    return int(matched.iloc[0])


def _missing_report(results: list[TableCleanResult]) -> pd.DataFrame:
    pieces = [pd.DataFrame(result.missing_rows) for result in results if result.missing_rows]
    columns = ["table", "column", "missing_before", "missing_after", "treatment"]
    if not pieces:
        return pd.DataFrame(columns=columns)
    report = pd.concat(pieces, ignore_index=True)
    return report.sort_values(["table", "missing_before"], ascending=[True, False]).reset_index(drop=True)


def _headline(transactions: pd.DataFrame, analysis: AnalysisResult) -> dict[str, object]:
    revenue = pd.to_numeric(transactions["line_revenue"], errors="coerce").to_numpy(dtype=float)
    delivered = transactions["order_status"].eq("delivered").to_numpy()
    delivered_revenue = float(np.nansum(revenue[delivered])) if delivered.any() else 0.0
    if "order_payment_total" in transactions.columns:
        per_order = transactions.drop_duplicates("order_id")
        payment_values = pd.to_numeric(per_order["order_payment_total"], errors="coerce").to_numpy(dtype=float)
        payment_all = float(np.nansum(payment_values))
    else:
        payment_all = float("nan")
    purchased = transactions["order_purchase_timestamp"] if "order_purchase_timestamp" in transactions.columns else None
    return {
        "item_rows": int(len(transactions)),
        "orders": int(transactions["order_id"].nunique()),
        "item_revenue_all": float(np.nansum(revenue)),
        "item_revenue_eligible": float(analysis.kpis["item_revenue"]),
        "item_revenue_delivered": delivered_revenue,
        "payment_revenue_all": payment_all,
        "payment_revenue_eligible": analysis.kpis["payment_revenue"],
        "date_min": None if purchased is None or purchased.isna().all() else purchased.min(),
        "date_max": None if purchased is None or purchased.isna().all() else purchased.max(),
    }


def _revenue_notes(headline: dict[str, object]) -> list[str]:
    notes = [
        (
            f"Item revenue across every status is {format_brl(headline['item_revenue_all'])}. "
            f"Eligible item revenue, which excludes canceled and unavailable orders, is "
            f"{format_brl(headline['item_revenue_eligible'])}. "
            f"Delivered orders alone are {format_brl(headline['item_revenue_delivered'])}."
        )
    ]
    try:
        payment = float(headline["payment_revenue_all"])
    except (TypeError, ValueError):
        payment = float("nan")
    if math.isfinite(payment):
        notes.append(
            f"Payments added once per order come to {format_brl(payment)}. "
            "This differs from item revenue when a payment is missing, a voucher changes the amount, "
            "or interest is included. Adding order_payment_total on every item row would count a "
            "multi-item order more than once."
        )
    return notes


def _say(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
