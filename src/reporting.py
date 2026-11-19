"""Plain-text reports printed by the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.formatters import describe_correlation, format_brl, format_int, format_pct
from src.pipeline import PipelineResult


def display_path(path: Path) -> str:
    """Show a path relative to the project when that is possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def plain_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a small dataframe as aligned text."""
    if frame.empty:
        return "(no rows)"
    shown = frame.head(max_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:,.2f}")
    with pd.option_context("display.max_columns", 24, "display.width", 140, "display.max_colwidth", 60):
        text = shown.to_string(index=False)
    remaining = len(frame) - len(shown)
    if remaining > 0:
        text += f"\n... {remaining:,} more rows"
    return text


def experiment_one_report(result: PipelineResult) -> str:
    """Build the Experiment 1 lab report."""
    headline = result.headline
    sample_columns = [
        column
        for column in ("order_id", "order_item_id", "order_status", "price", "freight_value", "line_revenue")
        if column in result.transactions.columns
    ]
    lines = [
        "Experiment 1 — Preprocessing and revenue",
        "=" * 44,
        "Each CSV is cleaned, related tables are merged onto one row per item,",
        "and line revenue is calculated as price + freight.",
        "",
        "Cleaning summary",
        "-" * 16,
        plain_table(result.table_overview),
        "",
        "Merge steps",
        "-" * 11,
        plain_table(_percent_merge(result.merge_report)),
        "",
        "Revenue by order status",
        "-" * 23,
        plain_table(result.status_totals),
        "",
        "Headline numbers",
        "-" * 16,
        f"Item rows:                         {format_int(headline['item_rows'])}",
        f"Orders:                            {format_int(headline['orders'])}",
        f"Item revenue, all statuses:        {format_brl(headline['item_revenue_all'])}",
        f"Item revenue, eligible statuses:   {format_brl(headline['item_revenue_eligible'])}",
        f"Item revenue, delivered only:      {format_brl(headline['item_revenue_delivered'])}",
        f"Payments, once per order:          {format_brl(headline['payment_revenue_all'])}",
        f"Purchase dates:                    {_date_span(result)}",
        "",
        "Sample rows (price + freight = line revenue)",
        "-" * 44,
        plain_table(result.transactions[sample_columns].head(8)),
        "",
        "Notes",
        "-" * 5,
        *[f"- {note}" for note in result.notes],
        "",
        "Quality checks",
        "-" * 14,
        plain_table(result.sanity_checks),
        "",
    ]
    return "\n".join(lines)


def experiment_two_report(result: PipelineResult) -> str:
    """Build the Experiment 2 lab report."""
    stats = result.default_analysis.numpy_stats
    correlations = result.default_analysis.correlations
    category = result.default_analysis.by_category.copy()
    if "revenue_share" in category.columns:
        category["revenue_share"] = category["revenue_share"].map(
            lambda value: "" if pd.isna(value) else f"{float(value) * 100:.1f}%"
        )
    lines = [
        "Experiment 2 — One-hot encoding and revenue analysis",
        "=" * 52,
        "Categorical columns become 0/1 flags. Revenue statistics are",
        "calculated with NumPy, category tables with Pandas, and the",
        "charts with Matplotlib.",
        "",
        "The statistics below exclude canceled and unavailable orders.",
        "",
        "One-hot encoding",
        "-" * 17,
        plain_table(result.encoding_summary),
        "",
        "NumPy revenue statistics",
        "-" * 24,
        f"Items:              {format_int(stats['count'])}",
        f"Total:              {format_brl(stats['total'])}",
        f"Mean:               {format_brl(stats['mean'])}",
        f"Median:             {format_brl(stats['median'])}",
        f"Standard deviation: {format_brl(stats['std'])}",
        f"Minimum:            {format_brl(stats['min'])}",
        f"25th percentile:    {format_brl(stats['p25'])}",
        f"75th percentile:    {format_brl(stats['p75'])}",
        f"95th percentile:    {format_brl(stats['p95'])}",
        f"Maximum:            {format_brl(stats['max'])}",
        "",
        "The standard deviation is NumPy's sample standard deviation (divided by N - 1).",
        "",
        "Relationships",
        "-" * 13,
        "- " + describe_correlation(correlations["review_vs_order_revenue"], "review score", "order revenue"),
        "- " + describe_correlation(correlations["freight_vs_price"], "freight", "item price"),
        "- " + describe_correlation(correlations["delivery_days_vs_review"], "delivery time", "review score"),
        "",
        "Top categories",
        "-" * 14,
        plain_table(
            category.head(15)[["category", "units", "orders", "revenue", "revenue_share"]]
            if "revenue_share" in category.columns
            else category.head(15)
        ),
        "",
        "Top customer states",
        "-" * 19,
        plain_table(result.default_analysis.by_state.head(10)[["state", "orders", "revenue", "avg_order_revenue"]]),
        "",
    ]
    return "\n".join(lines)


def _percent_merge(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    if "match_rate" in shown.columns:
        shown["match_rate"] = shown["match_rate"].map(lambda value: format_pct(value) if pd.notna(value) else "")
    return shown


def _date_span(result: PipelineResult) -> str:
    start = result.headline.get("date_min")
    end = result.headline.get("date_max")
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return "n/a"
    return f"{pd.Timestamp(start):%Y-%m-%d} to {pd.Timestamp(end):%Y-%m-%d}"
