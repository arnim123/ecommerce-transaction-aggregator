"""Matplotlib figures for Experiment 2.

The dashboard draws interactive Plotly charts from the same summaries.
These figures are the static files saved in outputs/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.analysis import AnalysisResult
from src.config import COLOR_CLAY, COLOR_GRID, COLOR_INK, COLOR_MUTED, COLOR_TEAL
from src.formatters import humanize_token

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titlelocation": "left",
        "axes.edgecolor": "#D6D3D1",
        "text.color": COLOR_INK,
        "axes.labelcolor": COLOR_INK,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
    }
)


def save_all_figures(analysis: AnalysisResult, directory: Path) -> list[Path]:
    """Draw every Experiment 2 chart and return the file paths."""
    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "monthly_revenue.png": monthly_revenue,
        "revenue_by_category.png": revenue_by_category,
        "revenue_by_state.png": revenue_by_state,
        "payment_type_share.png": payment_mix,
        "order_status_counts.png": status_mix,
        "review_score_vs_revenue.png": review_revenue,
        "revenue_distribution.png": revenue_histogram,
    }
    saved: list[Path] = []
    for filename, builder in figures.items():
        path = directory / filename
        figure = builder(analysis)
        figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        saved.append(path)
    return saved


def monthly_revenue(analysis: AnalysisResult) -> plt.Figure:
    """Line chart of item revenue by purchase month."""
    frame = analysis.monthly
    title = "Monthly item revenue"
    partial = analysis.kpis.get("partial_month_label")
    if analysis.kpis.get("last_month_is_partial") and partial:
        title = f"Monthly item revenue ({partial} is incomplete)"
    fig, ax = _axes(title, "Purchase month", "Item revenue (BRL)")
    if frame.empty:
        return _empty(ax, fig, "No purchases to plot.")
    ax.plot(frame["purchase_month_start"], frame["revenue"], color=COLOR_TEAL, marker="o", linewidth=2)
    ax.yaxis.set_major_formatter(_money_formatter())
    fig.autofmt_xdate()
    return fig


def revenue_by_category(analysis: AnalysisResult) -> plt.Figure:
    """Horizontal bar chart of the 12 categories with the most revenue."""
    frame = analysis.by_category.head(12).iloc[::-1]
    fig, ax = _axes("Item revenue by category", "Item revenue (BRL)", "", wide=True)
    if frame.empty:
        return _empty(ax, fig, "No categories to plot.")
    ax.barh(frame["category"], frame["revenue"], color=COLOR_TEAL)
    ax.xaxis.set_major_formatter(_money_formatter())
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COLOR_GRID)
    return fig


def revenue_by_state(analysis: AnalysisResult) -> plt.Figure:
    """Horizontal bar chart of the 12 customer states with the most revenue."""
    frame = analysis.by_state.head(12).iloc[::-1]
    fig, ax = _axes("Item revenue by customer state", "Item revenue (BRL)", "", wide=True)
    if frame.empty:
        return _empty(ax, fig, "No states to plot.")
    labels = frame["state"] if "state" in frame.columns else frame["customer_state"]
    ax.barh(labels, frame["revenue"], color=COLOR_TEAL)
    ax.xaxis.set_major_formatter(_money_formatter())
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COLOR_GRID)
    return fig


def payment_mix(analysis: AnalysisResult) -> plt.Figure:
    """Bar chart of item revenue by payment type."""
    frame = analysis.by_payment.sort_values("revenue", ascending=False)
    fig, ax = _axes("Item revenue by payment type", "Payment type", "Item revenue (BRL)")
    if frame.empty:
        return _empty(ax, fig, "No payments to plot.")
    labels = [humanize_token(value) for value in frame["payment_type"]]
    ax.bar(labels, frame["revenue"], color=COLOR_CLAY)
    ax.yaxis.set_major_formatter(_money_formatter())
    return fig


def status_mix(analysis: AnalysisResult) -> plt.Figure:
    """Bar chart of how many units sit in each order status in this view."""
    frame = analysis.by_status.sort_values("units", ascending=False)
    fig, ax = _axes("Units by order status", "Order status", "Units")
    if frame.empty:
        return _empty(ax, fig, "No statuses to plot.")
    labels = [humanize_token(value) for value in frame["order_status"]]
    ax.bar(labels, frame["units"], color=COLOR_TEAL)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    return fig


def review_revenue(analysis: AnalysisResult) -> plt.Figure:
    """Average order revenue for each rounded review score."""
    frame = analysis.by_review
    fig, ax = _axes("Average order revenue by review score", "Review score", "Average order revenue (BRL)")
    if frame.empty:
        return _empty(ax, fig, "No reviews to plot.")
    ax.bar(frame["review_score"].astype(int).astype(str), frame["average_order_revenue"], color=COLOR_TEAL)
    ax.yaxis.set_major_formatter(_money_formatter())
    return fig


def revenue_histogram(analysis: AnalysisResult) -> plt.Figure:
    """Distribution of item revenue up to the 99th percentile."""
    frame = analysis.histogram
    title = "Distribution of item revenue"
    fig, ax = _axes(title, "Item revenue bin", "Units")
    if frame.empty:
        return _empty(ax, fig, "No revenue values to plot.")
    ax.bar(frame["bin_label"], frame["count"], color=COLOR_TEAL)
    ax.tick_params(axis="x", labelrotation=55)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    note = "The highest 1% of item revenues are omitted so one large order does not stretch the axis."
    ax.text(0.0, -0.28, note, transform=ax.transAxes, fontsize=8, color=COLOR_MUTED)
    return fig


def _axes(title: str, xlabel: str, ylabel: str, wide: bool = False) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11, 6.2) if wide else (10.5, 5.4))
    ax.set_title(title, color=COLOR_INK, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLOR_GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax


def _empty(ax: plt.Axes, fig: plt.Figure, message: str) -> plt.Figure:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=COLOR_MUTED)
    return fig


def _money_formatter() -> mticker.FuncFormatter:
    return mticker.FuncFormatter(lambda value, _pos: f"R$ {value:,.0f}")
