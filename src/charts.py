"""Interactive Plotly charts for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis import AnalysisResult
from src.config import COLOR_CLAY, COLOR_GRID, COLOR_INK, COLOR_MUTED, COLOR_TEAL
from src.formatters import humanize_token


def monthly_revenue(analysis: AnalysisResult) -> go.Figure:
    """Revenue and order counts by month. The order line uses a second axis."""
    frame = analysis.monthly
    if frame.empty:
        return _empty("No purchases fall in this filter.")
    figure = go.Figure()
    figure.add_bar(
        x=frame["purchase_month_start"],
        y=frame["revenue"],
        name="Item revenue",
        marker_color=COLOR_TEAL,
        hovertemplate="%{x|%b %Y}<br>Revenue R$ %{y:,.2f}<extra></extra>",
    )
    figure.add_scatter(
        x=frame["purchase_month_start"],
        y=frame["orders"],
        name="Orders",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=COLOR_CLAY, width=2),
        hovertemplate="%{x|%b %Y}<br>Orders %{y:,}<extra></extra>",
    )
    _style(figure, "Monthly revenue and orders", height=460)
    figure.update_layout(
        yaxis=dict(title="Item revenue (BRL)", tickprefix="R$ ", tickformat=","),
        yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


def category_revenue(analysis: AnalysisResult, limit: int = 12) -> go.Figure:
    """Top categories by item revenue."""
    frame = analysis.by_category.head(limit).sort_values("revenue", ascending=True)
    if frame.empty:
        return _empty("No categories fall in this filter.")
    figure = px.bar(
        frame,
        x="revenue",
        y="category",
        orientation="h",
        labels={"revenue": "Item revenue (BRL)", "category": ""},
    )
    figure.update_traces(
        marker_color=COLOR_TEAL,
        hovertemplate="%{y}<br>R$ %{x:,.2f}<extra></extra>",
    )
    _style(figure, "Top categories by item revenue", height=max(420, 36 * len(frame) + 120), value_axis="x")
    figure.update_xaxes(tickprefix="R$ ", tickformat=",")
    return figure


def state_revenue(analysis: AnalysisResult, limit: int = 12) -> go.Figure:
    """Top customer states by item revenue."""
    frame = analysis.by_state.head(limit).sort_values("revenue", ascending=True)
    if frame.empty:
        return _empty("No states fall in this filter.")
    figure = px.bar(
        frame,
        x="revenue",
        y="state",
        orientation="h",
        labels={"revenue": "Item revenue (BRL)", "state": ""},
    )
    figure.update_traces(
        marker_color=COLOR_TEAL,
        hovertemplate="%{y}<br>R$ %{x:,.2f}<extra></extra>",
    )
    _style(figure, "Top customer states by item revenue", height=max(420, 36 * len(frame) + 120), value_axis="x")
    figure.update_xaxes(tickprefix="R$ ", tickformat=",")
    return figure


def payment_mix(analysis: AnalysisResult) -> go.Figure:
    """Item revenue by payment type."""
    frame = analysis.by_payment.copy()
    if frame.empty:
        return _empty("No payment types fall in this filter.")
    frame["label"] = frame["payment_type"].map(humanize_token)
    frame = frame.sort_values("revenue", ascending=False)
    figure = px.bar(frame, x="label", y="revenue", labels={"label": "Payment type", "revenue": "Item revenue (BRL)"})
    figure.update_traces(marker_color=COLOR_CLAY, hovertemplate="%{x}<br>R$ %{y:,.2f}<extra></extra>")
    _style(figure, "Item revenue by payment type")
    figure.update_yaxes(tickprefix="R$ ", tickformat=",")
    return figure


def status_mix(analysis: AnalysisResult) -> go.Figure:
    """Units in each order status."""
    frame = analysis.by_status.copy()
    if frame.empty:
        return _empty("No statuses fall in this filter.")
    frame["label"] = frame["order_status"].map(humanize_token)
    frame = frame.sort_values("units", ascending=False)
    figure = px.bar(frame, x="label", y="units", labels={"label": "Order status", "units": "Units"})
    figure.update_traces(marker_color=COLOR_TEAL, hovertemplate="%{x}<br>%{y:,} units<extra></extra>")
    _style(figure, "Units by order status")
    return figure


def review_revenue(analysis: AnalysisResult) -> go.Figure:
    """Average order revenue at each review score."""
    frame = analysis.by_review
    if frame.empty:
        return _empty("No review scores fall in this filter.")
    figure = px.bar(
        frame,
        x="review_score",
        y="average_order_revenue",
        labels={"review_score": "Review score", "average_order_revenue": "Average order revenue (BRL)"},
    )
    figure.update_traces(marker_color=COLOR_TEAL, hovertemplate="Score %{x}<br>R$ %{y:,.2f}<extra></extra>")
    _style(figure, "Average order revenue by review score")
    figure.update_yaxes(tickprefix="R$ ", tickformat=",")
    return figure


def revenue_histogram(analysis: AnalysisResult) -> go.Figure:
    """Histogram of item revenue up to the 99th percentile."""
    frame = analysis.histogram
    if frame.empty:
        return _empty("No revenue values fall in this filter.")
    figure = px.bar(frame, x="bin_label", y="count", labels={"bin_label": "Item revenue", "count": "Units"})
    figure.update_traces(marker_color=COLOR_TEAL, hovertemplate="%{x}<br>%{y:,} units<extra></extra>")
    _style(figure, "Distribution of item revenue")
    figure.update_xaxes(tickangle=-40)
    return figure


def _style(figure: go.Figure, title: str, height: int = 440, value_axis: str = "y") -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color=COLOR_INK)),
        height=height,
        margin=dict(l=8, r=8, t=72, b=8),
        font=dict(color=COLOR_INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if value_axis == "x":
        figure.update_xaxes(gridcolor=COLOR_GRID, showgrid=True, zeroline=False)
        figure.update_yaxes(showgrid=False, zeroline=False)
    else:
        figure.update_yaxes(gridcolor=COLOR_GRID, showgrid=True, zeroline=False)
        figure.update_xaxes(showgrid=False, zeroline=False)
    return figure


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font=dict(color=COLOR_MUTED, size=14))
    figure.update_layout(
        template="plotly_white",
        height=320,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure
