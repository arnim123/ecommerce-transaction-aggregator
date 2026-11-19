"""Revenue summaries used by Experiment 2 and the dashboard.

Item rows are the unit for revenue. Order rows are the unit for reviews,
delivery time, and payments. Mixing those grains is how revenue gets
double-counted, so the helpers below keep them apart.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.formatters import state_display


@dataclass
class AnalysisResult:
    """Aggregates for one filtered view of the transactions."""

    kpis: dict[str, object]
    numpy_stats: dict[str, float]
    monthly: pd.DataFrame
    by_category: pd.DataFrame
    by_state: pd.DataFrame
    by_payment: pd.DataFrame
    by_status: pd.DataFrame
    by_review: pd.DataFrame
    top_cities: pd.DataFrame
    histogram: pd.DataFrame
    histogram_cap: float
    correlations: dict[str, float]


def analyze_transactions(frame: pd.DataFrame) -> AnalysisResult:
    """Summarize revenue, categories, and review patterns for a set of items."""
    if frame.empty or "line_revenue" not in frame.columns:
        return _empty_analysis()

    revenue = pd.to_numeric(frame["line_revenue"], errors="coerce").to_numpy(dtype=float)
    revenue = revenue[np.isfinite(revenue)]
    stats = numpy_revenue_stats(revenue)
    order_level = _order_level(frame)
    kpis = _kpis(frame, order_level, stats)
    monthly, month_info = _monthly(frame)
    kpis.update(month_info)
    histogram, histogram_cap = _histogram(revenue)
    return AnalysisResult(
        kpis=kpis,
        numpy_stats=stats,
        monthly=monthly,
        by_category=_by_category(frame),
        by_state=_by_state(frame, order_level),
        by_payment=_group_money(frame, "payment_type", "payment_type"),
        by_status=_group_money(frame, "order_status", "order_status"),
        by_review=_by_review(order_level),
        top_cities=_top_cities(frame),
        histogram=histogram,
        histogram_cap=histogram_cap,
        correlations=_correlations(frame, order_level),
    )


def numpy_revenue_stats(revenue: np.ndarray) -> dict[str, float]:
    """Summarize a revenue array with NumPy.

    The standard deviation is the sample version (divided by N - 1), which is
    the one usually taught in a first statistics class.
    """
    values = np.asarray(revenue, dtype=float)
    values = values[np.isfinite(values)]
    empty = {
        "count": 0,
        "total": 0.0,
        "mean": float("nan"),
        "median": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "p25": float("nan"),
        "p75": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }
    if values.size == 0:
        return empty
    return {
        "count": int(values.size),
        "total": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size >= 2 else float("nan"),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def revenue_by_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Revenue by status for the full merged table, including canceled orders."""
    if frame.empty or "line_revenue" not in frame.columns:
        return pd.DataFrame(columns=["order_status", "units", "orders", "item_revenue", "counts_as_revenue"])
    status = frame["order_status"].astype("string").fillna("unknown") if "order_status" in frame.columns else "unknown"
    counts = frame["counts_as_revenue"].fillna(False).astype(bool) if "counts_as_revenue" in frame.columns else False
    summary = pd.DataFrame(
        {
            "order_status": status,
            "order_id": frame["order_id"],
            "line_revenue": pd.to_numeric(frame["line_revenue"], errors="coerce"),
            "counts_as_revenue": counts,
        }
    )
    grouped = summary.groupby("order_status", as_index=False, dropna=False).agg(
        units=("line_revenue", "size"),
        orders=("order_id", "nunique"),
        item_revenue=("line_revenue", "sum"),
        counts_as_revenue=("counts_as_revenue", "max"),
    )
    return grouped.sort_values("item_revenue", ascending=False).reset_index(drop=True)


def _kpis(frame: pd.DataFrame, orders: pd.DataFrame, stats: dict[str, float]) -> dict[str, object]:
    item_revenue = float(stats["total"])
    order_count = int(orders["order_id"].nunique()) if "order_id" in orders.columns else 0
    customers = _nunique(frame, "customer_unique_id")
    if customers == 0:
        customers = _nunique(frame, "customer_id")
    payment_revenue = _payment_total(orders)
    freight = _finite_sum(frame, "freight_value")
    return {
        "item_revenue": item_revenue,
        "payment_revenue": payment_revenue,
        "orders": order_count,
        "units": int(len(frame)),
        "customers": customers,
        "sellers": _nunique(frame, "seller_id"),
        "average_order_value": (item_revenue / order_count) if order_count else float("nan"),
        "average_review": _finite_mean(orders, "review_score"),
        "freight_share": (freight / item_revenue) if item_revenue else float("nan"),
        "average_delivery_days": _average_delivery(orders),
        "early_arrival_rate": _early_rate(orders),
        "delivery_outliers": _delivery_outlier_count(orders),
    }


def _order_level(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per order. Revenue is summed. Status and review are taken once."""
    if "order_id" not in frame.columns:
        return frame.copy()
    named: dict[str, tuple[str, str]] = {
        "order_revenue": ("line_revenue", "sum"),
        "units": ("line_revenue", "size"),
    }
    for column, function in (
        ("order_status", "first"),
        ("order_purchase_timestamp", "first"),
        ("purchase_month_start", "first"),
        ("customer_unique_id", "first"),
        ("customer_id", "first"),
        ("customer_state", "first"),
        ("customer_city", "first"),
        ("payment_type", "first"),
        ("order_payment_total", "first"),
        ("review_score", "first"),
        ("delivery_days", "first"),
        ("delivery_delay_days", "first"),
        ("counts_as_revenue", "max"),
    ):
        if column in frame.columns:
            named[column] = (column, function)
    grouped = frame.groupby("order_id", as_index=False).agg(**named)
    return grouped


def _payment_total(orders: pd.DataFrame) -> float:
    """Sum payments once per order. Summing the item table would double-count."""
    if "order_payment_total" not in orders.columns:
        return float("nan")
    return _finite_sum(orders, "order_payment_total")


def _monthly(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    empty_info = {
        "month_revenue_change": float("nan"),
        "comparison_month": "",
        "comparison_month_revenue": float("nan"),
        "previous_month": "",
        "previous_month_revenue": float("nan"),
        "last_month_is_partial": False,
        "partial_month_label": "",
    }
    if "purchase_month_start" not in frame.columns:
        return pd.DataFrame(columns=["purchase_month_start", "revenue", "units", "orders"]), empty_info
    usable = frame.dropna(subset=["purchase_month_start"])
    if usable.empty:
        return pd.DataFrame(columns=["purchase_month_start", "revenue", "units", "orders"]), empty_info
    monthly = usable.groupby("purchase_month_start", as_index=False).agg(
        revenue=("line_revenue", "sum"),
        units=("line_revenue", "size"),
        orders=("order_id", "nunique"),
    )
    monthly = monthly.sort_values("purchase_month_start").reset_index(drop=True)
    max_ts = frame["order_purchase_timestamp"].max() if "order_purchase_timestamp" in frame.columns else pd.NaT
    complete, partial, partial_label = _complete_months(monthly, max_ts)
    info = dict(empty_info)
    info["last_month_is_partial"] = partial
    info["partial_month_label"] = partial_label
    if len(complete) >= 2:
        latest = complete.iloc[-1]
        previous = complete.iloc[-2]
        previous_revenue = float(previous["revenue"])
        change = float("nan")
        if previous_revenue:
            change = (float(latest["revenue"]) - previous_revenue) / previous_revenue
        info.update(
            {
                "month_revenue_change": change,
                "comparison_month": pd.Timestamp(latest["purchase_month_start"]).strftime("%B %Y"),
                "comparison_month_revenue": float(latest["revenue"]),
                "previous_month": pd.Timestamp(previous["purchase_month_start"]).strftime("%B %Y"),
                "previous_month_revenue": previous_revenue,
            }
        )
    return monthly, info


def _complete_months(
    monthly: pd.DataFrame, max_ts: object
) -> tuple[pd.DataFrame, bool, str]:
    """Drop the last month when the data file ends before that month is over."""
    if monthly.empty or pd.isna(max_ts):
        return monthly, False, ""
    last_start = pd.Timestamp(monthly.iloc[-1]["purchase_month_start"])
    days_in_month = calendar.monthrange(last_start.year, last_start.month)[1]
    covered = (pd.Timestamp(max_ts).normalize() - last_start.normalize()).days + 1
    partial = covered < days_in_month
    label = last_start.strftime("%B %Y")
    if partial and len(monthly) > 1:
        return monthly.iloc[:-1].copy(), True, label
    return monthly.copy(), partial, label if partial else ""


def _by_category(frame: pd.DataFrame) -> pd.DataFrame:
    column = "product_category_label" if "product_category_label" in frame.columns else "product_category"
    if column not in frame.columns:
        return pd.DataFrame(columns=["category", "units", "orders", "revenue", "avg_item_revenue", "avg_review", "freight_share", "revenue_share"])
    summary = _group_money(frame, column, "category")
    if "review_score" in frame.columns:
        reviews = frame.groupby(frame[column].astype("string").fillna("Unknown"), dropna=False)["review_score"].mean()
        summary["avg_review"] = summary["category"].map(reviews)
    else:
        summary["avg_review"] = np.nan
    return summary


def _by_state(frame: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if "customer_state" not in frame.columns:
        return pd.DataFrame(columns=["customer_state", "state", "units", "orders", "revenue", "avg_order_revenue", "avg_review", "avg_delivery_days"])
    items = _group_money(frame, "customer_state", "customer_state")
    delivery = pd.DataFrame(columns=["customer_state", "avg_delivery_days", "avg_review"])
    if "customer_state" in orders.columns:
        state = orders["customer_state"].astype("string").fillna("unknown")
        pieces = {"customer_state": state}
        agg = {}
        if "delivery_days" in orders.columns:
            pieces["delivery_days"] = pd.to_numeric(orders["delivery_days"], errors="coerce")
            agg["avg_delivery_days"] = ("delivery_days", "mean")
        if "review_score" in orders.columns:
            pieces["review_score"] = pd.to_numeric(orders["review_score"], errors="coerce")
            agg["avg_review"] = ("review_score", "mean")
        if agg:
            delivery = pd.DataFrame(pieces).groupby("customer_state", as_index=False).agg(**agg)
    merged = items.merge(delivery, on="customer_state", how="left")
    merged["state"] = merged["customer_state"].map(state_display)
    if "avg_review" not in merged.columns:
        merged["avg_review"] = np.nan
    if "avg_delivery_days" not in merged.columns:
        merged["avg_delivery_days"] = np.nan
    return merged.sort_values("revenue", ascending=False).reset_index(drop=True)


def _group_money(frame: pd.DataFrame, source: str, output: str) -> pd.DataFrame:
    columns = [output, "units", "orders", "revenue", "freight", "avg_item_revenue", "freight_share", "revenue_share", "avg_order_revenue"]
    if source not in frame.columns:
        return pd.DataFrame(columns=columns)
    keys = frame[source].astype("string").fillna("unknown")
    if source in {"payment_type", "order_status", "product_category"}:
        keys = keys.replace({"<NA>": "unknown"})
    summary = pd.DataFrame(
        {
            output: keys,
            "order_id": frame["order_id"],
            "line_revenue": pd.to_numeric(frame["line_revenue"], errors="coerce"),
            "freight_value": pd.to_numeric(frame["freight_value"], errors="coerce") if "freight_value" in frame.columns else 0.0,
        }
    )
    grouped = summary.groupby(output, as_index=False, dropna=False).agg(
        units=("line_revenue", "size"),
        orders=("order_id", "nunique"),
        revenue=("line_revenue", "sum"),
        freight=("freight_value", "sum"),
    )
    grouped["avg_item_revenue"] = grouped["revenue"] / grouped["units"].replace(0, np.nan)
    grouped["avg_order_revenue"] = grouped["revenue"] / grouped["orders"].replace(0, np.nan)
    grouped["freight_share"] = grouped["freight"] / grouped["revenue"].replace(0, np.nan)
    total = float(grouped["revenue"].sum())
    grouped["revenue_share"] = grouped["revenue"] / total if total else 0.0
    return grouped.sort_values("revenue", ascending=False).reset_index(drop=True)


def _by_review(orders: pd.DataFrame) -> pd.DataFrame:
    if "review_score" not in orders.columns:
        return pd.DataFrame(columns=["review_score", "orders", "average_order_revenue"])
    score = pd.to_numeric(orders["review_score"], errors="coerce").round()
    revenue = pd.to_numeric(orders["order_revenue"], errors="coerce") if "order_revenue" in orders.columns else np.nan
    summary = pd.DataFrame({"review_score": score, "order_revenue": revenue}).dropna(subset=["review_score"])
    summary = summary.loc[summary["review_score"].between(1, 5)]
    if summary.empty:
        return pd.DataFrame(columns=["review_score", "orders", "average_order_revenue"])
    grouped = summary.groupby("review_score", as_index=False).agg(
        orders=("review_score", "size"),
        average_order_revenue=("order_revenue", "mean"),
    )
    return grouped.sort_values("review_score")


def _top_cities(frame: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    if "customer_city" not in frame.columns:
        return pd.DataFrame(columns=["city", "orders", "revenue"])
    state = frame["customer_state"].astype("string").fillna("unknown") if "customer_state" in frame.columns else ""
    city = frame["customer_city"].astype("string").fillna("Unknown")
    label = city + " (" + state.astype("string") + ")"
    summary = pd.DataFrame(
        {
            "city": label,
            "order_id": frame["order_id"],
            "line_revenue": pd.to_numeric(frame["line_revenue"], errors="coerce"),
        }
    )
    grouped = summary.groupby("city", as_index=False).agg(
        orders=("order_id", "nunique"),
        revenue=("line_revenue", "sum"),
    )
    return grouped.sort_values("revenue", ascending=False).head(limit).reset_index(drop=True)


def _histogram(revenue: np.ndarray, bins: int = 16) -> tuple[pd.DataFrame, float]:
    """Bin item revenue up to the 99th percentile so one huge order cannot stretch the chart."""
    columns = ["bin_start", "bin_end", "bin_label", "count"]
    values = np.asarray(revenue, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return pd.DataFrame(columns=columns), float("nan")
    cap = float(np.quantile(values, 0.99))
    clipped = values[values <= cap]
    if clipped.size == 0:
        clipped = values
        cap = float(np.max(values))
    counts, edges = np.histogram(clipped, bins=bins)
    rows = []
    for index, count in enumerate(counts):
        start = float(edges[index])
        end = float(edges[index + 1])
        rows.append(
            {
                "bin_start": start,
                "bin_end": end,
                "bin_label": f"R$ {start:,.0f}–{end:,.0f}",
                "count": int(count),
            }
        )
    return pd.DataFrame(rows), cap


def _correlations(frame: pd.DataFrame, orders: pd.DataFrame) -> dict[str, float]:
    correlations = {
        "review_vs_order_revenue": _corr(
            orders.get("review_score") if "review_score" in orders.columns else None,
            orders.get("order_revenue") if "order_revenue" in orders.columns else None,
        ),
        "freight_vs_price": _corr(
            frame.get("freight_value") if "freight_value" in frame.columns else None,
            frame.get("price") if "price" in frame.columns else None,
        ),
        "delivery_days_vs_review": _corr(
            orders.get("delivery_days") if "delivery_days" in orders.columns else None,
            orders.get("review_score") if "review_score" in orders.columns else None,
        ),
    }
    return correlations


def _corr(left: pd.Series | None, right: pd.Series | None) -> float:
    """Pearson correlation. None when either column has no variation."""
    if left is None or right is None:
        return float("nan")
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3:
        return float("nan")
    left_values = pair["left"].to_numpy(dtype=float)
    right_values = pair["right"].to_numpy(dtype=float)
    if np.std(left_values) == 0 or np.std(right_values) == 0:
        return float("nan")
    coefficient = np.corrcoef(left_values, right_values)[0, 1]
    return float(coefficient)


def _average_delivery(orders: pd.DataFrame) -> float:
    if "delivery_days" not in orders.columns:
        return float("nan")
    values = pd.to_numeric(orders["delivery_days"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    usable = values[(values >= 0) & (values <= 180)]
    if usable.size == 0:
        return float("nan")
    return float(np.mean(usable))


def _delivery_outlier_count(orders: pd.DataFrame) -> int:
    if "delivery_days" not in orders.columns:
        return 0
    values = pd.to_numeric(orders["delivery_days"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return int(np.sum((values < 0) | (values > 180)))


def _early_rate(orders: pd.DataFrame) -> float:
    if "delivery_delay_days" not in orders.columns:
        return float("nan")
    values = pd.to_numeric(orders["delivery_delay_days"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values < 0))


def _finite_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sum(values))


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].nunique(dropna=True))


def _empty_analysis() -> AnalysisResult:
    money_columns = ["units", "orders", "revenue", "freight", "avg_item_revenue", "freight_share", "revenue_share", "avg_order_revenue"]
    return AnalysisResult(
        kpis={
            "item_revenue": 0.0,
            "payment_revenue": float("nan"),
            "orders": 0,
            "units": 0,
            "customers": 0,
            "sellers": 0,
            "average_order_value": float("nan"),
            "average_review": float("nan"),
            "freight_share": float("nan"),
            "average_delivery_days": float("nan"),
            "early_arrival_rate": float("nan"),
            "delivery_outliers": 0,
            "month_revenue_change": float("nan"),
            "comparison_month": "",
            "comparison_month_revenue": float("nan"),
            "previous_month": "",
            "previous_month_revenue": float("nan"),
            "last_month_is_partial": False,
            "partial_month_label": "",
        },
        numpy_stats=numpy_revenue_stats(np.array([])),
        monthly=pd.DataFrame(columns=["purchase_month_start", "revenue", "units", "orders"]),
        by_category=pd.DataFrame(columns=["category", *money_columns]),
        by_state=pd.DataFrame(columns=["customer_state", "state", "units", "orders", "revenue", "avg_order_revenue", "avg_review", "avg_delivery_days"]),
        by_payment=pd.DataFrame(columns=["payment_type", *money_columns]),
        by_status=pd.DataFrame(columns=["order_status", *money_columns]),
        by_review=pd.DataFrame(columns=["review_score", "orders", "average_order_revenue"]),
        top_cities=pd.DataFrame(columns=["city", "orders", "revenue"]),
        histogram=pd.DataFrame(columns=["bin_start", "bin_end", "bin_label", "count"]),
        histogram_cap=float("nan"),
        correlations={
            "review_vs_order_revenue": float("nan"),
            "freight_vs_price": float("nan"),
            "delivery_days_vs_review": float("nan"),
        },
    )
