"""Interactive dashboard for the e-commerce transaction aggregator.

Run it from the project folder:

    streamlit run streamlit_app.py

Vercel does not start that command. It imports the ASGI app in app.py,
which loads this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import analyze_transactions
from src.charts import (
    category_revenue,
    monthly_revenue,
    payment_mix,
    revenue_histogram,
    review_revenue,
    state_revenue,
    status_mix,
)
from src.config import (
    DATASET_DIR,
    DATA_DICTIONARY,
    FIGURES_DIR,
    OUTPUT_DIR,
    REVENUE_EXCLUDED_STATUSES,
    TABLES,
)
from src.data_loader import directory_signature, load_tables_from_named_bytes
from src.formatters import (
    describe_correlation,
    format_brl,
    format_int,
    format_number,
    format_pct,
    state_display,
)
from src.pipeline import PipelineResult, build_from_tables, run_from_directory
from src.plotting import save_all_figures

PAGES = [
    "Overview",
    "Dataset preview",
    "Preprocessing",
    "Merged transactions",
    "One-hot encoding",
    "Revenue dashboard",
    "Category analysis",
    "Downloads",
]

STAT_ROWS = [
    ("count", "Items", "count"),
    ("total", "Total revenue", "money"),
    ("mean", "Mean", "money"),
    ("median", "Median", "money"),
    ("std", "Standard deviation", "money"),
    ("min", "Minimum", "money"),
    ("p25", "25th percentile", "money"),
    ("p75", "75th percentile", "money"),
    ("p95", "95th percentile", "money"),
    ("max", "Maximum", "money"),
]


def main() -> None:
    st.set_page_config(
        page_title="E-Commerce Transaction Aggregator",
        page_icon=":material/storefront:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Transaction Aggregator")
    st.sidebar.caption("Clean Olist orders, calculate revenue, and encode categories.")
    source = st.sidebar.radio("Dataset", ["Bundled dataset", "Upload CSV files"], key="data_source")
    uploads = None
    if source == "Upload CSV files":
        uploads = st.sidebar.file_uploader(
            "Olist CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Use the original file names, such as olist_orders_dataset.csv. Orders and order items are required.",
        )
        with st.sidebar.expander("Expected file names"):
            for spec in TABLES.values():
                required = "required" if spec["required"] else "optional"
                st.caption(f"{spec['filename']} ({required})")

    page = st.sidebar.radio("Section", PAGES, key="section")
    try:
        result = load_result(source, uploads)
    except (FileNotFoundError, ValueError) as exc:
        st.title("E-Commerce Transaction Aggregator")
        st.error(str(exc))
        st.info(
            "Use the bundled dataset, or upload the Olist CSV files. "
            "The orders file and the order items file are required."
        )
        return
    except Exception as exc:
        st.title("E-Commerce Transaction Aggregator")
        st.error("The pipeline stopped because of an unexpected error.")
        st.exception(exc)
        return

    for warning in result.warnings:
        st.sidebar.warning(warning)

    filtered = render_filters(result)
    st.sidebar.caption(f"pandas {pd.__version__} · numpy {np.__version__}")
    render_page(page, result, filtered)


@st.cache_data(show_spinner=False)
def run_bundled(dataset_dir: str, signature: str) -> PipelineResult:
    """Cache the bundled dataset. `signature` changes when a CSV changes."""
    del signature
    return run_from_directory(dataset_dir, require_all=True)


@st.cache_data(show_spinner=False)
def run_uploaded(payload: tuple[tuple[str, bytes], ...]) -> tuple[PipelineResult, tuple[str, ...]]:
    """Cache an upload. The bytes are the cache key."""
    tables, warnings = load_tables_from_named_bytes(list(payload))
    result = build_from_tables(tables)
    result.warnings = list(warnings) + result.warnings
    return result, tuple(warnings)


def load_result(source: str, uploads) -> PipelineResult:
    """Load either the files in dataset/ or the files the user uploaded."""
    if source == "Bundled dataset":
        if not DATASET_DIR.exists():
            raise FileNotFoundError(f"Dataset folder was not found: {DATASET_DIR}")
        signature = directory_signature(DATASET_DIR)
        with st.spinner("Cleaning and merging the nine CSV files. The first run can take a minute."):
            return run_bundled(str(DATASET_DIR), signature)

    if not uploads:
        raise ValueError(
            "Upload the Olist CSV files to continue. Start with "
            "olist_orders_dataset.csv and olist_order_items_dataset.csv."
        )
    payload = tuple(sorted((file.name, file.getvalue()) for file in uploads))
    with st.spinner("Reading the uploaded files and building the revenue table."):
        result, _warnings = run_uploaded(payload)
    return result


def render_filters(result: PipelineResult) -> pd.DataFrame:
    """Sidebar filters. An empty category or state selection means all values."""
    frame = result.transactions
    signature = f"{len(frame)}-{frame['order_purchase_timestamp'].min()}"
    statuses = sorted(frame["order_status"].fillna("unknown").astype(str).unique())
    default_statuses = [status for status in statuses if status not in REVENUE_EXCLUDED_STATUSES] or statuses
    categories = sorted(frame["product_category_label"].dropna().astype(str).unique())
    states = sorted(frame["customer_state"].dropna().astype(str).unique()) if "customer_state" in frame.columns else []

    st.sidebar.markdown("**Filters**")
    st.sidebar.caption("Filters apply to the overview numbers, merged preview, revenue page, and category page.")
    chosen_statuses = st.sidebar.multiselect(
        "Order status",
        statuses,
        default=default_statuses,
        key=f"status_{signature}",
        help="Canceled and unavailable orders are off by default so revenue reflects orders the store still counts.",
    )
    chosen_categories = st.sidebar.multiselect(
        "Product categories",
        categories,
        placeholder="All categories",
        key=f"category_{signature}",
    )
    chosen_states = st.sidebar.multiselect(
        "Customer state",
        states,
        format_func=state_display,
        placeholder="All states",
        key=f"state_{signature}",
    )

    start = end = full_start = full_end = None
    if frame["order_purchase_timestamp"].notna().any():
        full_start = frame["order_purchase_timestamp"].min().date()
        full_end = frame["order_purchase_timestamp"].max().date()
        picked = st.sidebar.date_input(
            "Purchase dates",
            value=(full_start, full_end),
            min_value=full_start,
            max_value=full_end,
            key=f"dates_{signature}",
        )
        start, end = _date_pair(picked)

    filtered = frame
    if chosen_statuses:
        filtered = filtered.loc[filtered["order_status"].astype(str).isin(chosen_statuses)]
    else:
        filtered = filtered.iloc[0:0]
    if chosen_categories:
        filtered = filtered.loc[filtered["product_category_label"].astype(str).isin(chosen_categories)]
    if chosen_states and "customer_state" in filtered.columns:
        filtered = filtered.loc[filtered["customer_state"].astype(str).isin(chosen_states)]
    if start is not None and end is not None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        purchased = filtered["order_purchase_timestamp"]
        in_range = purchased.between(start_ts, end_ts)
        if start == full_start and end == full_end:
            in_range = in_range | purchased.isna()
        filtered = filtered.loc[in_range]
    return filtered


def render_page(page: str, result: PipelineResult, filtered: pd.DataFrame) -> None:
    pages = {
        "Overview": page_overview,
        "Dataset preview": page_preview,
        "Preprocessing": page_preprocessing,
        "Merged transactions": page_merged,
        "One-hot encoding": page_encoding,
        "Revenue dashboard": page_revenue,
        "Category analysis": page_category,
        "Downloads": page_downloads,
    }
    pages[page](result, filtered)


def page_overview(result: PipelineResult, filtered: pd.DataFrame) -> None:
    st.title("E-Commerce Transaction Aggregator")
    st.caption(
        "A beginner project on the Olist Brazilian e-commerce dataset. "
        "Experiment 1 cleans and merges the tables. Experiment 2 encodes categories and summarizes revenue."
    )
    st.badge("Brazilian reals (R$)", color="green")

    left, middle, right = st.columns(3)
    with left:
        st.subheader("1. Preprocess")
        st.write(
            "Load the CSV files, remove duplicate rows, decide what to do with missing values, "
            "and join customers, products, sellers, payments, and reviews onto each item."
        )
    with middle:
        st.subheader("2. Calculate revenue")
        st.write(
            "Each row is one unit. Line revenue is the unit price plus freight. "
            "Payments are stored once per order so a two-item order is not counted twice."
        )
    with right:
        st.subheader("3. Encode and explore")
        st.write(
            "Turn status, payment type, state, and category into 0/1 columns, "
            "then compare revenue with Pandas, NumPy, and the charts on this dashboard."
        )

    st.subheader("What the merge produced")
    merge_view = result.merge_report.copy()
    merge_view["match_rate"] = merge_view["match_rate"].map(format_pct)
    st.dataframe(merge_view, hide_index=True, width="stretch")

    st.subheader("Quality checks")
    st.dataframe(result.sanity_checks, hide_index=True, width="stretch")

    st.subheader("Revenue in the current filter")
    if filtered.empty:
        st.info("No transactions match the current filters.")
        return
    show_kpis(analyze_transactions(filtered), result)
    with st.expander("Notes from the pipeline"):
        for note in result.notes:
            st.write("- " + note)


def page_preview(result: PipelineResult, filtered: pd.DataFrame) -> None:
    del filtered
    st.header("Dataset preview")
    st.write("These are the original files after the headers are read, before rows are removed. Only the first 100 rows are shown.")
    labels = {key: str(TABLES[key]["label"]) for key in result.previews}
    selected = st.selectbox("Table", list(result.previews), format_func=lambda key: labels.get(key, key))
    overview = result.table_overview.loc[result.table_overview["table"] == selected]
    preview = result.previews[selected]
    if not overview.empty:
        row = overview.iloc[0]
        missing_cells = 0
        if not result.missing_report.empty:
            table_missing = result.missing_report.loc[result.missing_report["table"] == selected, "missing_before"]
            missing_cells = int(table_missing.sum()) if not table_missing.empty else 0
        one, two, three, four = st.columns(4)
        one.metric("Rows in file", f"{int(row['rows_loaded']):,}", border=True)
        two.metric("Columns", f"{int(row['columns']):,}", border=True)
        three.metric("Exact duplicate rows", f"{int(row['exact_duplicates_removed']):,}", border=True)
        four.metric("Missing cells", f"{missing_cells:,}", border=True)
    st.caption(TABLES[selected]["filename"])
    st.dataframe(preview, hide_index=True, width="stretch")


def page_preprocessing(result: PipelineResult, filtered: pd.DataFrame) -> None:
    del filtered
    st.header("Preprocessing")
    st.write(
        "This page describes the full loaded dataset. The sidebar filters do not change these counts. "
        "Missing delivery dates are kept, because filling them would invent deliveries."
    )
    st.subheader("Rows removed")
    st.dataframe(result.table_overview, hide_index=True, width="stretch")

    st.subheader("What changed in each table")
    for key, actions in result.actions.items():
        label = str(TABLES.get(key, {}).get("label", key))
        with st.expander(label):
            for action in actions:
                st.write("- " + action)

    st.subheader("Missing values")
    if result.missing_report.empty:
        st.info("No missing values were recorded.")
    else:
        chart = missing_chart(result.missing_report)
        st.plotly_chart(chart, key="missing_chart")
        st.dataframe(result.missing_report, hide_index=True, width="stretch")

    st.subheader("Merge")
    merge_view = result.merge_report.copy()
    merge_view["match_rate"] = merge_view["match_rate"].map(format_pct)
    st.dataframe(merge_view, hide_index=True, width="stretch")

    st.subheader("Revenue by status, before filters")
    st.write("Canceled and unavailable orders are still listed here. `counts_as_revenue` shows whether they enter the default totals.")
    st.dataframe(_money_frame(result.status_totals, ["item_revenue"]), hide_index=True, width="stretch")


def page_merged(result: PipelineResult, filtered: pd.DataFrame) -> None:
    st.header("Merged transactions")
    st.write(
        "One row is one unit sold. Line revenue is that unit's price plus its freight. "
        "The preview follows the sidebar filters; the full file is on the Downloads page."
    )
    if filtered.empty:
        st.info("No transactions match the current filters.")
        return
    one, two, three, four = st.columns(4)
    one.metric("Rows in this view", f"{len(filtered):,}", border=True)
    two.metric("Columns", f"{filtered.shape[1]:,}", border=True)
    three.metric("Item revenue", format_brl(filtered["line_revenue"].sum()), border=True)
    four.metric("Full cleaned table", f"{len(result.transactions):,} rows", border=True)

    st.subheader("How a row becomes revenue")
    sample = filtered[["order_id", "order_item_id", "price", "freight_value", "line_revenue"]].head(8)
    st.dataframe(sample, hide_index=True, width="stretch")
    st.caption("line_revenue = price + freight_value, rounded to the cent.")

    st.subheader("Preview")
    st.dataframe(filtered.head(200), hide_index=True, width="stretch")

    st.subheader("Column guide")
    dictionary = pd.DataFrame(DATA_DICTIONARY, columns=["column", "group", "meaning"])
    st.dataframe(dictionary, hide_index=True, width="stretch")


def page_encoding(result: PipelineResult, filtered: pd.DataFrame) -> None:
    del filtered
    st.header("One-hot encoding")
    st.write(
        "A category with three values becomes three columns. Exactly one of those columns is 1 on each row, "
        "and the others are 0. The encoded table has the same number of rows as the cleaned transactions. "
        "This page always shows the full dataset, not the sidebar filter."
    )
    before, after = st.columns(2)
    with before:
        st.markdown("**Before**")
        st.dataframe(
            pd.DataFrame({"payment_type": ["credit_card", "boleto"]}),
            hide_index=True,
            width="stretch",
        )
    with after:
        st.markdown("**After**")
        st.dataframe(
            pd.DataFrame(
                {
                    "payment_type_credit_card": [1, 0],
                    "payment_type_boleto": [0, 1],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    encoded = result.encoded
    dummy_count = int(result.encoding_summary["new_columns"].sum()) if not result.encoding_summary.empty else 0
    one, two, three = st.columns(3)
    one.metric("Encoded rows", f"{len(encoded):,}", border=True)
    two.metric("Encoded columns", f"{encoded.shape[1]:,}", border=True)
    three.metric("One-hot columns", f"{dummy_count:,}", border=True)
    st.dataframe(result.encoding_summary, hide_index=True, width="stretch")

    if not result.encoding_summary.empty:
        family = st.selectbox(
            "Inspect one encoded feature",
            result.encoding_summary["original_column"].tolist(),
        )
        dummy_columns = [column for column in encoded.columns if column.startswith(f"{family}_")]
        id_columns = [column for column in ("order_id", "order_item_id", "line_revenue") if column in encoded.columns]
        st.dataframe(encoded[id_columns + dummy_columns].head(25), hide_index=True, width="stretch")
        active = [column for column in dummy_columns if encoded.iloc[0][column] == 1]
        if active:
            st.caption("On the first row, these flags are 1: " + ", ".join(active) + ".")

    st.subheader("Columns that were not one-hot encoded")
    st.dataframe(result.not_encoded, hide_index=True, width="stretch")
    st.info(
        "If you later train a model to predict line revenue, remove line_revenue, price, and freight from the features. "
        "Price plus freight is the target, so leaving them in would make the task trivial. "
        "Review scores happen after delivery, so they are not known at purchase time.",
        title="Before you train a model",
    )


def page_revenue(result: PipelineResult, filtered: pd.DataFrame) -> None:
    st.header("Revenue dashboard")
    st.write("Item revenue is price plus freight. Order-level numbers, such as review score and delivery time, are counted once per order.")
    if filtered.empty:
        st.info("No transactions match the current filters.")
        return
    analysis = analyze_transactions(filtered)
    show_kpis(analysis, result)
    st.plotly_chart(monthly_revenue(analysis), key="monthly_revenue")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(category_revenue(analysis), key="category_revenue")
    with right:
        st.plotly_chart(state_revenue(analysis), key="state_revenue")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(payment_mix(analysis), key="payment_mix")
    with right:
        st.plotly_chart(status_mix(analysis), key="status_mix")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(review_revenue(analysis), key="review_revenue")
    with right:
        st.plotly_chart(revenue_histogram(analysis), key="histogram")
        if np.isfinite(analysis.histogram_cap):
            st.caption(
                f"Bins stop at the 99th percentile ({format_brl(analysis.histogram_cap)}). "
                "The statistics table still includes every item."
            )

    st.subheader("NumPy summary of item revenue")
    st.caption("The standard deviation is the sample version, divided by N − 1.")
    stats_left, stats_right = st.columns([1, 1])
    with stats_left:
        st.dataframe(stats_table(analysis.numpy_stats), hide_index=True, width="stretch")
    with stats_right:
        correlations = analysis.correlations
        st.write(describe_correlation(correlations["review_vs_order_revenue"], "review score", "order revenue"))
        st.write(describe_correlation(correlations["freight_vs_price"], "freight", "item price"))
        st.write(describe_correlation(correlations["delivery_days_vs_review"], "delivery time", "review score"))

    st.subheader("Top customer cities")
    cities = analysis.top_cities.copy()
    if not cities.empty:
        cities["revenue"] = cities["revenue"].map(format_brl)
        st.dataframe(cities, hide_index=True, width="stretch")


def page_category(result: PipelineResult, filtered: pd.DataFrame) -> None:
    del result
    st.header("Category analysis")
    st.write("Categories use the English name when a translation exists. Revenue is the sum of line revenue for items in that category.")
    if filtered.empty:
        st.info("No transactions match the current filters.")
        return
    analysis = analyze_transactions(filtered)
    summary = analysis.by_category.copy()
    if summary.empty:
        st.info("This view has no category column to summarize.")
        return
    display = summary.copy()
    display["revenue_share"] = display["revenue_share"] * 100
    display["freight_share"] = display["freight_share"] * 100
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "category": "Category",
            "units": st.column_config.NumberColumn("Units", format="%d"),
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "revenue": st.column_config.NumberColumn("Revenue (BRL)", format="%.2f"),
            "freight": st.column_config.NumberColumn("Freight (BRL)", format="%.2f"),
            "avg_item_revenue": st.column_config.NumberColumn("Avg item revenue", format="%.2f"),
            "avg_order_revenue": st.column_config.NumberColumn("Avg order revenue", format="%.2f"),
            "freight_share": st.column_config.NumberColumn("Freight share (%)", format="%.1f"),
            "revenue_share": st.column_config.NumberColumn("Revenue share (%)", format="%.1f"),
            "avg_review": st.column_config.NumberColumn("Avg review", format="%.2f"),
        },
    )
    st.plotly_chart(category_revenue(analysis, limit=15), key="category_page_bar")

    focus = st.selectbox("Focus category", summary["category"].tolist())
    focus_row = summary.loc[summary["category"] == focus].iloc[0]
    subset = filtered.loc[filtered["product_category_label"].astype(str) == str(focus)]
    focused = analyze_transactions(subset)
    one, two, three, four = st.columns(4)
    one.metric("Revenue", format_brl(focus_row["revenue"]), border=True)
    two.metric("Share of this view", format_pct(focus_row["revenue_share"]), border=True)
    three.metric("Units", f"{int(focus_row['units']):,}", border=True)
    four.metric("Orders", f"{int(focus_row['orders']):,}", border=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(monthly_revenue(focused), key="focus_monthly")
    with right:
        st.plotly_chart(payment_mix(focused), key="focus_payment")


def page_downloads(result: PipelineResult, filtered: pd.DataFrame) -> None:
    st.header("Downloads")
    st.write("The cleaned file is the merged table. The encoded file adds the 0/1 category columns used for modeling.")
    left, right = st.columns(2)
    with left:
        st.subheader("Cleaned transactions")
        st.caption(f"{len(result.transactions):,} rows · {result.transactions.shape[1]} columns · all statuses")
        st.download_button(
            "Download cleaned CSV",
            data=to_csv_bytes(result.transactions),
            file_name="cleaned_transactions.csv",
            mime="text/csv",
            key="download_clean",
            on_click="ignore",
        )
    with right:
        st.subheader("Encoded features")
        st.caption(f"{len(result.encoded):,} rows · {result.encoded.shape[1]} columns")
        st.download_button(
            "Download encoded CSV",
            data=to_csv_bytes(result.encoded),
            file_name="encoded_features.csv",
            mime="text/csv",
            key="download_encoded",
            on_click="ignore",
        )

    st.subheader("Current filter")
    if filtered.empty:
        st.info("The current filter has no rows to download.")
    else:
        analysis = analyze_transactions(filtered)
        one, two = st.columns(2)
        with one:
            st.caption(f"Filtered transactions · {len(filtered):,} rows")
            st.download_button(
                "Download filtered CSV",
                data=to_csv_bytes(filtered),
                file_name="filtered_transactions.csv",
                mime="text/csv",
                key="download_filtered",
                on_click="ignore",
            )
        with two:
            st.caption(f"Category summary · {len(analysis.by_category):,} categories")
            st.download_button(
                "Download category summary",
                data=to_csv_bytes(analysis.by_category),
                file_name="category_revenue_summary.csv",
                mime="text/csv",
                key="download_category",
                on_click="ignore",
            )
        if st.button("Save Matplotlib figures for this filter"):
            try:
                paths = save_all_figures(analysis, FIGURES_DIR)
            except OSError as error:
                st.error(
                    "This environment cannot write figure files "
                    f"({error}). The charts on the revenue page are still available."
                )
            else:
                st.success("Saved " + ", ".join(path.name for path in paths) + f" in {FIGURES_DIR}.")

    if st.button("Save the full CSV files into outputs/"):
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cleaned_path = OUTPUT_DIR / "cleaned_transactions.csv"
            encoded_path = OUTPUT_DIR / "encoded_features.csv"
            result.transactions.to_csv(cleaned_path, index=False)
            result.encoded.to_csv(encoded_path, index=False)
        except OSError as error:
            st.error(
                "This environment cannot write CSV files to disk "
                f"({error}). Use the download buttons above."
            )
        else:
            st.success(f"Saved {cleaned_path.name} and {encoded_path.name} in {OUTPUT_DIR}.")


def show_kpis(analysis, result: PipelineResult) -> None:
    """Render the main revenue metrics for the filtered view."""
    kpis = analysis.kpis
    first = st.columns(4)
    first[0].metric("Item revenue", format_brl(kpis["item_revenue"]), border=True, help="Sum of price plus freight for the rows in this view.")
    first[1].metric("Payments", format_brl(kpis["payment_revenue"]), border=True, help="Payment totals added once per order, not once per item.")
    first[2].metric("Orders", format_int(kpis["orders"]), border=True, help=f"{format_int(kpis['sellers'])} sellers in this view.")
    first[3].metric("Units", format_int(kpis["units"]), border=True, help="One row in the merged table is one unit.")
    second = st.columns(4)
    second[0].metric("Customers", format_int(kpis["customers"]), border=True, help="Counted with customer_unique_id, the id of the person.")
    second[1].metric("Average order value", format_brl(kpis["average_order_value"]), border=True)
    second[2].metric("Average review", format_number(kpis["average_review"], 2), border=True, help="Averaged once per order, from 1 to 5.")
    second[3].metric("Average delivery", _days(kpis["average_delivery_days"]), border=True, help="Days from purchase to delivery, ignoring missing values and times outside 0–180 days.")

    freight = format_pct(kpis["freight_share"])
    early = format_pct(kpis["early_arrival_rate"])
    sentence = f"Freight is {freight} of item revenue."
    if early != "n/a":
        sentence += f" Of orders with both a delivery date and an estimate, {early} arrived before the estimate."
    outliers = int(kpis.get("delivery_outliers", 0) or 0)
    if outliers:
        sentence += f" {outliers:,} orders were left out of the delivery average because the timing looked impossible."
    if kpis.get("comparison_month"):
        sentence += (
            f" Revenue in {kpis['comparison_month']} was {format_brl(kpis['comparison_month_revenue'])}, "
            f"{format_pct(kpis['month_revenue_change'])} compared with {kpis['previous_month']}."
        )
        if kpis.get("last_month_is_partial"):
            sentence += f" {kpis['partial_month_label']} is incomplete in the file, so it is not used in that comparison."
    st.caption(sentence)
    st.caption(
        f"Full cleaned table: {format_int(result.headline['item_rows'])} item rows and "
        f"{format_brl(result.headline['item_revenue_eligible'])} of eligible revenue before filters."
    )


def stats_table(stats: dict) -> pd.DataFrame:
    rows = []
    for key, label, kind in STAT_ROWS:
        value = stats.get(key, float("nan"))
        shown = format_int(value) if kind == "count" else format_brl(value)
        rows.append({"statistic": label, "value": shown})
    return pd.DataFrame(rows)


def missing_chart(report: pd.DataFrame):
    """Bar chart of the columns that started with the most missing values."""
    import plotly.express as px

    top = report.loc[report["missing_before"] > 0].sort_values("missing_before", ascending=False).head(12).copy()
    if top.empty:
        return revenue_histogram_placeholder()
    top["label"] = top["table"] + " · " + top["column"]
    long = top.melt(
        id_vars="label",
        value_vars=["missing_before", "missing_after"],
        var_name="stage",
        value_name="missing_cells",
    )
    long["stage"] = long["stage"].map({"missing_before": "Before", "missing_after": "After"})
    figure = px.bar(
        long,
        x="missing_cells",
        y="label",
        color="stage",
        barmode="group",
        orientation="h",
        labels={"missing_cells": "Missing cells", "label": ""},
        color_discrete_sequence=["#C2410C", "#0F766E"],
    )
    figure.update_layout(
        template="plotly_white",
        height=max(380, 28 * top["label"].nunique() + 120),
        title="Missing values before and after cleaning",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=8, r=8, t=72, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def revenue_histogram_placeholder():
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_annotation(text="No missing values to chart.", showarrow=False)
    figure.update_layout(height=240, template="plotly_white")
    return figure


def _money_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    shown = frame.copy()
    for column in columns:
        if column in shown.columns:
            shown[column] = shown[column].map(format_brl)
    return shown


def _days(value: object) -> str:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{float(value):.1f} days"
    return "n/a"


def _date_pair(value) -> tuple:
    if isinstance(value, tuple) and len(value) == 2 and all(part is not None for part in value):
        start, end = value
        if start > end:
            start, end = end, start
        return start, end
    return None, None


@st.cache_data(show_spinner="Preparing the CSV file...")
def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


if __name__ == "__main__":
    main()
