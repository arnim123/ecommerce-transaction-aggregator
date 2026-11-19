# E-Commerce Customer Transaction Aggregator

A beginner machine-learning project on the Olist Brazilian e-commerce dataset. It loads the related CSV files, cleans them, merges them into one row per item sold, calculates revenue, one-hot encodes the useful categories, and explores the results in charts and a Streamlit dashboard.

Currency is the Brazilian real (R$).

## What you practice

- Reading many CSV files and checking that the expected columns are present
- Removing duplicate rows and duplicate keys
- Choosing a missing-value rule instead of filling every blank
- Joining tables without duplicating revenue
- Turning categories into 0/1 columns
- Summarizing revenue with Pandas and NumPy, and drawing charts with Matplotlib
- Exploring the same pipeline in an interactive dashboard

## Dataset

Put the nine source files in `dataset/`. They are already in this folder.

| File | What it contains |
| --- | --- |
| `olist_orders_dataset.csv` | One row per order: status and timestamps |
| `olist_order_items_dataset.csv` | One row per unit sold: price and freight |
| `olist_order_payments_dataset.csv` | Payments. One order can have several rows |
| `olist_order_reviews_dataset.csv` | Review scores and comments |
| `olist_customers_dataset.csv` | Customer id, city, state, and zip prefix |
| `olist_sellers_dataset.csv` | Seller city, state, and zip prefix |
| `olist_products_dataset.csv` | Product category and size |
| `product_category_name_translation.csv` | Portuguese category to English category |
| `olist_geolocation_dataset.csv` | Many coordinates for each zip prefix |

Two ids look similar but mean different things. `customer_id` is a new id for every order. `customer_unique_id` is the person. Customer counts use `customer_unique_id`.

## Project structure

```text
streamlit_app.py           Streamlit dashboard
app.py                     ASGI entry point used by Vercel
experiments/
  experiment_01_preprocessing.py
  experiment_02_encoding_analysis.py
src/
  data_loader.py           Read the CSV files
  preprocessing.py         Clean, merge, and calculate revenue
  encoding.py              One-hot encoding
  analysis.py              Revenue statistics (NumPy and Pandas)
  plotting.py              Matplotlib figures
  charts.py                Interactive Plotly charts
  pipeline.py              Runs both experiments
  checks.py                Pass/fail checks
tests/test_pipeline.py
dataset/                   Source CSV files
outputs/                   Created when you run an experiment
```

## Setup

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Python 3.10 or newer is enough. This copy was run with Python 3.14.

## Experiment 1 — preprocessing and revenue

```bash
python experiments/experiment_01_preprocessing.py
```

The script prints a cleaning summary, the merge steps, revenue by status, and pass/fail checks. It writes:

- `outputs/cleaned_transactions.csv`
- `outputs/preprocessing_report.txt`
- `outputs/cleaning_summary.csv`
- `outputs/missing_values_report.csv`
- `outputs/merge_report.csv`

On this dataset the merged table has **112,650 item rows** and **98,666 orders**. Orders with no items stay in the orders file and are not copied into the merged table.

| Measure | Amount |
| --- | --- |
| Item revenue, all statuses | R$ 15,843,553.24 |
| Item revenue, excluding canceled and unavailable | R$ 15,735,527.03 |
| Delivered orders only | R$ 15,419,773.75 |
| Payments, added once per order | R$ 15,846,280.17 |

Purchases run from 4 September 2016 through 3 September 2018. September 2018 is only a few days, so the last point on the monthly chart is low because the month is incomplete.

### How a row is defined

Olist stores one row per unit, not a quantity column. Two units of the same product are two rows. The merged table uses that same grain, so adding `line_revenue` does not double-count items.

```text
line_revenue = price + freight_value
```

Both amounts are rounded to the cent before they are added.

### How missing values are handled

- Duplicate rows, and duplicate ids such as a repeated `order_id`, are removed. The first copy is kept.
- A missing or negative price is removed, because revenue would be undefined. A price of zero is kept.
- Missing freight is treated as 0.
- A blank product category becomes `unknown`. Missing product measurements are filled with the column median.
- Delivery timestamps are left blank. Filling them would invent deliveries that never happened.
- Review comments are mostly blank. They are not invented, and they are not copied into the revenue table.
- Review scores outside 1–5 are dropped. The rest are not imputed.

### How the tables are joined

Payments are summed to one row per order. Reviews are averaged to one score per order. The location file has many coordinates for each zip prefix; those coordinates are averaged to one center per prefix. Joining the raw location file would copy every order many times and inflate revenue.

The payment column `order_payment_total` belongs to the whole order. Do not add it across item rows. A two-item order would be counted twice. The dashboard adds payments only after keeping one row per order. Item revenue and payment totals are close, but they are not the same number: a voucher, interest, or a missing payment row can move them apart.

Canceled and unavailable orders stay in the cleaned file so you can see them. `counts_as_revenue` is false for those rows, and the default charts leave them out.

## Experiment 2 — one-hot encoding and revenue analysis

```bash
python experiments/experiment_02_encoding_analysis.py
```

You can run this file on its own. It repeats preprocessing, then encodes categories and writes the charts. Pandas builds the tables, NumPy computes the statistics, and Matplotlib saves the figures.

One-hot encoding turns a category into 0/1 columns. A payment type of `credit_card` becomes `payment_type_credit_card = 1` and `payment_type_boleto = 0`. After encoding, each row has exactly one `1` in each group.

These columns are encoded:

- `order_status`
- `payment_type`
- `customer_state`
- `seller_state`
- `product_category`

City names and ids are not encoded. There are thousands of cities and tens of thousands of products, so one column per value would be huge and almost entirely zeros. State and category carry the useful grouping. Year, month, price, freight, and review score stay numbers because they already have a numeric meaning.

The encoded table has 112,650 rows and 158 columns, of which 136 are one-hot flags.

Eligible item revenue (canceled and unavailable orders removed):

| Statistic | Value |
| --- | --- |
| Items | 112,101 |
| Total | R$ 15,735,527.03 |
| Mean | R$ 140.37 |
| Median | R$ 92.30 |
| Sample standard deviation | R$ 189.72 |

The largest categories are health and beauty, watches and gifts, and bed and bath. São Paulo is the largest customer state. Freight and price move together moderately. Delivery time and review score move in opposite directions: slower deliveries go with lower scores. That is a correlation, not proof that one causes the other.

Figures are saved in `outputs/figures/`:

- `monthly_revenue.png`
- `revenue_by_category.png`
- `revenue_by_state.png`
- `payment_type_share.png`
- `order_status_counts.png`
- `review_score_vs_revenue.png`
- `revenue_distribution.png`

The encoded file is `outputs/encoded_features.csv`.

If you later train a model to predict line revenue, remove `line_revenue`, `price`, and `freight` from the features first. Price plus freight is the answer, so leaving them in would make the task trivial. Review scores are collected after delivery, so they are not known at the moment of purchase.

## Dashboard

```bash
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. The first load reads about one million location rows and can take a minute. Later clicks reuse the result.

The sidebar can use the bundled `dataset/` folder or CSV files you upload. Orders and order items are required. The other files add customers, products, payments, reviews, and locations. Upload files with their original names, for example `olist_orders_dataset.csv`.

The pages are:

1. **Overview** — what the merge did, quality checks, and the main numbers
2. **Dataset preview** — the first 100 rows of each source file, with row counts
3. **Preprocessing** — duplicates, missing values before and after, and join match rates
4. **Merged transactions** — the revenue calculation and a column guide
5. **One-hot encoding** — a small before/after example and the real 0/1 columns
6. **Revenue dashboard** — KPIs and interactive charts
7. **Category analysis** — a sortable category table and a focus category
8. **Downloads** — cleaned CSV, encoded CSV, the filtered view, and Matplotlib figures

Sidebar filters for status, category, state, and purchase date apply to the overview numbers, the merged preview, the revenue page, and the category page. Canceled and unavailable orders are off by default. Preprocessing and encoding always describe the full loaded dataset.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover a small made-up store (duplicates, a negative price, a split payment, an invalid review score) and the real dataset in `dataset/`.

## Troubleshooting

- **`ModuleNotFoundError`** — activate the virtual environment and run `pip install -r requirements.txt` again.
- **Missing dataset file** — the experiment scripts expect all nine CSVs in `dataset/`. The error lists the file names.
- **Port already in use** — start Streamlit on another port: `streamlit run streamlit_app.py --server.port 8502`.
- **The monthly chart falls off at the end** — the source file stops on 3 September 2018, so that month is not a full month of sales.

## GitHub and Vercel

The dashboard is a long-running Streamlit server. `app.py` exposes that server as an ASGI app (`app = st.App("streamlit_app.py")`) so Vercel’s Python runtime can host it. Local experiments still use `streamlit run streamlit_app.py`.

Vercel’s project folder is read-only, so the “save into outputs/” buttons only work on your machine. The download buttons work in both places. The first visit processes about one million location rows and can take a minute. A session on the Hobby plan can stay connected for up to five minutes.
