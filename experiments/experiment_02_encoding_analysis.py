"""Experiment 2: one-hot encode categories and analyze revenue.

Pandas builds the tables, NumPy summarizes the revenue column, and
Matplotlib saves the charts in outputs/figures.

Run this file from the project folder:

    python experiments/experiment_02_encoding_analysis.py

The script runs preprocessing itself, so Experiment 1 does not have to be
run first. Running Experiment 1 first is still useful because it writes the
cleaned CSV and the preprocessing report.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import numpy_revenue_stats
from src.config import FIGURES_DIR, OUTPUT_DIR
from src.pipeline import run_from_directory
from src.plotting import save_all_figures
from src.reporting import display_path, experiment_two_report


def main() -> int:
    started = time.perf_counter()
    try:
        result = run_from_directory(progress=print)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Experiment 2 stopped: {exc}")
        return 1

    eligible = result.transactions.loc[result.transactions["counts_as_revenue"], "line_revenue"]
    revenue = pd.to_numeric(eligible, errors="coerce").to_numpy(dtype=float)
    revenue = revenue[np.isfinite(revenue)]
    # These three calls are the NumPy core of the experiment. The same helper
    # feeds the dashboard, and the values are checked against it below.
    stats = numpy_revenue_stats(revenue)
    numpy_total = float(np.sum(revenue))
    numpy_median = float(np.median(revenue)) if revenue.size else float("nan")
    if not np.isclose(numpy_total, stats["total"], rtol=0, atol=0.01):
        print("NumPy total does not match numpy_revenue_stats.")
        return 1
    if revenue.size and not np.isclose(numpy_median, stats["median"], rtol=0, atol=0.01):
        print("NumPy median does not match numpy_revenue_stats.")
        return 1
    if not np.isclose(numpy_total, result.default_analysis.numpy_stats["total"], rtol=0, atol=0.01):
        print("Eligible revenue total does not match the analysis module.")
        return 1

    report = experiment_two_report(result)
    print()
    print(report)
    print(f"NumPy sum of eligible line revenue: R$ {numpy_total:,.2f}")
    print(f"NumPy median item revenue:          R$ {numpy_median:,.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    encoded_path = OUTPUT_DIR / "encoded_features.csv"
    category_path = OUTPUT_DIR / "category_revenue_summary.csv"
    stats_path = OUTPUT_DIR / "numpy_revenue_stats.csv"
    report_path = OUTPUT_DIR / "encoding_report.txt"
    result.encoded.to_csv(encoded_path, index=False)
    result.default_analysis.by_category.to_csv(category_path, index=False)
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    figures = save_all_figures(result.default_analysis, FIGURES_DIR)

    print("\nSaved files")
    print("-----------")
    for path in (encoded_path, category_path, stats_path, report_path, *figures):
        print(f"  {display_path(path)}")

    failed = result.sanity_checks["status"].eq("Fail").any()
    elapsed = time.perf_counter() - started
    print(f"\nFinished in {elapsed:.1f} seconds.")
    if failed:
        print("One or more quality checks failed.")
        return 1
    print("All quality checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
