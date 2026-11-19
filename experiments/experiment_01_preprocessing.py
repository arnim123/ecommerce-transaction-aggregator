"""Experiment 1: load every CSV, clean it, merge it, and calculate revenue.

Run this file from the project folder:

    python experiments/experiment_01_preprocessing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR
from src.pipeline import run_from_directory
from src.reporting import display_path, experiment_one_report


def main() -> int:
    started = time.perf_counter()
    try:
        result = run_from_directory(progress=print)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Experiment 1 stopped: {exc}")
        return 1

    report = experiment_one_report(result)
    print()
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transactions_path = OUTPUT_DIR / "cleaned_transactions.csv"
    report_path = OUTPUT_DIR / "preprocessing_report.txt"
    result.transactions.to_csv(transactions_path, index=False)
    result.table_overview.to_csv(OUTPUT_DIR / "cleaning_summary.csv", index=False)
    result.missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False)
    result.merge_report.to_csv(OUTPUT_DIR / "merge_report.csv", index=False)
    report_path.write_text(report, encoding="utf-8")

    print("Saved files")
    print("-----------")
    for path in (
        transactions_path,
        report_path,
        OUTPUT_DIR / "cleaning_summary.csv",
        OUTPUT_DIR / "missing_values_report.csv",
        OUTPUT_DIR / "merge_report.csv",
    ):
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
