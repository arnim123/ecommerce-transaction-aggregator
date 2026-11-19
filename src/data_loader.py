"""Load the Olist CSV files from a folder or from an upload."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FILE_MATCH_RULES, TABLES


def match_table_key(filename: str) -> str | None:
    """Map a file name to a table key, or None when the name is not recognized.

    The rules are ordered so that order_items is tested before orders.
    A name like olist_order_items_dataset.csv contains "order_items" and must
    not be treated as the orders table.
    """
    stem = Path(filename).name.lower()
    for key, needles in FILE_MATCH_RULES:
        if any(needle in stem for needle in needles):
            return key
    return None


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and a UTF-8 byte-order mark from column names.

    The category translation file ships with a byte-order mark on the first
    column. Removing it lets the join key match the products table.
    """
    cleaned = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    frame = frame.copy()
    frame.columns = _unique_names(cleaned)
    return frame


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count}")
    return unique


def _read_csv(source: object) -> pd.DataFrame:
    """Read a CSV from a path or a file-like object, with a latin-1 fallback."""
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="latin-1", low_memory=False)


def read_table(source: object, key: str, display_name: str | None = None) -> pd.DataFrame:
    """Read one known table and reject it when required columns are missing."""
    spec = TABLES[key]
    label = display_name or str(spec["filename"])
    try:
        frame = _read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read {label}: {exc}") from exc

    frame = standardize_columns(frame)
    required = list(spec["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}."
        )
    if frame.empty:
        raise ValueError(f"{label} has headers but no data rows.")
    return frame


def load_tables_from_directory(
    directory: Path | str,
    require_all: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load every recognized CSV in a folder.

    When require_all is True, every file listed in TABLES must be present.
    The dashboard upload sets it to False so a smaller set of files can still
    run, as long as orders and order items are included.
    """
    folder = Path(directory)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder was not found: {folder}")

    missing_files = [
        str(spec["filename"])
        for spec in TABLES.values()
        if not (folder / str(spec["filename"])).exists()
    ]
    if require_all and missing_files:
        listed = "\n- ".join(missing_files)
        raise FileNotFoundError(
            f"These dataset files are missing from {folder}:\n- {listed}"
        )

    tables: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for key, spec in TABLES.items():
        path = folder / str(spec["filename"])
        if not path.exists():
            continue
        tables[key] = read_table(path, key)

    _require_core_tables(tables)
    return tables, warnings


def load_tables_from_named_bytes(
    files: list[tuple[str, bytes]],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load uploaded CSV bytes. The first item of each pair is the file name."""
    import io

    tables: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for name, payload in files:
        if not str(name).lower().endswith(".csv"):
            warnings.append(f"Skipped {name}: only .csv files are used.")
            continue
        key = match_table_key(name)
        if key is None:
            warnings.append(
                f"Skipped {name}: the file name does not match an Olist table."
            )
            continue
        if key in tables:
            warnings.append(f"{name} replaces an earlier upload for the {TABLES[key]['label']} table.")
        tables[key] = read_table(io.BytesIO(payload), key, display_name=name)

    _require_core_tables(tables)
    return tables, warnings


def _require_core_tables(tables: dict[str, pd.DataFrame]) -> None:
    missing = [
        str(TABLES[key]["filename"])
        for key, spec in TABLES.items()
        if spec["required"] and key not in tables
    ]
    if missing:
        listed = "\n- ".join(missing)
        raise ValueError(
            "The pipeline needs the orders file and the order items file.\n"
            f"Missing:\n- {listed}"
        )


def directory_signature(directory: Path | str) -> str:
    """Build a cache key that changes when any CSV in the folder changes."""
    folder = Path(directory)
    parts: list[str] = []
    if not folder.exists():
        return "missing"
    for path in sorted(folder.glob("*.csv")):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)
