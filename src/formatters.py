"""Small helpers that turn numbers and codes into readable text."""

from __future__ import annotations

import math

from src.config import BRAZIL_STATE_NAMES


def _missing(value: object) -> bool:
    """Return True when a number cannot be shown."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def format_brl(value: object) -> str:
    """Format a value as Brazilian reals, the currency of this dataset."""
    if _missing(value):
        return "n/a"
    return f"R$ {float(value):,.2f}"


def format_int(value: object) -> str:
    """Format a count with thousands separators."""
    if _missing(value):
        return "n/a"
    return f"{int(round(float(value))):,}"


def format_number(value: object, digits: int = 2) -> str:
    """Format a decimal number, or n/a when it is missing."""
    if _missing(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def format_pct(value: object) -> str:
    """Format a fraction (0.12) as a percent (12.0%)."""
    if _missing(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def state_display(code: object) -> str:
    """Turn a state code such as SP into 'São Paulo (SP)'."""
    if code is None:
        return "Unknown"
    text = str(code).strip().upper()
    if text in {"", "NAN", "NONE", "<NA>", "UNKNOWN"}:
        return "Unknown"
    name = BRAZIL_STATE_NAMES.get(text)
    if name:
        return f"{name} ({text})"
    return text


def humanize_token(value: object) -> str:
    """Turn a slug such as credit_card into 'Credit Card'."""
    if value is None:
        return "Unknown"
    text = str(value).replace("_", " ").strip()
    if text.lower() in {"", "nan", "none", "<na>", "unknown"}:
        return "Unknown"
    return text.title()


def describe_correlation(value: object, left: str, right: str) -> str:
    """Explain a correlation in plain language for the lab report and the app."""
    if _missing(value):
        return f"There are not enough values to compare {left} with {right}."
    score = float(value)
    magnitude = abs(score)
    if magnitude < 0.2:
        strength = "a weak"
    elif magnitude < 0.5:
        strength = "a moderate"
    else:
        strength = "a strong"
    direction = "positive" if score >= 0 else "negative"
    return (
        f"There is {strength} {direction} relationship between {left} and {right} "
        f"(correlation {score:.2f}). Correlation does not mean one causes the other."
    )
