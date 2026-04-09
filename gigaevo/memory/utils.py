"""Shared utility functions for the memory system.

These helpers are used across csv_loader, idea_bank, and ideas_tracker.
They live here so each module does not duplicate the definitions.
"""

from __future__ import annotations

import ast
import json
import math
import statistics
from typing import Any


def to_float(value: Any, *, default: float | None = None) -> float | None:
    """Convert value to float, returning ``default`` if conversion fails.

    Args:
        value: Anything that may be coercible to float (int, str, float).
        default: Returned when conversion fails, value is NaN, or value is
            infinite.  Defaults to ``None``.

    Returns:
        A finite float, or ``default``.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def parse_cell(value: Any) -> Any:
    """JSON-decode strings that start with ``{`` or ``[``; return other values unchanged.

    Used when reading CSVs where nested structures were JSON-serialised into a
    single cell (e.g. the ``parent_ids`` column produced by ``tools/redis2pd.py``).

    Args:
        value: Any value.  Non-strings are returned as-is.

    Returns:
        Decoded JSON value when applicable, otherwise the original value.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


def median(values: list[float]) -> float | None:
    """Compute the median of a list of floats.

    Args:
        values: List of floats.  May be empty.

    Returns:
        Median as a float, or ``None`` if the list is empty.
    """
    return float(statistics.median(values)) if values else None


def parse_string_list(value: Any) -> list[str]:
    """Parse a list of strings from various encoded forms.

    Handles: list, JSON-encoded list, AST-encoded list, bare string.
    Returns an empty list for None, empty string, or non-parseable input.

    Args:
        value: Input to parse. May be a list, JSON string, AST string, or bare string.

    Returns:
        A list of stripped non-empty strings.
    """
    if isinstance(value, list):
        return [str(i).strip() for i in value if str(i).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{(":
            try:
                return [str(i).strip() for i in json.loads(text) if str(i).strip()]
            except Exception:
                try:
                    return [
                        str(i).strip() for i in ast.literal_eval(text) if str(i).strip()
                    ]
                except Exception:
                    pass
        return [text]
    return []
