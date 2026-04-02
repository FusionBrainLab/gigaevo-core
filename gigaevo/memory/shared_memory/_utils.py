"""Shared utility functions for the memory module.

Extracted from memory.py to eliminate duplication across
memory.py, a_mem_memory_creation.py, and other submodules.
"""

from __future__ import annotations

from typing import Any


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str_or_empty(value: Any) -> str:
    """Convert to string, preserving falsy-but-valid values like 0."""
    if value is None:
        return ""
    return str(value)


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)
