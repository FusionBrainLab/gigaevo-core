"""Utility helpers shared across *metaevolve* codebase."""
from __future__ import annotations

from importlib import import_module as _import_module

# Re-export JSON helper so callers can do `from src.utils import json`
json = _import_module("src.utils.json")

__all__ = ["json"] 