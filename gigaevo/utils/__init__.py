from __future__ import annotations

from importlib import import_module as _import_module

json = _import_module("gigaevo.utils.json")

__all__ = ["json"]
