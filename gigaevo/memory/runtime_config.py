"""Runtime config path helpers for the memory module.

Provides only path resolution utilities.  All type-coercion helpers
(``to_bool``, ``to_int``, ``to_str``, ``to_list``, ``deep_get``,
``load_settings``) were removed; config loading uses OmegaConf directly.
"""

from __future__ import annotations

import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent


def resolve_settings_path(settings_path: str | Path | None = None) -> Path:
    """Return the settings YAML path.

    Priority: explicit argument → EVO_MEMORY_CONFIG_PATH env var →
    EVO_MEMORY_SETTINGS_PATH env var → default memory_backend.yaml.
    """
    if settings_path is not None:
        return Path(settings_path)
    env_primary = os.getenv("EVO_MEMORY_CONFIG_PATH")
    if env_primary:
        return Path(env_primary)
    env_fallback = os.getenv("EVO_MEMORY_SETTINGS_PATH")
    if env_fallback:
        return Path(env_fallback)
    return _THIS_DIR.parents[2] / "config" / "memory_backend.yaml"


def resolve_local_path(
    base: Path,
    raw: str | None,
    default_relative: str,
) -> Path:
    """Resolve *raw* relative to *base*.

    If *raw* is empty or None, returns ``base / default_relative``.
    Absolute paths are returned as-is.
    """
    if not raw:
        return base / default_relative
    p = Path(raw)
    if p.is_absolute():
        return p
    return base / p
