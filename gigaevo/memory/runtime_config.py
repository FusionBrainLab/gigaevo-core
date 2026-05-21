from __future__ import annotations

import os
from pathlib import Path
import types
from typing import Any

_yaml: types.ModuleType | None

try:
    import yaml as _yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - defensive fallback
    _yaml = None


_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_SETTINGS_PATHS: tuple[Path, ...] = (_MODULE_DIR / "config.yaml",)


def resolve_settings_path(path: str | Path | None = None) -> Path:
    """Resolve the runtime-config YAML path consulted by ``load_settings``.

    Resolution order:

    1. The explicit ``path`` argument when provided.
    2. ``EVO_MEMORY_CONFIG_PATH`` environment variable.
    3. ``EVO_MEMORY_SETTINGS_PATH`` environment variable.
    4. The first existing entry in the package-local defaults
       (``gigaevo/memory/config.yaml``).

    When nothing resolves, returns the first default path as a stable
    placeholder. ``load_settings`` then short-circuits to ``{}`` if that
    placeholder does not exist on disk, so callers that rely on
    ``deep_get(..., default=...)`` continue to work without any config
    file present.
    """
    if path is not None:
        return Path(path)

    from_env = os.getenv("EVO_MEMORY_CONFIG_PATH")
    if from_env:
        return Path(from_env)

    from_env = os.getenv("EVO_MEMORY_SETTINGS_PATH")
    if from_env:
        return Path(from_env)

    for candidate in _DEFAULT_SETTINGS_PATHS:
        if candidate.exists():
            return candidate

    return _DEFAULT_SETTINGS_PATHS[0]


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load runtime settings as a mapping.

    Resolves the source file via :func:`resolve_settings_path`, then:

    - Returns ``{}`` when no file is present on disk so the caller can
      fall back to its hard-coded defaults via :func:`deep_get`.
    - Raises ``RuntimeError`` when PyYAML is unavailable.
    - Raises ``ValueError`` when the YAML payload is not a mapping.
    """
    settings_path = resolve_settings_path(path)
    if not settings_path.exists():
        return {}

    if _yaml is None:
        raise RuntimeError("PyYAML is required to read runtime settings")

    with settings_path.open("r", encoding="utf-8") as file_obj:
        payload = _yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid settings format in {settings_path}: expected a mapping"
        )

    return payload


def deep_get(payload: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def to_str(value: Any, default: str | None = "") -> str | None:
    if value is None:
        return default
    return str(value)


def resolve_local_path(base_dir: Path, value: Any, default_relative: str) -> Path:
    raw_val = to_str(value, default=default_relative)
    raw = raw_val.strip() if raw_val else default_relative
    if not raw:
        raw = default_relative
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path
