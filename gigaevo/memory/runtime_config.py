from __future__ import annotations

import os
from pathlib import Path
import types
from typing import Any

_yaml: types.ModuleType | None

try:
    import yaml

    _yaml = yaml
except Exception:  # pragma: no cover - defensive fallback
    _yaml = None


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_DEFAULT_SETTINGS_PATHS = (
    _PROJECT_ROOT / "config" / "memory.yaml",
    _MODULE_DIR / "config.yaml",
)


def resolve_settings_path(path: str | Path | None = None) -> Path:
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
    raw = to_str(value, default=default_relative).strip()
    if not raw:
        raw = default_relative
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path
