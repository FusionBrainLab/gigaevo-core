from __future__ import annotations

import json as _stdlib_json
import types
from typing import Any

__all__ = ["dumps", "loads", "json"]

json: types.ModuleType

try:
    import orjson as _orjson

    def dumps(obj: Any) -> str:
        """Serialize *obj* to a ``str`` using orjson (bytes -> str)."""
        return _orjson.dumps(obj).decode()

    def loads(data: str | bytes | bytearray) -> Any:
        """Deserialize *data* using orjson."""
        return _orjson.loads(data)

    json = _orjson

except ModuleNotFoundError:  # pragma: no cover -- dev/test envs without orjson

    def dumps(obj: Any) -> str:  # type: ignore[misc]  # redefinition for fallback branch
        """Serialize *obj* to a ``str`` using the stdlib *json* module."""
        return _stdlib_json.dumps(obj)

    def loads(data: str | bytes | bytearray) -> Any:  # type: ignore[misc]  # redefinition for fallback branch
        """Deserialize *data* using the stdlib *json* module."""
        return _stdlib_json.loads(data)

    json = _stdlib_json
