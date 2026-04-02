from __future__ import annotations

import types
from typing import Any

__all__ = ["dumps", "loads", "json"]

json: types.ModuleType

try:
    import orjson

    json = orjson
except ModuleNotFoundError:  # pragma: no cover – dev/test envs without orjson
    import json as _json_stdlib

    def dumps(obj: Any) -> str:  # type: ignore[override]
        """Serialize *obj* to a ``str`` using the stdlib *json* module."""
        return _backend.dumps(obj)  # type: ignore[return-value]


def dumps(obj: Any) -> str:
    """Serialize *obj* to a JSON ``str`` (orjson returns bytes, so decode)."""
    raw = json.dumps(obj)
    return raw.decode() if isinstance(raw, bytes) else raw


def loads(data: str | bytes | bytearray) -> Any:
    """Deserialize *data* from JSON."""
    return json.loads(data)
