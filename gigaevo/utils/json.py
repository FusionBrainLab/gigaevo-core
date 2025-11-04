from __future__ import annotations

from typing import Any, Union

__all__ = ["dumps", "loads", "json"]

try:
    import orjson as _backend  # type: ignore

    def dumps(obj: Any) -> str:  # type: ignore[override]
        """Serialize *obj* to a ``str`` using orjson (bytes → str)."""
        return _backend.dumps(obj).decode()

    loads = _backend.loads  # type: ignore
    json = _backend  # type: ignore

except ModuleNotFoundError:  # pragma: no cover – dev/test envs without orjson
    import json as _backend  # type: ignore

    def dumps(obj: Any) -> str:  # type: ignore[override]
        """Serialize *obj* to a ``str`` using the stdlib *json* module."""
        return _backend.dumps(obj)

    def loads(data: Union[str, bytes, bytearray]):  # type: ignore[override]
        """Deserialize *data* using the stdlib *json* module."""
        return _backend.loads(data)

    json = _backend  # type: ignore
