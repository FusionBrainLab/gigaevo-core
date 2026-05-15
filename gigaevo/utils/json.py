from __future__ import annotations

import json as _stdlib_json
import types
from typing import Any

from gigaevo.utils.text_sanitize import deep_sanitize_for_json

__all__ = ["dumps", "loads", "json"]

json: types.ModuleType

try:
    import orjson as _orjson

    def dumps(obj: Any) -> str:
        """Serialize *obj* to a ``str`` using orjson (bytes -> str).

        Walks ``obj`` first to replace lone UTF-16 surrogates with U+FFFD.
        orjson (and stdlib ``json``) raise ``UnicodeEncodeError`` on lone
        surrogates; LLM-derived text frequently carries them on the path
        from Triton / CUDA / CUTLASS / Mojo error formatters. The walk is
        a no-op for clean structures.
        """
        return _orjson.dumps(deep_sanitize_for_json(obj)).decode()

    def loads(data: str | bytes | bytearray) -> Any:
        """Deserialize *data* using orjson."""
        return _orjson.loads(data)

    json = _orjson

except ModuleNotFoundError:  # pragma: no cover -- dev/test envs without orjson

    def dumps(obj: Any) -> str:  # type: ignore[misc]  # redefinition for fallback branch
        """Serialize *obj* to a ``str`` using the stdlib *json* module.

        See the orjson branch for the surrogate-scrub rationale.
        """
        return _stdlib_json.dumps(deep_sanitize_for_json(obj))

    def loads(data: str | bytes | bytearray) -> Any:  # type: ignore[misc]  # redefinition for fallback branch
        """Deserialize *data* using the stdlib *json* module."""
        return _stdlib_json.loads(data)

    json = _stdlib_json
