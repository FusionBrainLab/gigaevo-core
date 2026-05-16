"""Canonical JSON codec + content-hash computation.

The dataplane has one and only one way to serialise a value for the
wire and one and only one way to compute its content hash. Both
operations are *deterministic* — same logical input produces the same
bytes on every Python version and every host.

Why a custom canonicaliser instead of orjson directly:
    - ``sort_keys=True`` gives stable key order at every nesting level.
    - Floats use ``repr()`` (Python's shortest-round-trip representation),
      not the platform's libc ``%g``.
    - Sets / frozensets are encoded as sorted JSON arrays.
    - Tuples become JSON arrays (no special encoding).
    - UUIDs, datetimes, Decimals, Paths, Enums all have canonical
      encodings; unknown custom types fail closed with
      :class:`CanonicalEncodingError`.
    - ``ensure_ascii=False`` keeps non-ASCII text as-is; lone surrogate
      code points raise rather than being silently sanitised.

``compute_content_hash`` is also mirrored on the Redis side inside
``emit_event.lua``; a debug-only Python-side recomputation cross-checks
that both sides agree.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
import enum
import hashlib
import json
import pathlib
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from .errors import CanonicalEncodingError

# ── canonical JSON ────────────────────────────────────────────────────


def _canonical_default(obj: Any) -> Any:
    """JSON encoder fallback for types ``json.dumps`` cannot serialise.

    Each branch produces a canonical, order-independent representation.
    Unknown types fail closed with :class:`CanonicalEncodingError` so a
    silently-incorrect serialisation cannot land on the wire.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, _dt.time):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    raise CanonicalEncodingError(
        type_name=type(obj).__name__,
        reason="no canonical encoder registered",
    )


def encode_canonical(payload: Any) -> bytes:
    """Encode ``payload`` to canonical JSON bytes.

    The dataplane uses this for every persisted-blob write and every
    event emission. The output is UTF-8 with sorted keys, no
    insignificant whitespace, and no NaN / +Inf / -Inf (JSON forbids
    them and silently emitting non-standard JSON would break consumers).
    """
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_canonical_default,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        if isinstance(exc, CanonicalEncodingError):
            raise
        raise CanonicalEncodingError(
            type_name=type(payload).__name__,
            reason=str(exc),
        ) from exc


def decode_canonical(raw: bytes | str) -> Any:
    """Decode canonical JSON. Lone surrogate code points raise.

    The dataplane forbids surrogate-replay behaviour; if the input
    contains a lone surrogate it raises :class:`UnicodeDecodeError`
    rather than silently substituting U+FFFD.
    """
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8", errors="strict"))
    # Strings via str.encode roundtrip detect lone surrogates the same
    # way as bytes inputs.
    return json.loads(raw.encode("utf-8", errors="strict").decode("utf-8"))


# ── content hash ──────────────────────────────────────────────────────


_HASH_VERSION_SEP = b"|"


def compute_content_hash(payload: Any, *, schema_version: int) -> bytes:
    """Stable 32-byte sha256 of ``(schema_version, canonical(payload))``.

    The schema_version is prefixed inside the hash input so two
    structurally-identical payloads at different schema versions hash
    differently — replay across a schema upgrade still distinguishes
    them.

    Returns 32 raw bytes (sha256 digest). Use ``.hex()`` for storage in
    Redis hash fields or :func:`compute_content_hash_hex` directly.
    """
    if schema_version < 1:
        raise ValueError(f"schema_version must be >= 1, got {schema_version}")
    h = hashlib.sha256()
    h.update(f"v{schema_version}".encode("ascii"))
    h.update(_HASH_VERSION_SEP)
    h.update(encode_canonical(payload))
    return h.digest()


def compute_content_hash_hex(payload: Any, *, schema_version: int) -> str:
    """Convenience: 64-char hex of :func:`compute_content_hash`."""
    return compute_content_hash(payload, schema_version=schema_version).hex()


__all__ = [
    "compute_content_hash",
    "compute_content_hash_hex",
    "decode_canonical",
    "encode_canonical",
]
