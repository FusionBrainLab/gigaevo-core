"""Canonical JSON codec + content-hash computation.

The dataplane has one and only one way to serialise a value for the
wire and one and only one way to compute its content hash. Both
operations are *deterministic* — same logical input produces the same
bytes on every Python version and every host.

Why a custom canonicaliser instead of orjson directly:

    - ``sort_keys=True`` gives stable key order at every nesting level.
    - Floats use ``repr()`` (Python's shortest-round-trip representation),
      not the platform's libc ``%g``.
    - Sets / frozensets are encoded as JSON arrays sorted by canonical
      bytes — heterogeneous-element sets (e.g. ``{1, "1"}``) get a
      deterministic order without relying on ``str(value)`` collisions.
    - Tuples become JSON arrays (no special encoding).
    - UUIDs, datetimes, Decimals, Paths, Enums all have canonical
      encodings; unknown custom types fail closed with
      :class:`CanonicalEncodingError`.
    - ``ensure_ascii=False`` keeps non-ASCII text as-is; lone surrogate
      code points raise rather than being silently sanitised.
    - Integers outside the signed-64-bit range raise — Python ints are
      unbounded but the Lua / JSON consumers downstream truncate at
      2^53 (number) or 2^63 (integer) silently. Refusing the value is
      better than producing a hash that any consumer can disagree with.

``compute_content_hash_hex`` is also mirrored on the Redis side inside
the ``emit_event`` Lua script; a debug-only Python-side recomputation
cross-checks that both sides agree.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
import enum
import hashlib
import json
import pathlib
from typing import Any, Final
from uuid import UUID

from pydantic import BaseModel

from .errors import CanonicalEncodingError

# Signed 64-bit range. Values outside this window cannot be represented
# losslessly by Lua's integer type (Redis 7+), most JSON parsers in
# strongly-typed languages, or JavaScript's Number above 2^53. Treating
# them as canonical-encodable would silently break content-hash agreement
# across consumers — refuse instead.
_INT64_MIN: Final[int] = -(2**63)
_INT64_MAX: Final[int] = 2**63 - 1


# ── canonical JSON ────────────────────────────────────────────────────


def _canonical_default(obj: Any) -> Any:
    """JSON encoder fallback for types ``json.dumps`` cannot serialise.

    Each branch produces a canonical, order-independent representation.
    Unknown types fail closed with :class:`CanonicalEncodingError` so a
    silently-incorrect serialisation cannot land on the wire.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, bytes | bytearray | memoryview):
        return bytes(obj).hex()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, _dt.time):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        # Normalise so Decimal("1.5") and Decimal("1.50") hash identically;
        # collapse signed zero so Decimal("-0") == Decimal("0") on the wire.
        if obj.is_finite():
            if obj.is_zero():
                return "0"
            return format(obj.normalize(), "f")
        raise CanonicalEncodingError(
            type_name="Decimal",
            reason=f"non-finite Decimal ({obj}) is not JSON-encodable",
        )
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, set | frozenset):
        return _canonical_sorted_set(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise CanonicalEncodingError(
        type_name=type(obj).__name__,
        reason="no canonical encoder registered",
    )


def _canonical_sorted_set(items: set[Any] | frozenset[Any]) -> list[Any]:
    """Order set elements by canonical bytes, returning the originals.

    Sorting on the encoded form gives a total order regardless of the
    runtime types — ``{1, "1"}`` produces ``[1, "1"]`` deterministically
    even though ``str(1) == str("1")`` collides on a naive ``key=str`` sort.
    """
    keyed = [(encode_canonical(element), element) for element in items]
    keyed.sort(key=lambda pair: pair[0])
    return [original for _, original in keyed]


def _check_int_range(payload: Any, _seen: set[int] | None = None) -> None:
    """Recursively reject integer values outside the signed-64-bit range.

    Runs before ``json.dumps`` so the failure surfaces as
    :class:`CanonicalEncodingError` rather than as opaque downstream
    truncation. Only JSON-native containers (dict/list/tuple) are
    traversed; ``_seen`` defends against circular references.
    """
    if isinstance(payload, bool):
        # bool is a subclass of int; True / False are always in range.
        return
    if isinstance(payload, int):
        if not (_INT64_MIN <= payload <= _INT64_MAX):
            raise CanonicalEncodingError(
                type_name="int",
                reason=(
                    f"integer {payload} is outside the signed-64-bit range "
                    f"[{_INT64_MIN}, {_INT64_MAX}]; downstream consumers "
                    "(Lua, JS, strongly-typed parsers) would truncate"
                ),
            )
        return
    if not isinstance(payload, dict | list | tuple):
        return
    seen = _seen if _seen is not None else set()
    payload_id = id(payload)
    if payload_id in seen:
        # Cycle: ``json.dumps`` will raise "Circular reference detected".
        return
    seen.add(payload_id)
    try:
        if isinstance(payload, dict):
            for value in payload.values():
                _check_int_range(value, seen)
        else:
            for item in payload:
                _check_int_range(item, seen)
    finally:
        seen.discard(payload_id)


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
        _check_int_range(payload)
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_canonical_default,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except CanonicalEncodingError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalEncodingError(
            type_name=type(payload).__name__,
            reason=str(exc),
        ) from exc


def decode_canonical(raw: bytes | str) -> Any:
    """Decode canonical JSON. Lone surrogate code points raise.

    Lone surrogates or malformed byte sequences fail closed with
    :class:`CanonicalEncodingError`; both ``bytes`` and ``str`` inputs
    surface the same error type so callers can write one ``except`` branch.
    """
    try:
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="strict")
        else:
            # Re-encode via UTF-8 strict so lone surrogates raise the same
            # error type as the bytes path.
            text = raw.encode("utf-8", errors="strict").decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as exc:
        raise CanonicalEncodingError(
            type_name="bytes" if isinstance(raw, bytes) else "str",
            reason=str(exc),
        ) from exc


# ── content hash ──────────────────────────────────────────────────────


_HASH_VERSION_SEP: Final[bytes] = b"|"


def compute_content_hash_hex(payload: Any, *, schema_version: int) -> str:
    """Stable 64-char hex sha256 of ``(schema_version, canonical(payload))``.

    The schema_version is prefixed inside the hash input so two
    structurally-identical payloads at different schema versions hash
    differently — replay across a schema upgrade still distinguishes
    them.
    """
    if schema_version < 1:
        raise ValueError(f"schema_version must be >= 1, got {schema_version}")
    h = hashlib.sha256()
    h.update(f"v{schema_version}".encode("ascii"))
    h.update(_HASH_VERSION_SEP)
    h.update(encode_canonical(payload))
    return h.hexdigest()


__all__ = [
    "compute_content_hash_hex",
    "decode_canonical",
    "encode_canonical",
]
