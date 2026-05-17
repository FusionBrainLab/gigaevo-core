"""DataPlane error hierarchy.

Every error crossing the dataplane boundary is a typed subclass of
:class:`DataPlaneError`. Subclasses carrying structured fields inherit
from :class:`_StructuredError`, which populates ``Exception.args`` from
the dataclass fields so ``str(err)`` shows the failure detail.

Structured fields may opt into redaction via
``dataclasses.field(metadata={"redact": True})`` — the field name still
appears in ``str(err)``, the value is replaced with ``<redacted>``.

Callers can ``except`` on a base class or pattern-match on
:class:`gigaevo.dataplane.models.Result` (``Err`` variant).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

# Sentinel placed in field metadata to mark the value as sensitive. The
# field name remains visible in ``str(err)``; the value is replaced with
# :data:`_REDACTED_PLACEHOLDER` so secrets cannot leak through tracebacks
# or log aggregators that capture ``str(exc)``.
REDACT_META_KEY: Final[str] = "redact"
_REDACTED_PLACEHOLDER: Final[str] = "<redacted>"

# Cap the repr of any single field value so a runaway long string (e.g. a
# decoded payload preview) cannot blow up traceback output. The cap is
# generous enough to keep human-readable detail, tight enough to stay
# bounded in pathological cases.
_FIELD_REPR_MAX_CHARS: Final[int] = 256
_FIELD_REPR_ELLIPSIS: Final[str] = "...<truncated>"


def _format_field(value: Any, *, redact: bool) -> str:
    """Format one field value for inclusion in ``str(err)``.

    Redacted values show :data:`_REDACTED_PLACEHOLDER`. Non-redacted
    reprs are truncated at :data:`_FIELD_REPR_MAX_CHARS`. A field whose
    ``__repr__`` raises renders as ``<repr-failed>`` so an error class
    always produces a usable string.
    """
    if redact:
        return _REDACTED_PLACEHOLDER
    try:
        rendered = repr(value)
    except BaseException as repr_exc:  # noqa: BLE001 - defensive boundary
        return f"<repr-failed: {type(repr_exc).__name__}>"
    if len(rendered) > _FIELD_REPR_MAX_CHARS:
        keep = _FIELD_REPR_MAX_CHARS - len(_FIELD_REPR_ELLIPSIS)
        rendered = rendered[:keep] + _FIELD_REPR_ELLIPSIS
    return rendered


class DataPlaneError(Exception):
    """Bare ``Exception`` base — for ad-hoc ``raise DataPlaneError("msg")``
    and as the catch-all for callers that want one ``except`` clause.

    Structured subclasses inherit from :class:`_StructuredError`.
    """


class _StructuredError(DataPlaneError):
    """Mixin that wires ``Exception.args`` from the dataclass fields.

    ``@dataclass.__init__`` does not call ``super().__init__``;
    ``__post_init__`` here populates ``Exception.args`` so tracebacks
    stay informative. Subclasses MUST be
    ``@dataclass(slots=True, frozen=True)`` with hashable fields — the
    :meth:`__init_subclass__` hook rejects mutable-container field types
    by name (string-annotation-aware best effort, not a full type check).
    """

    _BANNED_FIELD_TYPE_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"list", "dict", "set", "bytearray"}
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not dataclasses.is_dataclass(cls):
            # Non-dataclass intermediate bases (abstract markers) are fine.
            return
        for f in dataclasses.fields(cls):
            type_repr = (
                f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
            )
            banned = cls._BANNED_FIELD_TYPE_NAMES
            if any(banned_name in type_repr for banned_name in banned):
                raise TypeError(
                    f"{cls.__name__}.{f.name}: structured error fields must be "
                    f"hashable; declared type {type_repr!r} contains a mutable "
                    "container. Use a tuple / frozenset / Mapping[str, ...] alias."
                )

    def __post_init__(self) -> None:
        try:
            field_list = dataclasses.fields(self)  # type: ignore[arg-type]  # subclasses are dataclasses
        except TypeError:
            # Non-dataclass subclass (e.g. raw ``DataPlaneError("msg")``).
            return
        parts: list[str] = []
        for f in field_list:
            try:
                value = getattr(self, f.name)
            except AttributeError:
                # Slot declared but not initialised; surface in str(err).
                parts.append(f"{f.name}=<uninitialised>")
                continue
            redact = bool(f.metadata.get(REDACT_META_KEY, False))
            parts.append(f"{f.name}={_format_field(value, redact=redact)}")
        Exception.__init__(self, ", ".join(parts))


# ── lifecycle ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class StartupError(_StructuredError):
    """Raised during ``DataPlane.startup`` if Redis is unreachable, scripts
    fail to load, or any other initial-state precondition is violated."""

    reason: str


@dataclass(slots=True, frozen=True)
class ShutdownError(_StructuredError):
    """Raised if ``DataPlane.shutdown`` cannot complete cleanly."""

    reason: str


@dataclass(slots=True, frozen=True)
class NotStartedError(_StructuredError):
    """A coordinator method was called before ``startup()`` completed."""

    method: str


# ── deadlines ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class DeadlineExceeded(_StructuredError):
    """A coordinator call exceeded its monotonic deadline."""

    elapsed_s: float
    budget_s: float


# ── script registry ───────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ScriptLostError(_StructuredError):
    """``EVALSHA`` returned ``NOSCRIPT`` twice in a row.

    First ``NOSCRIPT`` triggers a reload-and-retry inside
    :class:`gigaevo.dataplane.scripts.LuaRegistry`; a second immediate
    failure surfaces here as a loud error rather than a silent retry.
    """

    script_name: str


@dataclass(slots=True, frozen=True)
class ScriptNotRegisteredError(_StructuredError):
    """``LuaRegistry.evalsha`` was called for a name with no registered source."""

    script_name: str


# ── state machines ────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TransitionError(_StructuredError):
    """A state-machine transition was rejected.

    ``kind`` is one of ``"stale" | "illegal" | "duplicate" | "invalid"
    | "unknown"``; classmethod constructors below produce the
    appropriate variant.

    The ``"invalid"`` variant covers caller-side input that the Lua
    script could not act on (malformed patch JSON, corrupt persisted
    blob); it is distinct from ``"illegal"`` (the transition pair is
    rejected by the FSM table) and from ``"unknown"`` (a server-side
    status the wrapper did not anticipate).
    """

    kind: str
    detail: str

    @classmethod
    def stale(cls, detail: str) -> TransitionError:
        return cls(kind="stale", detail=detail)

    @classmethod
    def illegal(cls, detail: str) -> TransitionError:
        return cls(kind="illegal", detail=detail)

    @classmethod
    def duplicate(cls, detail: str) -> TransitionError:
        return cls(kind="duplicate", detail=detail)

    @classmethod
    def invalid(cls, detail: str) -> TransitionError:
        return cls(kind="invalid", detail=detail)

    @classmethod
    def unknown(cls, status: str, payload: str) -> TransitionError:
        return cls(kind="unknown", detail=f"{status}: {payload}")


@dataclass(slots=True, frozen=True)
class StaleReadError(_StructuredError):
    """A read returned a value older than the caller's freshness floor."""

    observed_epoch: int
    observed_generation: int
    min_epoch: int
    min_generation: int


@dataclass(slots=True, frozen=True)
class EliteInvalidError(_StructuredError):
    """An elite-swap candidate was rejected at the script input boundary.

    Distinct from a ``rejected`` outcome (which carries the surviving
    occupant); this variant signals the candidate could not be compared
    at all — non-finite score, empty cell key, empty candidate id, or
    similar caller-side malformations.
    """

    detail: str


# ── locks ─────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class LockHeld(_StructuredError):
    """``acquire_instance_lock`` failed because the lock is currently held.

    ``holder`` is lease-token-like and redacted from ``str(err)``; the
    attribute itself is accessible on the instance.
    """

    key: str
    holder: str | None = field(metadata={REDACT_META_KEY: True})


@dataclass(slots=True, frozen=True)
class LockLost(_StructuredError):
    """A renewal or release found the lock no longer owned by us."""

    key: str


# ── tokens ────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TokenAlreadyConsumed(_StructuredError):
    """A move-only ``Token[Tag]`` was passed to a consuming method twice.

    ``tag_repr`` may carry caller-derived identifiers; it is redacted
    from ``str(err)``.
    """

    tag_repr: str = field(metadata={REDACT_META_KEY: True})


@dataclass(slots=True, frozen=True)
class TokenNotPickleable(_StructuredError):
    """Tokens are linear; pickle would silently duplicate the witness."""

    tag_repr: str = field(metadata={REDACT_META_KEY: True})


@dataclass(slots=True, frozen=True)
class TokenTagCollisionError(_StructuredError):
    """``mint_split`` / ``mint_split_n`` was passed colliding child tags.

    Linear-permission disjointness requires distinct sub-tags; minting
    duplicates would produce two tokens claiming the same subspace.
    """

    duplicate_tag_repr: str = field(metadata={REDACT_META_KEY: True})


# ── content hash ──────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ContentHashMismatchError(_StructuredError):
    """Client-computed and server-recomputed content hashes disagree."""

    expected_hex: str
    actual_hex: str


# ── schema ────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class SchemaVersionMissingError(_StructuredError):
    """A persisted blob lacked the required ``schema_version`` field."""

    type_name: str
    raw_preview: str


@dataclass(slots=True, frozen=True)
class UpcasterMissingError(_StructuredError):
    """No upcaster chain available to migrate to the current model version."""

    type_name: str
    source_version: int
    target_version: int


# ── canonical encoding ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CanonicalEncodingError(_StructuredError):
    """Canonical JSON refused to encode a value.

    Common causes: surrogate code points in a string, NaN/Inf float,
    integer outside the int64 range, custom type with no registered
    encoder.
    """

    type_name: str
    reason: str


# ── exports ───────────────────────────────────────────────────────────


def all_error_types() -> tuple[type[DataPlaneError], ...]:
    """Tuple of every concrete error class, for exhaustive-match checks."""
    return (
        StartupError,
        ShutdownError,
        NotStartedError,
        DeadlineExceeded,
        ScriptLostError,
        ScriptNotRegisteredError,
        TransitionError,
        StaleReadError,
        LockHeld,
        LockLost,
        TokenAlreadyConsumed,
        TokenNotPickleable,
        TokenTagCollisionError,
        ContentHashMismatchError,
        SchemaVersionMissingError,
        UpcasterMissingError,
        CanonicalEncodingError,
    )


__all__ = [
    "REDACT_META_KEY",
    "CanonicalEncodingError",
    "ContentHashMismatchError",
    "DataPlaneError",
    "DeadlineExceeded",
    "LockHeld",
    "LockLost",
    "NotStartedError",
    "SchemaVersionMissingError",
    "ScriptLostError",
    "ScriptNotRegisteredError",
    "ShutdownError",
    "StaleReadError",
    "StartupError",
    "TokenAlreadyConsumed",
    "TokenNotPickleable",
    "TokenTagCollisionError",
    "TransitionError",
    "UpcasterMissingError",
    "all_error_types",
]
