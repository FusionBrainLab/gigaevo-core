"""DataPlane error hierarchy.

Every error that crosses the dataplane boundary is a typed subclass of
:class:`DataPlaneError`. Subclasses that carry structured fields inherit
from :class:`_StructuredError`, which populates ``Exception.args`` from
the dataclass fields so ``str(err)`` shows the failure detail.

Structured fields may opt into redaction via
``dataclasses.field(metadata={"redact": True})`` — the field name still
appears in ``str(err)`` but the value is replaced with ``<redacted>``.
Used on fields that can hold lease tokens, holder identities, or other
material that should not land in logs verbatim.

Callers that want exception-style handling can ``except`` on a base
class. Callers that prefer discriminated returns use
:class:`gigaevo.dataplane.models.Result` and match on ``Err``.
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

    Redacted fields show :data:`_REDACTED_PLACEHOLDER` regardless of the
    underlying value. Non-redacted reprs are truncated at
    :data:`_FIELD_REPR_MAX_CHARS` to keep traceback output bounded.

    A field whose ``__repr__`` raises is rendered as ``<repr-failed>``
    rather than propagating the inner exception — an error class must
    always render to a usable string so failure handlers, log
    aggregators, and tracebacks can attribute the original failure
    instead of crashing inside ``__post_init__``.
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

    ``@dataclass.__init__`` does not call ``super().__init__``, so
    ``str(err)`` would otherwise be empty. The ``__post_init__`` hook
    inherited by every subclass formats the fields into ``Exception.args``
    so tracebacks and log messages stay informative.

    Subclasses MUST be ``@dataclass(slots=True, frozen=True)``. The
    :meth:`__init_subclass__` hook enforces this and rejects any field
    whose declared type is mutable (``list`` / ``dict`` / ``set``); error
    instances are hashable, so their fields must be too.
    """

    # Field types that defeat ``frozen=True`` hashability. The check is
    # type-name based because evaluating string annotations at class
    # creation time is unreliable; this is a guard rail, not a full type
    # check.
    _BANNED_FIELD_TYPE_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"list", "dict", "set", "bytearray"}
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not dataclasses.is_dataclass(cls):
            # Non-dataclass intermediate bases (e.g. abstract markers)
            # are allowed; ``__post_init__`` still tolerates them.
            return
        for f in dataclasses.fields(cls):
            type_repr = (
                f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
            )
            # Strip ``Optional[...]`` / ``X | None`` wrappers crudely for
            # the name check — the goal is to catch obvious offenders, not
            # to reimplement typing.get_type_hints.
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
                # Slot declared but not initialised — pre-empt the
                # AttributeError so ``str(err)`` still produces useful
                # output during partially-constructed-instance teardown.
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
    :class:`gigaevo.dataplane.scripts.LuaRegistry`. A second one in
    immediate succession surfaces here — Redis is in a bad state and the
    caller should not paper over it.
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

    ``holder`` is a lease-token-like identifier; it is treated as
    sensitive and redacted from ``str(err)`` so it cannot leak through
    logs verbatim. The field is still accessible on the instance for
    callers that legitimately need to inspect it.
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
    """Tuple of every concrete error class. Used for exhaustive matching tests."""
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
