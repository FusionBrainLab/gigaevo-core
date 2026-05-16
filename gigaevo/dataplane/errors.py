"""DataPlane error hierarchy.

Every error that crosses the dataplane boundary is a typed subclass of
:class:`DataPlaneError`. Subclasses that carry structured fields inherit
from :class:`_StructuredError`, which populates ``Exception.args`` from
the dataclass fields so ``str(err)`` shows the failure detail.

Callers that want exception-style handling can ``except`` on a base
class. Callers that prefer discriminated returns use
:class:`gigaevo.dataplane.models.Result` and match on ``Err``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


class DataPlaneError(Exception):
    """Bare ``Exception`` base — for ad-hoc ``raise DataPlaneError("msg")``
    and as the catch-all for callers that want one ``except`` clause.

    Structured subclasses inherit from :class:`_StructuredError`.
    """


class _StructuredError(DataPlaneError):
    """Mixin that wires Exception.args from the dataclass fields.

    ``@dataclass.__init__`` does not call ``super().__init__``, so
    ``str(err)`` would otherwise be empty. The ``__post_init__`` hook
    inherited by every subclass formats the fields into ``Exception.args``
    so tracebacks and log messages stay informative.
    """

    def __post_init__(self) -> None:
        try:
            field_list = dataclasses.fields(self)  # type: ignore[arg-type]  # subclasses are dataclasses
        except TypeError:
            return
        parts = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in field_list)
        Exception.__init__(self, parts)


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

    ``kind`` is one of ``"stale" | "illegal" | "duplicate" | "unknown"``;
    classmethod constructors below produce the appropriate variant.
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
    def unknown(cls, status: str, payload: str) -> TransitionError:
        return cls(kind="unknown", detail=f"{status}: {payload}")


@dataclass(slots=True, frozen=True)
class StaleReadError(_StructuredError):
    """A read returned a value older than the caller's freshness floor."""

    observed_epoch: int
    observed_generation: int
    min_epoch: int
    min_generation: int


# ── locks ─────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class LockHeld(_StructuredError):
    """``acquire_instance_lock`` failed because the lock is currently held."""

    key: str
    holder: str | None


@dataclass(slots=True, frozen=True)
class LockLost(_StructuredError):
    """A renewal or release found the lock no longer owned by us."""

    key: str


# ── tokens ────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TokenAlreadyConsumed(_StructuredError):
    """A move-only ``Token[Tag]`` was passed to a consuming method twice."""

    tag_repr: str


@dataclass(slots=True, frozen=True)
class TokenNotPickleable(_StructuredError):
    """Tokens are linear; pickle would silently duplicate the witness."""

    tag_repr: str


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
    custom type with no registered encoder.
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
        ContentHashMismatchError,
        SchemaVersionMissingError,
        UpcasterMissingError,
        CanonicalEncodingError,
    )


# Re-export Any so ``Any`` users don't need a separate import.
_: Any = None  # type: ignore[assignment]  # purely to silence lint when Any is unused above
del _


__all__ = [
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
    "TransitionError",
    "UpcasterMissingError",
    "all_error_types",
]
