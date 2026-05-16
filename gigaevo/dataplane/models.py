"""Type-system primitives the dataplane uses everywhere.

The wrappers below carry invariants that turn ordinary function
signatures into invariant declarations:

    Versioned[T]      — epoch/generation freshness witness; admission gate.
    Sourced[T, S]     — provenance tag (Local / Cached / Replayed / Gossiped).
    Result[T, E]      — discriminated return; no exception crosses the
                         dataplane boundary except KeyboardInterrupt /
                         CancelledError.
    Monotonic[T]      — counter that rejects retrograde writes at runtime.
    HlcTimestamp      — hybrid logical clock pair (physical_ns, counter).

:class:`Token` (move-only permission) lives in :mod:`permissions` to
avoid a circular import.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .errors import StaleReadError


class _Comparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...
    def __le__(self, other: Any, /) -> bool: ...


# ── Versioned ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Versioned[T]:
    """A value witnessed by an ``(epoch, generation)`` pair.

    Every cached / projected read carries one. Callers that require
    freshness pass ``min_epoch=`` / ``min_generation=``; values below
    either floor raise :class:`StaleReadError`.

    The two axes compose pointwise via the product lattice: epoch is
    bumped by the global epoch counter on every state-changing
    coordinator call; generation is per-aggregate.
    """

    value: T
    epoch: int
    generation: int

    def is_at_least(self, min_epoch: int, min_generation: int) -> bool:
        """Return True iff ``(self.epoch, self.generation) ⩾ floor``."""
        return self.epoch >= min_epoch and self.generation >= min_generation

    def require_at_least(self, min_epoch: int, min_generation: int) -> Versioned[T]:
        """Return ``self`` if fresh enough; otherwise raise :class:`StaleReadError`."""
        if not self.is_at_least(min_epoch, min_generation):
            raise StaleReadError(
                observed_epoch=self.epoch,
                observed_generation=self.generation,
                min_epoch=min_epoch,
                min_generation=min_generation,
            )
        return self

    def combine_max(self, other: Versioned[T]) -> Versioned[T]:
        """Pointwise lattice JOIN.

        Picks the side with the strictly-greater ``(epoch, generation)``
        tuple. On a tie, returns ``self`` deterministically (stable
        choice for re-application).
        """
        if (other.epoch, other.generation) > (self.epoch, self.generation):
            return other
        return self

    def map(self, fn: Any) -> Versioned[Any]:
        """Apply ``fn`` to the value, preserve epoch / generation.

        Convenient when the read pipeline transforms the inner value
        (e.g. project a single field from a Program) without losing the
        freshness witness.
        """
        return Versioned(
            value=fn(self.value), epoch=self.epoch, generation=self.generation
        )


# ── Sourced ───────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Sourced[T, S]:
    """Phantom provenance wrapper. ``S`` is a ``Literal`` tag, not an instance.

    Function signatures use the type aliases below to require / reject
    specific provenance::

        def admit(value: LocalValue[Program]) -> ...

    Refuses to accept ``CachedValue[Program]`` at mypy-strict time.
    """

    value: T

    def retag(self, _new_tag_marker: Any) -> Sourced[T, Any]:
        """Re-tag the value with a different provenance marker.

        The marker argument is type-only at the call site (callers do
        ``cached.retag(Literal["local"])``). Runtime cost: one fresh
        ``Sourced`` wrapper.
        """
        return Sourced(value=self.value)


type LocalValue[T] = Sourced[T, Literal["local"]]
type CachedValue[T] = Sourced[T, Literal["cached"]]
type ReplayedValue[T] = Sourced[T, Literal["replayed"]]
type GossipedValue[T] = Sourced[T, Literal["gossiped"]]
type ExternalValue[T] = Sourced[T, Literal["external"]]
type SanitizedValue[T] = Sourced[T, Literal["sanitized"]]


# ── Result ────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Ok[T]:
    """Success branch of :data:`Result`."""

    value: T

    def is_ok(self) -> Literal[True]:
        return True

    def is_err(self) -> Literal[False]:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(slots=True, frozen=True)
class Err[E]:
    """Failure branch of :data:`Result`."""

    error: E

    def is_ok(self) -> Literal[False]:
        return False

    def is_err(self) -> Literal[True]:
        return True

    def unwrap(self) -> Any:
        """Raise the underlying error.

        ``Err.error`` is usually a :class:`gigaevo.dataplane.errors.DataPlaneError`
        (which is an ``Exception``) so raising directly works. Non-
        exception ``E`` types are wrapped in ``RuntimeError``.
        """
        err = self.error
        if isinstance(err, BaseException):
            raise err
        raise RuntimeError(f"Err.unwrap on non-exception payload: {err!r}")


type Result[T, E] = Ok[T] | Err[E]
"""Discriminated return; callers ``match result: case Ok(v): ... case Err(e): ...``."""


# ── Monotonic ─────────────────────────────────────────────────────────


class Monotonic[CT: _Comparable]:
    """A counter / step / version that only advances.

    ``advance(new)`` requires ``current <= new``; a retrograde
    assignment raises :class:`ValueError` immediately. Equality counts
    as a valid advance (idempotent re-emit).
    """

    __slots__ = ("_value",)

    def __init__(self, initial: CT) -> None:
        self._value: CT = initial

    def peek(self) -> CT:
        return self._value

    def advance(self, new_value: CT) -> None:
        if new_value < self._value:
            raise ValueError(
                f"Monotonic violation: {new_value!r} < current {self._value!r}"
            )
        self._value = new_value

    def bump(self) -> CT:
        """Increment by one; only valid when ``CT`` supports ``+ 1``."""
        self._value = self._value + 1  # type: ignore[operator]
        return self._value

    def __repr__(self) -> str:
        return f"Monotonic({self._value!r})"


# ── HlcTimestamp ──────────────────────────────────────────────────────


_U64_MAX = (1 << 64) - 1
_U32_MAX = (1 << 32) - 1


@dataclass(slots=True, frozen=True, order=True)
class HlcTimestamp:
    """Hybrid logical clock timestamp.

    ``(physical_ns, counter)`` packed pair carried on events.
    Lexicographic order on the pair is the causality order. Field order
    matters for :func:`dataclasses.dataclass(order=True)` — physical
    nanoseconds dominate, counter breaks ties.
    """

    physical_ns: int
    counter: int

    def __post_init__(self) -> None:
        if not (0 <= self.physical_ns <= _U64_MAX):
            raise ValueError(
                f"HlcTimestamp.physical_ns out of uint64 range: {self.physical_ns}"
            )
        if not (0 <= self.counter <= _U32_MAX):
            raise ValueError(
                f"HlcTimestamp.counter out of uint32 range: {self.counter}"
            )

    def pack_hex(self) -> str:
        """Big-endian 32-hex-char encoding ``physical_ns||counter||0``.

        The trailing 32 zero bits match the wire format used by the
        Lua side of the LWW-register write.
        """
        return f"{self.physical_ns:016x}{self.counter:08x}00000000"

    @classmethod
    def unpack_hex(cls, packed: str) -> HlcTimestamp:
        if len(packed) != 32:
            raise ValueError(
                f"HlcTimestamp.unpack_hex: expected 32 hex chars, got {len(packed)}"
            )
        physical = int(packed[0:16], 16)
        counter = int(packed[16:24], 16)
        return cls(physical_ns=physical, counter=counter)


# Silence unused-import warning for dataclasses; it's used by Pydantic-shaped tests.
_ = dataclasses


__all__ = [
    "CachedValue",
    "Err",
    "ExternalValue",
    "GossipedValue",
    "HlcTimestamp",
    "LocalValue",
    "Monotonic",
    "Ok",
    "ReplayedValue",
    "Result",
    "SanitizedValue",
    "Sourced",
    "Versioned",
]
