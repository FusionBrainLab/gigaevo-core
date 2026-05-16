"""Type-system primitives the dataplane uses everywhere.

The wrappers below carry invariants that turn ordinary function
signatures into invariant declarations:

    Versioned[T]      — epoch/generation freshness witness; admission gate.
    Sourced[T, S]     — provenance tag (Local / Cached / Replayed / Gossiped).
    Result[T, E]      — discriminated return; no exception crosses the
                         dataplane boundary except KeyboardInterrupt /
                         CancelledError.
    Monotonic[T]      — counter that rejects retrograde writes at runtime.
    MonotonicCounter  — integer specialisation of Monotonic with ``bump()``.
    HlcTimestamp      — hybrid logical clock pair (physical_ns, counter).

:class:`Token` (move-only permission) lives in :mod:`permissions` to
avoid a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, Protocol

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

    Invariants enforced in ``__post_init__``:
        - ``epoch >= 0``
        - ``generation >= 0``

    Both axes are unsigned counters in the persisted representation;
    negative values would round-trip incorrectly and are rejected at the
    construction boundary so a corrupted blob fails loudly rather than
    silently lying about freshness.
    """

    value: T
    epoch: int
    generation: int

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(f"Versioned.epoch must be non-negative: {self.epoch}")
        if self.generation < 0:
            raise ValueError(
                f"Versioned.generation must be non-negative: {self.generation}"
            )

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
        """Pointwise lattice JOIN on the ``(epoch, generation)`` pair.

        Picks the side with the strictly-greater ``(epoch, generation)``
        tuple. On a tie, returns ``self`` deterministically (stable
        choice for re-application).

        Comparison is on ``(epoch, generation)`` only — the wrapped
        ``value`` is NOT required to be comparable. ``T`` may be any
        type. This is the right semantics for the dataplane: freshness
        is decided by the witness, not by the value's natural order.
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

        Runtime: this is intentionally a near-no-op — ``S`` is a phantom
        ``Literal`` type parameter with no runtime field, so retagging is
        a fresh wrapper around the same value. The new tag flows through
        the caller's annotation: ``local: LocalValue[int] = cached.retag(...)``
        assigns the call-site type.

        The marker parameter exists so a future strengthening that
        captures the new tag at runtime (e.g. for logging or audit) can
        land without changing call sites. It is otherwise unused.
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

    def unwrap(self) -> NoReturn:
        """Raise the underlying error. Never returns.

        ``Err.error`` is usually a :class:`gigaevo.dataplane.errors.DataPlaneError`
        (which is an ``Exception``) so raising directly works. Non-
        exception ``E`` types are wrapped in ``RuntimeError``.

        Typed ``NoReturn`` because the function always raises; mypy will
        consider any code after ``r.unwrap()`` unreachable, which is the
        intended contract.
        """
        err = self.error
        if isinstance(err, BaseException):
            raise err
        raise RuntimeError(f"Err.unwrap on non-exception payload: {err!r}")


type Result[T, E] = Ok[T] | Err[E]
"""Discriminated return; callers ``match result: case Ok(v): ... case Err(e): ...``.

The ``E`` parameter is unbounded at the alias level so the type can hold
any payload, but every coordinator-level signature in this package
bounds ``E`` to :class:`gigaevo.dataplane.errors.DataPlaneError` so
callers know which exception classes ``match Err(e)`` can yield.
"""


# ── Monotonic ─────────────────────────────────────────────────────────


class Monotonic[CT: _Comparable]:
    """A value (counter / step / version) that only advances.

    ``advance(new)`` requires ``current <= new``; a retrograde
    assignment raises :class:`ValueError` immediately. Equality counts
    as a valid advance (idempotent re-emit).

    The base class supports any ``_Comparable`` type. For integer
    counters with a ``bump()`` helper, use :class:`MonotonicCounter`.
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"


class MonotonicCounter(Monotonic[int]):
    """Integer specialisation of :class:`Monotonic` with ``bump()``.

    The base ``Monotonic`` class is typed against ``_Comparable`` so it
    works for any ordered type (e.g. ``HlcTimestamp``, ``str``).
    Incrementing by one is only meaningful for numerics; isolating it
    here lets mypy reject ``Monotonic("v1").bump()`` at type-check time
    without a runtime ``+ 1`` failure mode hiding behind a stringly-typed
    counter.
    """

    __slots__ = ()

    def bump(self) -> int:
        """Increment by one and return the new value."""
        self._value = self._value + 1
        return self._value


# ── HlcTimestamp ──────────────────────────────────────────────────────


_U64_MAX: Final[int] = (1 << 64) - 1
_U32_MAX: Final[int] = (1 << 32) - 1
_HLC_TRAILING_PAD: Final[str] = "00000000"
_HLC_HEX_LEN: Final[int] = 32


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
        """Big-endian 32-hex-char encoding ``physical_ns||counter||pad``.

        Layout (32 hex chars, big-endian):
            - chars 0-15  : ``physical_ns`` as 16-hex (uint64)
            - chars 16-23 : ``counter`` as 8-hex (uint32)
            - chars 24-31 : reserved-zero pad (8-hex)

        The trailing 32 bits are reserved for a future shard-id /
        process-id field that the Lua write path may inject server-side.
        Today they are written and required to be zero on the wire so
        the format is forward-compatible: a future reader can split the
        field at the same offset without an encoding flag day.
        :meth:`unpack_hex` rejects non-zero trailing bits.
        """
        return f"{self.physical_ns:016x}{self.counter:08x}{_HLC_TRAILING_PAD}"

    @classmethod
    def unpack_hex(cls, packed: str) -> HlcTimestamp:
        """Decode a packed HLC string. Rejects malformed inputs eagerly.

        Raises :class:`ValueError` on any of:
            - length != 32
            - non-hex characters
            - non-zero trailing pad bits (reserved field invariant)
            - decoded ``physical_ns`` or ``counter`` out of range
        """
        if len(packed) != _HLC_HEX_LEN:
            raise ValueError(
                "HlcTimestamp.unpack_hex: expected "
                f"{_HLC_HEX_LEN} hex chars, got {len(packed)}"
            )
        trailing = packed[24:32]
        if trailing != _HLC_TRAILING_PAD:
            raise ValueError(
                "HlcTimestamp.unpack_hex: trailing 8 hex chars must be "
                f"{_HLC_TRAILING_PAD!r} (reserved field), got {trailing!r}"
            )
        try:
            physical = int(packed[0:16], 16)
            counter = int(packed[16:24], 16)
        except ValueError as exc:
            raise ValueError(
                f"HlcTimestamp.unpack_hex: non-hex characters in {packed!r}"
            ) from exc
        return cls(physical_ns=physical, counter=counter)


__all__ = [
    "CachedValue",
    "Err",
    "ExternalValue",
    "GossipedValue",
    "HlcTimestamp",
    "LocalValue",
    "Monotonic",
    "MonotonicCounter",
    "Ok",
    "ReplayedValue",
    "Result",
    "SanitizedValue",
    "Sourced",
    "Versioned",
]
