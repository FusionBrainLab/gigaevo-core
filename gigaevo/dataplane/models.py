"""Type-system primitives the dataplane uses everywhere.

The wrappers below carry invariants that turn ordinary function
signatures into invariant declarations:

    Versioned[T]      — epoch/generation freshness witness; admission gate.
    Sourced[T, S]     — provenance tag (Local / Cached / Replayed / Gossiped).
    Result[T, E]      — discriminated return; no exception crosses the
                         dataplane boundary except KeyboardInterrupt /
                         CancelledError.
    HlcTimestamp      — hybrid logical clock pair (physical_ns, counter).

:class:`Token` (move-only permission) lives in :mod:`permissions` to
avoid a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn

from .errors import StaleReadError

# ── Versioned ─────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Versioned[T]:
    """A value witnessed by an ``(epoch, generation)`` pair.

    Every cached / projected read carries one. Below-floor reads raise
    :class:`StaleReadError`. ``epoch`` is bumped by the global counter
    on every state-changing call; ``generation`` is per-aggregate. Both
    axes are non-negative (corrupted blobs fail loudly at construction).
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
        """Pointwise lattice JOIN on ``(epoch, generation)``.

        Picks the strictly-greater side; ties return ``self`` (stable for
        re-application). ``T`` is not required to be comparable — freshness
        is decided by the witness, not the value.
        """
        if (other.epoch, other.generation) > (self.epoch, self.generation):
            return other
        return self

    def map(self, fn: Any) -> Versioned[Any]:
        """Apply ``fn`` to the value, preserving the freshness witness."""
        return Versioned(
            value=fn(self.value), epoch=self.epoch, generation=self.generation
        )


# ── Freshness ────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class FreshnessEventual:
    """Accept any persisted value regardless of staleness.

    Read methods consulted with this freshness never raise
    :class:`StaleReadError` on the freshness axis; decoding errors and
    absent keys still surface.
    """


@dataclass(slots=True, frozen=True)
class FreshnessAtLeast:
    """Demand a floor on the ``(epoch, generation)`` lattice.

    Reads return ``Ok(Versioned(...))`` iff component-wise above the
    floor; otherwise ``Err(StaleReadError(...))``. The ``epoch=0``,
    ``generation=0`` default is the trivially-clearable identity floor;
    production callers pin the expected counter explicitly.
    """

    epoch: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(
                f"FreshnessAtLeast.epoch must be non-negative: {self.epoch}"
            )
        if self.generation < 0:
            raise ValueError(
                f"FreshnessAtLeast.generation must be non-negative: {self.generation}"
            )


@dataclass(slots=True, frozen=True)
class FreshnessStrict:
    """Demand the read observes the latest epoch at call time.

    The coordinator snapshots the global epoch counter first; the
    persisted blob's epoch must be ``>=`` that snapshot. A concurrent
    bump between snapshot and blob read raises :class:`StaleReadError`.

    Strongest freshness witness available on a plain GET; the post-write
    return value from :meth:`transition_program_state` is stronger still
    (same atomic round-trip, no declaration needed).
    """


type Freshness = FreshnessEventual | FreshnessAtLeast | FreshnessStrict
"""Discriminated declaration of how stale a read may be.

    * :class:`FreshnessEventual` — no floor.
    * :class:`FreshnessAtLeast(epoch=N, generation=M)` — below-floor
      raises :class:`StaleReadError`.
    * :class:`FreshnessStrict` — re-read the epoch counter first. Two
      round-trips; use only when the caller cannot supply its own floor.
"""


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

        Near-no-op at runtime since ``S`` is phantom; the new tag flows
        through the caller's annotation. The marker parameter is reserved
        for a future runtime-capture extension.
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
        """Raise the underlying error; never returns.

        ``Err.error`` is usually a :class:`DataPlaneError` (Exception
        subclass) and raises directly; non-exception payloads are wrapped
        in ``RuntimeError``. ``NoReturn`` so mypy marks post-call code
        unreachable.
        """
        err = self.error
        if isinstance(err, BaseException):
            raise err
        raise RuntimeError(f"Err.unwrap on non-exception payload: {err!r}")


type Result[T, E] = Ok[T] | Err[E]
"""Discriminated return; callers ``match result: case Ok(v): ... case Err(e): ...``.

``E`` is unbounded at the alias level; every coordinator-level signature
bounds it to :class:`DataPlaneError` so the caller knows the variants.
"""


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

        Layout: chars 0-15 ``physical_ns`` (uint64), 16-23 ``counter``
        (uint32), 24-31 reserved-zero pad. The pad reserves space for a
        future shard/process-id field; :meth:`unpack_hex` rejects non-zero
        trailing bits to keep the format forward-compatible.
        """
        return f"{self.physical_ns:016x}{self.counter:08x}{_HLC_TRAILING_PAD}"

    @classmethod
    def unpack_hex(cls, packed: str) -> HlcTimestamp:
        """Decode a packed HLC string. Rejects malformed inputs eagerly.

        Raises :class:`ValueError` for: length != 32, non-lowercase-hex
        characters, non-zero trailing pad bits, or out-of-range fields.
        Strict lowercase keeps content-hash inputs byte-stable across
        encoders.
        """
        if len(packed) != _HLC_HEX_LEN:
            raise ValueError(
                "HlcTimestamp.unpack_hex: expected "
                f"{_HLC_HEX_LEN} hex chars, got {len(packed)}"
            )
        for ch in packed:
            if not (ch.isdigit() or ("a" <= ch <= "f")):
                raise ValueError(
                    "HlcTimestamp.unpack_hex: non-hex characters "
                    f"(or uppercase hex) at {ch!r} in {packed!r}"
                )
        trailing = packed[24:32]
        if trailing != _HLC_TRAILING_PAD:
            raise ValueError(
                "HlcTimestamp.unpack_hex: trailing 8 hex chars must be "
                f"{_HLC_TRAILING_PAD!r} (reserved field), got {trailing!r}"
            )
        physical = int(packed[0:16], 16)
        counter = int(packed[16:24], 16)
        return cls(physical_ns=physical, counter=counter)


__all__ = [
    "CachedValue",
    "Err",
    "ExternalValue",
    "Freshness",
    "FreshnessAtLeast",
    "FreshnessEventual",
    "FreshnessStrict",
    "GossipedValue",
    "HlcTimestamp",
    "LocalValue",
    "Ok",
    "ReplayedValue",
    "Result",
    "SanitizedValue",
    "Sourced",
    "Versioned",
]
