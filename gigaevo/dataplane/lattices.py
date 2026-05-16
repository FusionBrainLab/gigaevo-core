"""Lattice catalog and the protocol every lattice satisfies.

A *lattice* is a set with a partial order plus join (least upper bound)
and meet (greatest lower bound). The dataplane uses lattices to compose
freshness witnesses (epoch and generation) and to back CRDT merge.

The catalog is intentionally small:

    EpochLattice         total order on int, join = max
    GenerationLattice    total order on int, join = max
    ProductLattice[A, B] pair join componentwise (backs Versioned[T])
    MonotoneLattice[T]   total order under a Comparator, join = max
    BoolLattice          {False, True}, join = OR

``EpochLattice`` and ``GenerationLattice`` are byte-identical
implementations — the *point* is that they are separate types so mypy
rejects mixed-axis comparisons. Do not consolidate.

A ``HappensBeforeLattice`` for distributed causal order is added when
the distributed-worker workload requires it; not present here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class _Comparable(Protocol):
    """Minimal Comparable interface — used to bound :class:`MonotoneLattice`."""

    def __lt__(self, other: Any, /) -> bool: ...
    def __le__(self, other: Any, /) -> bool: ...


# ── concept ───────────────────────────────────────────────────────────


@runtime_checkable
class Lattice[E](Protocol):
    """A lattice over ``element_type = E``.

    Implementations expose three static / classmethod operations. ``meet``
    is required even though early read paths only need ``join`` — keeping
    the protocol symmetric leaves the door open for CRDT lattices whose
    ``merge`` is naturally expressed as ``meet`` (and avoids a v2 protocol
    split later).
    """

    @staticmethod
    def leq(a: E, b: E) -> bool:
        """Return True iff ``a ⩽ b`` in the lattice's partial order."""
        ...

    @staticmethod
    def join(a: E, b: E) -> E:
        """Least upper bound of ``a`` and ``b``."""
        ...

    @staticmethod
    def meet(a: E, b: E) -> E:
        """Greatest lower bound of ``a`` and ``b``."""
        ...


# ── concrete lattices ─────────────────────────────────────────────────


class EpochLattice:
    """Total order on ``int``; join = max, meet = min.

    Tracks the global write epoch (incremented on every state-changing
    coordinator call). Kept structurally identical to and distinct from
    :class:`GenerationLattice` so mismatched-axis comparisons fail at
    mypy time.
    """

    @staticmethod
    def leq(a: int, b: int) -> bool:
        return a <= b

    @staticmethod
    def join(a: int, b: int) -> int:
        return a if a >= b else b

    @staticmethod
    def meet(a: int, b: int) -> int:
        return a if a <= b else b


class GenerationLattice:
    """Total order on ``int``; join = max, meet = min.

    Per-aggregate generation counter. Distinct type from
    :class:`EpochLattice` on purpose (see module docstring).
    """

    @staticmethod
    def leq(a: int, b: int) -> bool:
        return a <= b

    @staticmethod
    def join(a: int, b: int) -> int:
        return a if a >= b else b

    @staticmethod
    def meet(a: int, b: int) -> int:
        return a if a <= b else b


class BoolLattice:
    """``{False, True}`` with ``False ⩽ True``; join = OR, meet = AND."""

    @staticmethod
    def leq(a: bool, b: bool) -> bool:
        return (not a) or b

    @staticmethod
    def join(a: bool, b: bool) -> bool:
        return a or b

    @staticmethod
    def meet(a: bool, b: bool) -> bool:
        return a and b


class ProductLattice[A, B]:
    """Pointwise join over a pair ``(A, B)``.

    Used by :class:`gigaevo.dataplane.models.Versioned` to compose epoch
    and generation into a single freshness witness whose admission gate
    is ``is_at_least``. Instances hold the two component lattice types
    so component operations can be dispatched without templating tricks.
    """

    def __init__(self, left: type[Lattice[A]], right: type[Lattice[B]]) -> None:
        self._L: type[Lattice[A]] = left
        self._R: type[Lattice[B]] = right

    def leq(self, a: tuple[A, B], b: tuple[A, B]) -> bool:
        return self._L.leq(a[0], b[0]) and self._R.leq(a[1], b[1])

    def join(self, a: tuple[A, B], b: tuple[A, B]) -> tuple[A, B]:
        return (self._L.join(a[0], b[0]), self._R.join(a[1], b[1]))

    def meet(self, a: tuple[A, B], b: tuple[A, B]) -> tuple[A, B]:
        return (self._L.meet(a[0], b[0]), self._R.meet(a[1], b[1]))


class MonotoneLattice[C: _Comparable]:
    """Total order under ``<=``; join = max, meet = min.

    Generic over any totally-ordered element type. Backs
    :class:`gigaevo.dataplane.models.Monotonic` to reject retrograde
    assignments at runtime.
    """

    @staticmethod
    def leq(a: C, b: C) -> bool:
        return a <= b

    @staticmethod
    def join(a: C, b: C) -> C:
        return a if (b <= a) else b

    @staticmethod
    def meet(a: C, b: C) -> C:
        return a if (a <= b) else b


__all__ = [
    "BoolLattice",
    "EpochLattice",
    "GenerationLattice",
    "Lattice",
    "MonotoneLattice",
    "ProductLattice",
]
