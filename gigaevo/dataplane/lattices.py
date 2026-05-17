"""Lattice catalog and the protocol every lattice satisfies.

A *lattice* is a set with a partial order plus join and meet. The
dataplane uses lattices to compose freshness witnesses (epoch and
generation) and to back CRDT merge.

    EpochLattice         total order on int, join = max
    GenerationLattice    total order on int, join = max
    ProductLattice[A, B] pair join componentwise (backs Versioned[T])
    MonotoneLattice[T]   total order under a Comparator, join = max
    BoolLattice          {False, True}, join = OR

``EpochLattice`` and ``GenerationLattice`` are byte-identical but kept
as distinct types so mypy rejects mixed-axis comparisons.

The :class:`Lattice` protocol is structural and runtime-checkable; both
class-level (static methods) and instance-level (bound methods)
implementations satisfy it.
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
    """A lattice over ``element_type = E`` exposing leq / join / meet.

    Implementations may expose the operations as ``@staticmethod`` or as
    instance methods on a stateful lattice such as
    :class:`ProductLattice`; both satisfy ``isinstance(obj, Lattice)``.
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

    Tracks the global write epoch. Structurally identical to
    :class:`GenerationLattice`; the two are distinct types so mypy
    rejects mixed-axis comparisons.
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
    """Total order on ``int``; join = max, meet = min. Per-aggregate counter."""

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

    Composes epoch and generation into a single freshness witness. Two
    instances compare equal iff their component lattice classes match,
    so callers do not need to thread a shared instance.
    """

    __slots__ = ("_left", "_right")

    def __init__(self, left: type[Lattice[A]], right: type[Lattice[B]]) -> None:
        self._left: type[Lattice[A]] = left
        self._right: type[Lattice[B]] = right

    def leq(self, a: tuple[A, B], b: tuple[A, B]) -> bool:
        return self._left.leq(a[0], b[0]) and self._right.leq(a[1], b[1])

    def join(self, a: tuple[A, B], b: tuple[A, B]) -> tuple[A, B]:
        return (self._left.join(a[0], b[0]), self._right.join(a[1], b[1]))

    def meet(self, a: tuple[A, B], b: tuple[A, B]) -> tuple[A, B]:
        return (self._left.meet(a[0], b[0]), self._right.meet(a[1], b[1]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProductLattice):
            return NotImplemented
        return self._left is other._left and self._right is other._right

    def __hash__(self) -> int:
        return hash((ProductLattice, self._left, self._right))

    def __repr__(self) -> str:
        return f"ProductLattice({self._left.__name__}, {self._right.__name__})"


class MonotoneLattice[C: _Comparable]:
    """Total order under ``<=``; join = max, meet = min. Generic over ``C``."""

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
