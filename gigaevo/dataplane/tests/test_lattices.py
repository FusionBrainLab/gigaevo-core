"""Tests for the lattice catalog.

Each concrete lattice must satisfy:

    leq is reflexive and transitive on the test domain
    join is commutative, associative, idempotent
    join(a, b) is leq-greater than both a and b
    meet(a, b) is leq-less than both a and b
"""

from __future__ import annotations

import pytest

from gigaevo.dataplane.lattices import (
    BoolLattice,
    EpochLattice,
    GenerationLattice,
    Lattice,
    MonotoneLattice,
    ProductLattice,
)

# ── EpochLattice ──────────────────────────────────────────────────────


class TestEpochLattice:
    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (3, 3), (-1, 0)])
    def test_leq_reflexive_and_basic(self, a: int, b: int) -> None:
        assert EpochLattice.leq(a, b) == (a <= b)

    def test_join_idempotent(self) -> None:
        for x in (-5, 0, 7, 100):
            assert EpochLattice.join(x, x) == x

    def test_join_commutative(self) -> None:
        for a, b in [(1, 2), (5, -3), (0, 0), (100, 99)]:
            assert EpochLattice.join(a, b) == EpochLattice.join(b, a)

    def test_join_associative(self) -> None:
        for a, b, c in [(1, 2, 3), (-5, 0, 5), (10, 10, 10)]:
            left = EpochLattice.join(EpochLattice.join(a, b), c)
            right = EpochLattice.join(a, EpochLattice.join(b, c))
            assert left == right

    def test_meet_is_min(self) -> None:
        assert EpochLattice.meet(3, 7) == 3
        assert EpochLattice.meet(-1, -5) == -5
        assert EpochLattice.meet(0, 0) == 0


# ── GenerationLattice ────────────────────────────────────────────────


class TestGenerationLatticeStructurallyIdenticalToEpoch:
    def test_join_matches_epoch_lattice(self) -> None:
        for a, b in [(1, 2), (5, 5), (-3, 10)]:
            assert GenerationLattice.join(a, b) == EpochLattice.join(a, b)

    def test_meet_matches_epoch_lattice(self) -> None:
        for a, b in [(1, 2), (5, 5), (-3, 10)]:
            assert GenerationLattice.meet(a, b) == EpochLattice.meet(a, b)

    def test_distinct_class_identity(self) -> None:
        # The two are byte-identical but DIFFERENT types — that's the point.
        assert EpochLattice is not GenerationLattice


# ── BoolLattice ──────────────────────────────────────────────────────


class TestBoolLattice:
    def test_leq(self) -> None:
        assert BoolLattice.leq(False, False)
        assert BoolLattice.leq(False, True)
        assert not BoolLattice.leq(True, False)
        assert BoolLattice.leq(True, True)

    def test_join_is_or(self) -> None:
        assert BoolLattice.join(True, False) is True
        assert BoolLattice.join(False, False) is False
        assert BoolLattice.join(True, True) is True

    def test_meet_is_and(self) -> None:
        assert BoolLattice.meet(True, False) is False
        assert BoolLattice.meet(True, True) is True


# ── ProductLattice ───────────────────────────────────────────────────


class TestProductLattice:
    @pytest.fixture
    def product(self) -> ProductLattice[int, int]:
        return ProductLattice(EpochLattice, GenerationLattice)

    def test_leq_componentwise(self, product: ProductLattice[int, int]) -> None:
        assert product.leq((1, 2), (3, 4))
        assert product.leq((1, 2), (1, 2))
        # 2 > 1 on the second axis breaks ordering even if first axis is lower
        assert not product.leq((1, 5), (2, 3))
        # First axis lower, second higher — still incomparable -> not leq
        assert not product.leq((2, 1), (1, 5))

    def test_join_componentwise(self, product: ProductLattice[int, int]) -> None:
        assert product.join((1, 5), (3, 2)) == (3, 5)

    def test_meet_componentwise(self, product: ProductLattice[int, int]) -> None:
        assert product.meet((1, 5), (3, 2)) == (1, 2)


# ── MonotoneLattice ──────────────────────────────────────────────────


class TestMonotoneLattice:
    def test_int(self) -> None:
        assert MonotoneLattice.join(3, 7) == 7
        assert MonotoneLattice.meet(3, 7) == 3
        assert MonotoneLattice.leq(3, 3)

    def test_str_lexicographic(self) -> None:
        assert MonotoneLattice.join("apple", "banana") == "banana"
        assert MonotoneLattice.meet("apple", "banana") == "apple"

    def test_tuple_lexicographic(self) -> None:
        assert MonotoneLattice.join((1, 0), (1, 1)) == (1, 1)
        assert MonotoneLattice.meet((1, 0), (1, 1)) == (1, 0)


# ── Lattice protocol ─────────────────────────────────────────────────


class TestLatticeProtocol:
    def test_concrete_lattices_satisfy_protocol(self) -> None:
        # runtime_checkable Protocols approximate structural typing;
        # this is a smoke test, not exhaustive.
        for cls in (EpochLattice, GenerationLattice, BoolLattice):
            assert isinstance(cls, Lattice)
