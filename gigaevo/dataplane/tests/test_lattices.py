"""Tests for the lattice catalog.

Each concrete lattice must satisfy:

    leq is reflexive and transitive on the test domain.
    join is commutative, associative, idempotent.
    join(a, b) is leq-greater than both a and b.
    meet(a, b) is leq-less than both a and b.

The product lattice additionally has value-equality semantics — two
``ProductLattice`` instances built from the same component classes
compare equal and hash equal.
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

    def test_join_dominates_both_inputs(self) -> None:
        # join(a, b) ⩾ a and join(a, b) ⩾ b for every a, b.
        for a, b in [(1, 2), (5, -3), (0, 0), (100, 99), (-7, -7)]:
            j = EpochLattice.join(a, b)
            assert EpochLattice.leq(a, j)
            assert EpochLattice.leq(b, j)

    def test_meet_dominated_by_both_inputs(self) -> None:
        # meet(a, b) ⩽ a and meet(a, b) ⩽ b for every a, b.
        for a, b in [(1, 2), (5, -3), (0, 0)]:
            m = EpochLattice.meet(a, b)
            assert EpochLattice.leq(m, a)
            assert EpochLattice.leq(m, b)


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

    def test_value_equality(self) -> None:
        # Two instances with the same component classes compare equal.
        p1 = ProductLattice(EpochLattice, GenerationLattice)
        p2 = ProductLattice(EpochLattice, GenerationLattice)
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_inequality_with_different_components(self) -> None:
        p1 = ProductLattice(EpochLattice, GenerationLattice)
        p2 = ProductLattice(GenerationLattice, EpochLattice)
        # Same classes, swapped roles — still different operation suite.
        assert p1 != p2

    def test_equality_with_non_product_returns_notimplemented_path(self) -> None:
        # ``__eq__`` returning NotImplemented surfaces as False when the
        # comparison runs.
        p = ProductLattice(EpochLattice, GenerationLattice)
        assert p != 42
        assert p != "ProductLattice"
        assert p != EpochLattice

    def test_repr_mentions_components(self) -> None:
        p = ProductLattice(EpochLattice, GenerationLattice)
        r = repr(p)
        assert "EpochLattice" in r
        assert "GenerationLattice" in r

    def test_join_dominates_both_inputs(
        self, product: ProductLattice[int, int]
    ) -> None:
        for a, b in [((1, 2), (3, 4)), ((5, 0), (1, 10)), ((-1, -1), (-1, -1))]:
            j = product.join(a, b)
            assert product.leq(a, j)
            assert product.leq(b, j)

    def test_instance_satisfies_protocol(
        self, product: ProductLattice[int, int]
    ) -> None:
        # ``runtime_checkable`` Protocol checks attribute presence; the
        # instance exposes leq/join/meet bound methods.
        assert isinstance(product, Lattice)


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

    def test_join_dominates_both_inputs(self) -> None:
        # join(a, b) ⩾ a and join(a, b) ⩾ b under leq for tuple inputs
        # — exercises the tuple-comparison code path.
        pairs = [
            ((1, 0), (1, 1)),
            ((2, 0), (1, 5)),
            ((0, 0), (0, 0)),
            ((5, 5), (1, 9)),
        ]
        for a, b in pairs:
            j = MonotoneLattice.join(a, b)
            assert MonotoneLattice.leq(a, j)
            assert MonotoneLattice.leq(b, j)

    def test_join_idempotent(self) -> None:
        for x in (0, 1, "abc", (1, 2)):
            assert MonotoneLattice.join(x, x) == x  # type: ignore[type-var]

    def test_join_commutative(self) -> None:
        for a, b in [(1, 2), ("a", "b"), ((1, 0), (0, 1))]:
            assert MonotoneLattice.join(a, b) == MonotoneLattice.join(b, a)  # type: ignore[type-var]


# ── Lattice protocol ─────────────────────────────────────────────────


class TestLatticeProtocol:
    def test_concrete_static_lattices_satisfy_protocol(self) -> None:
        # runtime_checkable Protocols approximate structural typing.
        # Static lattices satisfy at the *class* level — leq/join/meet
        # are accessible on the class itself.
        for cls in (EpochLattice, GenerationLattice, BoolLattice, MonotoneLattice):
            assert isinstance(cls, Lattice)

    def test_product_lattice_instance_satisfies_protocol(self) -> None:
        # Instance-shaped lattice; the protocol still matches because
        # the *instance* exposes leq/join/meet as bound methods.
        p = ProductLattice(EpochLattice, GenerationLattice)
        assert isinstance(p, Lattice)
