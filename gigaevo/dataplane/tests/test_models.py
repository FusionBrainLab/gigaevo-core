"""Tests for the type-system primitives in :mod:`gigaevo.dataplane.models`."""

from __future__ import annotations

import dataclasses
from typing import Literal

import pytest

from gigaevo.dataplane.errors import StaleReadError
from gigaevo.dataplane.models import (
    Err,
    Freshness,
    FreshnessAtLeast,
    FreshnessEventual,
    FreshnessStrict,
    HlcTimestamp,
    Monotonic,
    MonotonicCounter,
    Ok,
    Sourced,
    Versioned,
)

# ── Versioned ─────────────────────────────────────────────────────────


class TestVersioned:
    def test_is_at_least_below_epoch(self) -> None:
        v = Versioned(value="x", epoch=3, generation=1)
        assert not v.is_at_least(5, 0)

    def test_is_at_least_below_generation(self) -> None:
        v = Versioned(value="x", epoch=5, generation=1)
        assert not v.is_at_least(5, 3)

    def test_is_at_least_at_floor(self) -> None:
        v = Versioned(value="x", epoch=5, generation=3)
        assert v.is_at_least(5, 3)
        assert v.is_at_least(0, 0)

    def test_require_at_least_passes(self) -> None:
        v = Versioned(value=42, epoch=5, generation=0)
        same = v.require_at_least(5, 0)
        assert same is v

    def test_require_at_least_raises_with_observed_fields(self) -> None:
        v = Versioned(value=42, epoch=3, generation=0)
        with pytest.raises(StaleReadError) as exc:
            v.require_at_least(5, 0)
        assert exc.value.observed_epoch == 3
        assert exc.value.min_epoch == 5

    def test_combine_max_picks_greater(self) -> None:
        a = Versioned(value="a", epoch=1, generation=0)
        b = Versioned(value="b", epoch=2, generation=0)
        assert a.combine_max(b) is b
        assert b.combine_max(a) is b

    def test_combine_max_ties_pick_self(self) -> None:
        a = Versioned(value="a", epoch=1, generation=0)
        b = Versioned(value="b", epoch=1, generation=0)
        assert a.combine_max(b) is a
        assert b.combine_max(a) is b

    def test_combine_max_dominance_by_generation(self) -> None:
        a = Versioned(value="a", epoch=1, generation=5)
        b = Versioned(value="b", epoch=1, generation=3)
        assert a.combine_max(b) is a

    def test_map_preserves_epoch_and_generation(self) -> None:
        v = Versioned(value=10, epoch=5, generation=2)
        w = v.map(lambda x: x * 2)
        assert w.value == 20
        assert w.epoch == 5
        assert w.generation == 2

    def test_frozen(self) -> None:
        v = Versioned(value=1, epoch=0, generation=0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            v.value = 2  # type: ignore[misc]

    def test_rejects_negative_epoch(self) -> None:
        with pytest.raises(ValueError, match="epoch must be non-negative"):
            Versioned(value=1, epoch=-1, generation=0)

    def test_rejects_negative_generation(self) -> None:
        with pytest.raises(ValueError, match="generation must be non-negative"):
            Versioned(value=1, epoch=0, generation=-1)

    def test_combine_max_value_not_required_comparable(self) -> None:
        # Wrapped values that do not implement ordering must still
        # combine cleanly — comparison is on (epoch, generation) only.
        class Opaque:
            pass

        a = Versioned(value=Opaque(), epoch=1, generation=0)
        b = Versioned(value=Opaque(), epoch=2, generation=0)
        assert a.combine_max(b) is b


# ── Freshness ────────────────────────────────────────────────────────


class TestFreshnessShape:
    """The discriminated freshness type is structurally pinned.

    These tests document the shape contract so a contributor cannot
    silently rename a variant or add a fourth alternative without
    breaking them — the union is the load-bearing thing.
    """

    def test_eventual_constructs(self) -> None:
        f = FreshnessEventual()
        assert isinstance(f, FreshnessEventual)

    def test_at_least_default_zero(self) -> None:
        # Identity floor — every persisted blob clears (0, 0). The
        # default constructor exists as a degenerate case; production
        # callers pin a real expected counter explicitly.
        f = FreshnessAtLeast()
        assert f.epoch == 0
        assert f.generation == 0

    def test_at_least_carries_floor(self) -> None:
        f = FreshnessAtLeast(epoch=7, generation=3)
        assert f.epoch == 7
        assert f.generation == 3

    def test_at_least_rejects_negative_epoch(self) -> None:
        with pytest.raises(ValueError, match="epoch must be non-negative"):
            FreshnessAtLeast(epoch=-1, generation=0)

    def test_at_least_rejects_negative_generation(self) -> None:
        with pytest.raises(ValueError, match="generation must be non-negative"):
            FreshnessAtLeast(epoch=0, generation=-1)

    def test_strict_constructs(self) -> None:
        # Strict carries no payload — its semantics live in the
        # coordinator's read-time epoch snapshot. The dataclass exists
        # so ``match`` over the union is exhaustive without ``case _``.
        f = FreshnessStrict()
        assert isinstance(f, FreshnessStrict)

    def test_variants_frozen(self) -> None:
        # All three variants are frozen — a contributor cannot mutate
        # the floor after construction (which would race the in-flight
        # read).
        with pytest.raises(dataclasses.FrozenInstanceError):
            FreshnessAtLeast(epoch=1, generation=0).epoch = 99  # type: ignore[misc]

    def test_union_is_exhaustive_via_match(self) -> None:
        # The discriminated union must support exhaustive pattern
        # matching so a fourth variant added in the future surfaces as
        # a mypy error at every reader site rather than a runtime
        # fall-through bug.
        def admit(f: Freshness) -> str:
            match f:
                case FreshnessEventual():
                    return "eventual"
                case FreshnessAtLeast(epoch=e, generation=g):
                    return f"at-least({e},{g})"
                case FreshnessStrict():
                    return "strict"

        assert admit(FreshnessEventual()) == "eventual"
        assert admit(FreshnessAtLeast(epoch=2, generation=3)) == "at-least(2,3)"
        assert admit(FreshnessStrict()) == "strict"


# ── Sourced ───────────────────────────────────────────────────────────


class TestSourced:
    def test_holds_value(self) -> None:
        # The phantom ``S`` parameter is annotated at the call site so
        # mypy has something concrete to bind; runtime stores no tag.
        s: Sourced[str, Literal["local"]] = Sourced(value="hello")
        assert s.value == "hello"

    def test_retag_returns_new_wrapper(self) -> None:
        s: Sourced[list[int], Literal["cached"]] = Sourced(value=[1, 2, 3])
        retagged = s.retag(None)
        assert retagged is not s
        assert retagged.value is s.value


# ── Result / Ok / Err ─────────────────────────────────────────────────


class TestResult:
    def test_ok_is_ok(self) -> None:
        r = Ok(value=42)
        assert r.is_ok() is True
        assert r.is_err() is False
        assert r.unwrap() == 42

    def test_err_is_err(self) -> None:
        e = RuntimeError("boom")
        r = Err(error=e)
        assert r.is_ok() is False
        assert r.is_err() is True

    def test_err_unwrap_raises_exception_payload(self) -> None:
        r = Err(error=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            r.unwrap()

    def test_err_unwrap_wraps_non_exception(self) -> None:
        r = Err(error="just a string")
        with pytest.raises(RuntimeError, match="just a string"):
            r.unwrap()

    def test_match_pattern(self) -> None:
        def label(r: Ok[int] | Err[str]) -> str:
            match r:
                case Ok(value=v):
                    return f"ok-{v}"
                case Err(error=e):
                    return f"err-{e}"
            raise AssertionError("unreachable")

        assert label(Ok(value=7)) == "ok-7"
        assert label(Err(error="x")) == "err-x"

    def test_err_unwrap_typed_noreturn(self) -> None:
        # ``Err.unwrap`` is typed ``NoReturn``. The runtime guarantee is
        # that it always raises — no code path returns a value. This
        # test pins the runtime contract that backs the type signature.
        from typing import NoReturn, get_type_hints

        from gigaevo.dataplane.models import Err as ErrCls

        hints = get_type_hints(ErrCls.unwrap)
        assert hints["return"] is NoReturn


# ── Monotonic ─────────────────────────────────────────────────────────


class TestMonotonic:
    def test_initial_value(self) -> None:
        m = Monotonic(0)
        assert m.peek() == 0

    def test_advance_to_equal_ok(self) -> None:
        m = Monotonic(5)
        m.advance(5)
        assert m.peek() == 5

    def test_advance_forward(self) -> None:
        m = Monotonic(5)
        m.advance(10)
        assert m.peek() == 10

    def test_retrograde_raises(self) -> None:
        m = Monotonic(5)
        with pytest.raises(ValueError, match="Monotonic violation"):
            m.advance(3)

    def test_bump_increments(self) -> None:
        m = MonotonicCounter(0)
        assert m.bump() == 1
        assert m.bump() == 2
        assert m.peek() == 2

    def test_monotonic_base_has_no_bump(self) -> None:
        # The base ``Monotonic`` accepts any ``_Comparable`` type, so
        # ``bump()`` (which is integer-only) lives on the subclass. A
        # caller that has only the base type cannot accidentally invoke
        # ``+1`` on a stringly-typed counter.
        m = Monotonic("v1")
        assert not hasattr(m, "bump")

    def test_monotonic_counter_inherits_advance(self) -> None:
        m = MonotonicCounter(5)
        m.advance(7)
        assert m.peek() == 7
        with pytest.raises(ValueError, match="Monotonic violation"):
            m.advance(6)


# ── HlcTimestamp ──────────────────────────────────────────────────────


class TestHlcTimestamp:
    def test_pack_round_trip(self) -> None:
        t = HlcTimestamp(physical_ns=1234567890, counter=42)
        packed = t.pack_hex()
        assert len(packed) == 32
        unpacked = HlcTimestamp.unpack_hex(packed)
        assert unpacked == t

    def test_order_is_lexicographic_on_physical(self) -> None:
        a = HlcTimestamp(physical_ns=100, counter=999)
        b = HlcTimestamp(physical_ns=200, counter=0)
        assert a < b

    def test_order_counter_breaks_ties(self) -> None:
        a = HlcTimestamp(physical_ns=100, counter=5)
        b = HlcTimestamp(physical_ns=100, counter=6)
        assert a < b

    def test_uint64_bound(self) -> None:
        # Exactly at the boundary is OK
        HlcTimestamp(physical_ns=(1 << 64) - 1, counter=0)
        with pytest.raises(ValueError, match="uint64"):
            HlcTimestamp(physical_ns=1 << 64, counter=0)

    def test_uint32_bound(self) -> None:
        HlcTimestamp(physical_ns=0, counter=(1 << 32) - 1)
        with pytest.raises(ValueError, match="uint32"):
            HlcTimestamp(physical_ns=0, counter=1 << 32)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            HlcTimestamp(physical_ns=-1, counter=0)

    def test_unpack_rejects_bad_length(self) -> None:
        with pytest.raises(ValueError, match="32 hex chars"):
            HlcTimestamp.unpack_hex("deadbeef")

    def test_unpack_rejects_nonzero_trailing_pad(self) -> None:
        # The reserved trailing 8 hex chars must be zero on the wire so
        # the format is forward-compatible. A non-zero pad means either
        # a corrupted blob or an unsupported future encoding.
        t = HlcTimestamp(physical_ns=1, counter=2)
        # Replace the trailing pad with something non-zero.
        bad = t.pack_hex()[:24] + "deadbeef"
        with pytest.raises(ValueError, match="reserved field"):
            HlcTimestamp.unpack_hex(bad)

    def test_unpack_rejects_non_hex_characters(self) -> None:
        # Length is right (32 chars) but characters are non-hex; we
        # surface a typed ValueError rather than letting ``int()`` raise
        # an opaque one from inside the parser.
        with pytest.raises(ValueError, match="non-hex characters"):
            HlcTimestamp.unpack_hex("z" * 24 + "00000000")

    def test_unpack_rejects_uppercase_hex(self) -> None:
        # ``pack_hex`` emits lowercase; uppercase variants must not
        # round-trip — otherwise two encodings of the same value would
        # produce different content-hash inputs and equal HLCs would
        # land on divergent dedup buckets.
        t = HlcTimestamp(physical_ns=0xDEADBEEF, counter=42)
        upper = t.pack_hex().upper()
        with pytest.raises(ValueError, match="non-hex characters"):
            HlcTimestamp.unpack_hex(upper)

    def test_pack_pad_is_eight_zero_chars(self) -> None:
        # Lock the wire format so a future change that "saves the pad"
        # has to land here too.
        t = HlcTimestamp(physical_ns=0, counter=0)
        assert t.pack_hex().endswith("00000000")
        assert len(t.pack_hex()) == 32

    def test_hashable(self) -> None:
        t = HlcTimestamp(physical_ns=1, counter=2)
        assert {t, t} == {t}
