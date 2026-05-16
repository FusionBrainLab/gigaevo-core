"""Integration tests for the bounded recency list API.

Covers ``bounded_list_push.lua`` plus :meth:`DataPlane.bounded_list_push`
and :meth:`DataPlane.bounded_list_range`. The push primitive is a one
atomic LPUSH + LTRIM round-trip; concurrent writers must never observe
the list briefly over-cap, and the trim must take effect even if many
writers race on the same key.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
    """Coordinator wired to an isolated fakeredis instance.

    Mirrors the pattern used by :mod:`test_lwwr` and
    :mod:`test_crdt_counter` so the test fixtures stay uniform across
    the dataplane primitive suite.
    """
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="test")
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    try:
        yield coord
    finally:
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await fake.aclose()  # type: ignore[attr-defined]


# ── basic push / range ───────────────────────────────────────────────


class TestBoundedListBasic:
    async def test_single_push_returns_length_one(self, coord: dp.DataPlane) -> None:
        out = await coord.bounded_list_push("k", 0.42, cap=5)
        assert isinstance(out, dp.Ok)
        assert out.value == 1

    async def test_range_after_push_returns_value(self, coord: dp.DataPlane) -> None:
        await coord.bounded_list_push("k", 0.42, cap=5)
        read = await coord.bounded_list_range("k", count=5)
        assert isinstance(read, dp.Ok)
        assert read.value == [0.42]

    async def test_pushes_appear_newest_first(self, coord: dp.DataPlane) -> None:
        for v in (1, 2, 3):
            await coord.bounded_list_push("k", v, cap=5)
        read = await coord.bounded_list_range("k", count=5)
        assert isinstance(read, dp.Ok)
        assert read.value == [3, 2, 1]

    async def test_range_missing_list_returns_empty(self, coord: dp.DataPlane) -> None:
        read = await coord.bounded_list_range("nonexistent", count=10)
        assert isinstance(read, dp.Ok)
        assert read.value == []

    async def test_string_value_round_trips(self, coord: dp.DataPlane) -> None:
        await coord.bounded_list_push("k", "hello", cap=5)
        read = await coord.bounded_list_range("k", count=5)
        assert isinstance(read, dp.Ok)
        assert read.value == ["hello"]


# ── cap enforcement ──────────────────────────────────────────────────


class TestBoundedListCap:
    async def test_cap_enforced_on_overflow(self, coord: dp.DataPlane) -> None:
        cap = 3
        for v in range(10):
            out = await coord.bounded_list_push("k", v, cap=cap)
            assert isinstance(out, dp.Ok)
            assert out.value <= cap
        read = await coord.bounded_list_range("k", count=10)
        assert isinstance(read, dp.Ok)
        # Newest-first: last three pushes were 7, 8, 9.
        assert read.value == [9, 8, 7]

    async def test_post_push_length_capped(self, coord: dp.DataPlane) -> None:
        for v in range(5):
            out = await coord.bounded_list_push("k", v, cap=2)
            assert isinstance(out, dp.Ok)
            assert out.value <= 2

    async def test_zero_cap_rejected(self, coord: dp.DataPlane) -> None:
        out = await coord.bounded_list_push("k", 1, cap=0)
        assert isinstance(out, dp.Err)

    async def test_negative_cap_rejected(self, coord: dp.DataPlane) -> None:
        out = await coord.bounded_list_push("k", 1, cap=-3)
        assert isinstance(out, dp.Err)

    async def test_count_must_be_positive(self, coord: dp.DataPlane) -> None:
        out = await coord.bounded_list_range("k", count=0)
        assert isinstance(out, dp.Err)


# ── concurrency ──────────────────────────────────────────────────────


class TestBoundedListConcurrency:
    async def test_many_concurrent_writers_respect_cap(
        self, coord: dp.DataPlane
    ) -> None:
        """50 concurrent pushers with cap=10; final length must equal 10.

        The script's atomic LPUSH+LTRIM is the load-bearing invariant.
        A naive two-call implementation could leave the list at length
        50 momentarily; the script forces every observer to see a
        length-bounded list at all times.
        """
        cap = 10
        n_writers = 50

        async def writer(idx: int) -> None:
            await coord.bounded_list_push("k", idx, cap=cap)

        await asyncio.gather(*(writer(i) for i in range(n_writers)))
        read = await coord.bounded_list_range("k", count=n_writers)
        assert isinstance(read, dp.Ok)
        assert len(read.value) == cap
        # Every retained entry is from the original push set.
        for v in read.value:
            assert isinstance(v, int)
            assert 0 <= v < n_writers


# ── key validation ───────────────────────────────────────────────────


class TestBoundedListValidation:
    async def test_empty_key_rejected(self, coord: dp.DataPlane) -> None:
        with pytest.raises(ValueError):
            await coord.bounded_list_push("", 1, cap=5)

    async def test_nul_in_key_rejected(self, coord: dp.DataPlane) -> None:
        with pytest.raises(ValueError):
            await coord.bounded_list_push("bad\x00key", 1, cap=5)

    async def test_colon_in_key_allowed(self, coord: dp.DataPlane) -> None:
        # Hierarchical keys carry colons as namespace separators.
        out = await coord.bounded_list_push("prompt:fitnesses:abc", 0.5, cap=5)
        assert isinstance(out, dp.Ok)
        assert out.value == 1
