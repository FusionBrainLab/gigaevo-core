"""Integration tests for the LWW-register API.

Covers ``lwwr_set.lua`` plus :meth:`DataPlane.lwwr_set` /
:meth:`DataPlane.lwwr_get`. The HLC-tiebreak invariant: a write
strictly newer than the stored HLC replaces; equal or older writes
are silently kept. The Lua script does a lexicographic comparison on
the 32-char hex HLC, which is correct because the big-endian
``physical_ns || counter || pad`` encoding is order-preserving.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
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


def _hlc(physical_ns: int, counter: int = 0) -> dp.HlcTimestamp:
    return dp.HlcTimestamp(physical_ns=physical_ns, counter=counter)


# ── basic set / get ──────────────────────────────────────────────────


class TestSetThenGet:
    async def test_first_write_replaces_empty(self, coord: dp.DataPlane) -> None:
        outcome = await coord.lwwr_set(
            "last-known-model",
            {"name": "v3.2", "rev": 7},
            hlc=_hlc(100),
        )
        assert isinstance(outcome, dp.Ok)
        assert outcome.value == "replaced"

    async def test_get_after_set_returns_value(self, coord: dp.DataPlane) -> None:
        value = {"name": "v3.2", "rev": 7}
        await coord.lwwr_set("registry", value, hlc=_hlc(100))
        read = await coord.lwwr_get("registry")
        assert isinstance(read, dp.Ok)
        assert read.value is not None
        assert read.value.value == value
        assert read.value.hlc == _hlc(100)

    async def test_get_before_any_write_returns_ok_none(
        self, coord: dp.DataPlane
    ) -> None:
        read = await coord.lwwr_get("never-written")
        assert isinstance(read, dp.Ok)
        assert read.value is None


# ── HLC ordering ─────────────────────────────────────────────────────


class TestHlcOrdering:
    async def test_newer_hlc_replaces(self, coord: dp.DataPlane) -> None:
        await coord.lwwr_set("k", "v-old", hlc=_hlc(100))
        outcome = await coord.lwwr_set("k", "v-new", hlc=_hlc(200))
        assert isinstance(outcome, dp.Ok)
        assert outcome.value == "replaced"
        read = await coord.lwwr_get("k")
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value == "v-new"
        assert read.value.hlc == _hlc(200)

    async def test_older_hlc_kept(self, coord: dp.DataPlane) -> None:
        await coord.lwwr_set("k", "v-new", hlc=_hlc(200))
        outcome = await coord.lwwr_set("k", "v-old", hlc=_hlc(100))
        assert isinstance(outcome, dp.Ok)
        assert outcome.value == "kept"
        read = await coord.lwwr_get("k")
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value == "v-new"  # untouched

    async def test_equal_hlc_kept(self, coord: dp.DataPlane) -> None:
        # Equal HLC means a duplicate write; LWW semantics keep the
        # first arrival (the stored value's HLC is already >= candidate).
        await coord.lwwr_set("k", "first", hlc=_hlc(100, 5))
        outcome = await coord.lwwr_set("k", "second", hlc=_hlc(100, 5))
        assert isinstance(outcome, dp.Ok)
        assert outcome.value == "kept"
        read = await coord.lwwr_get("k")
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value == "first"

    async def test_counter_breaks_physical_tie(self, coord: dp.DataPlane) -> None:
        # Same physical_ns; counter advances. The newer counter wins.
        await coord.lwwr_set("k", "a", hlc=_hlc(100, counter=1))
        outcome = await coord.lwwr_set("k", "b", hlc=_hlc(100, counter=2))
        assert isinstance(outcome, dp.Ok)
        assert outcome.value == "replaced"
        read = await coord.lwwr_get("k")
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value == "b"

    async def test_out_of_order_writes_converge_to_max(
        self, coord: dp.DataPlane
    ) -> None:
        """Write four versions in shuffled HLC order; only the max
        survives, regardless of arrival order."""
        writes = [
            ("v50", _hlc(50)),
            ("v200", _hlc(200)),
            ("v10", _hlc(10)),
            ("v100", _hlc(100)),
        ]
        for value, hlc in writes:
            await coord.lwwr_set("k", value, hlc=hlc)
        read = await coord.lwwr_get("k")
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value == "v200"
        assert read.value.hlc == _hlc(200)


# ── value shapes ─────────────────────────────────────────────────────


class TestValueShapes:
    async def test_complex_value_round_trips(self, coord: dp.DataPlane) -> None:
        value = {
            "nested": {"list": [1, 2, 3], "set-as-list": [True, False]},
            "scalar": 42,
            "none": None,
        }
        await coord.lwwr_set("k", value, hlc=_hlc(1))
        read = await coord.lwwr_get("k")
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value == value

    async def test_primitive_values(self, coord: dp.DataPlane) -> None:
        # Each iteration uses a strictly-newer HLC so the test exercises
        # replacement rather than the keep path.
        for i, primitive in enumerate((42, 3.14, "hello", True, False, None)):
            hlc = _hlc(physical_ns=(i + 1) * 100)
            await coord.lwwr_set("k", primitive, hlc=hlc)
            read = await coord.lwwr_get("k")
            assert isinstance(read, dp.Ok) and read.value is not None
            assert read.value.value == primitive
            assert read.value.hlc == hlc

    async def test_overwrite_changes_value_and_hlc_together(
        self, coord: dp.DataPlane
    ) -> None:
        await coord.lwwr_set("k", "v1", hlc=_hlc(100))
        first = await coord.lwwr_get("k")
        await coord.lwwr_set("k", "v2", hlc=_hlc(200))
        second = await coord.lwwr_get("k")
        assert isinstance(first, dp.Ok) and first.value is not None
        assert isinstance(second, dp.Ok) and second.value is not None
        assert first.value.value == "v1"
        assert first.value.hlc == _hlc(100)
        assert second.value.value == "v2"
        assert second.value.hlc == _hlc(200)


# ── concurrent writers ───────────────────────────────────────────────


class TestConcurrency:
    async def test_concurrent_writers_max_hlc_wins(self, coord: dp.DataPlane) -> None:
        """N coroutines with different HLCs racing the same key — the
        register must converge to the value of the writer with the
        largest HLC."""
        writers = [(f"v-{i}", _hlc(i * 10)) for i in range(20)]

        async def writer(name: str, hlc: dp.HlcTimestamp) -> None:
            await coord.lwwr_set("racing", name, hlc=hlc)

        await asyncio.gather(*(writer(n, h) for n, h in writers))
        read = await coord.lwwr_get("racing")
        assert isinstance(read, dp.Ok) and read.value is not None
        # Largest HLC was 190 (i=19).
        assert read.value.value == "v-19"
        assert read.value.hlc == _hlc(190)


# ── isolation across keys ────────────────────────────────────────────


class TestKeyIsolation:
    async def test_independent_keys_isolated(self, coord: dp.DataPlane) -> None:
        await coord.lwwr_set("a", "value-a", hlc=_hlc(100))
        await coord.lwwr_set("b", "value-b", hlc=_hlc(50))
        ra = await coord.lwwr_get("a")
        rb = await coord.lwwr_get("b")
        assert isinstance(ra, dp.Ok) and ra.value is not None
        assert isinstance(rb, dp.Ok) and rb.value is not None
        assert ra.value.value == "value-a"
        assert rb.value.value == "value-b"
