"""Integration tests for the small-set directory and raw-key escape hatch.

Covers :meth:`DataPlane.set_add` / :meth:`DataPlane.set_members` and the
:meth:`DataPlane.raw_*` family that addresses fully-qualified Redis keys
outside the coordinator's own ``key_prefix`` namespace. These primitives
back the prompt-fitness metric-name directory and the cross-run reads
performed by the prompt co-evolution path.
"""

from __future__ import annotations

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


# ── set primitives ──────────────────────────────────────────────────


class TestSetAdd:
    async def test_first_add_returns_one(self, coord: dp.DataPlane) -> None:
        out = await coord.set_add("k", "alpha")
        assert isinstance(out, dp.Ok)
        assert out.value == 1

    async def test_duplicate_add_returns_zero(self, coord: dp.DataPlane) -> None:
        await coord.set_add("k", "alpha")
        out = await coord.set_add("k", "alpha")
        assert isinstance(out, dp.Ok)
        assert out.value == 0

    async def test_members_after_adds(self, coord: dp.DataPlane) -> None:
        for m in ("alpha", "beta", "alpha", "gamma"):
            await coord.set_add("k", m)
        out = await coord.set_members("k")
        assert isinstance(out, dp.Ok)
        assert out.value == frozenset({"alpha", "beta", "gamma"})

    async def test_members_missing_set_empty(self, coord: dp.DataPlane) -> None:
        out = await coord.set_members("never-written")
        assert isinstance(out, dp.Ok)
        assert out.value == frozenset()

    async def test_empty_member_rejected(self, coord: dp.DataPlane) -> None:
        out = await coord.set_add("k", "")
        assert isinstance(out, dp.Err)


# ── raw-key access ──────────────────────────────────────────────────


class TestRawKeyAccess:
    async def test_raw_get_missing(self, coord: dp.DataPlane) -> None:
        out = await coord.raw_get("absent:key")
        assert isinstance(out, dp.Ok)
        assert out.value is None

    async def test_raw_get_after_set(self, coord: dp.DataPlane) -> None:
        await coord._connection.pool.set("foo:bar", "v1")  # type: ignore[misc]
        out = await coord.raw_get("foo:bar")
        assert isinstance(out, dp.Ok)
        assert out.value == "v1"

    async def test_raw_hash_get_missing(self, coord: dp.DataPlane) -> None:
        out = await coord.raw_hash_get("h:none", "field")
        assert isinstance(out, dp.Ok)
        assert out.value is None

    async def test_raw_hash_get_present(self, coord: dp.DataPlane) -> None:
        await coord._connection.pool.hset("h:x", "f", "v")  # type: ignore[misc]
        out = await coord.raw_hash_get("h:x", "f")
        assert isinstance(out, dp.Ok)
        assert out.value == "v"

    async def test_raw_hash_values_missing(self, coord: dp.DataPlane) -> None:
        out = await coord.raw_hash_values("h:none")
        assert isinstance(out, dp.Ok)
        assert out.value == []

    async def test_raw_hash_values_present(self, coord: dp.DataPlane) -> None:
        await coord._connection.pool.hset(  # type: ignore[misc]
            "h:y", mapping={"a": "1", "b": "2", "c": "3"}
        )
        out = await coord.raw_hash_values("h:y")
        assert isinstance(out, dp.Ok)
        assert sorted(out.value) == ["1", "2", "3"]

    async def test_raw_get_addresses_cross_namespace(self, coord: dp.DataPlane) -> None:
        """Raw-key access ignores the coordinator's ``key_prefix``.

        Demonstrates the cross-run scenario: a key written by a peer
        with a different key prefix is visible to ``raw_get`` even
        though the coordinator's typed primitives would prefix the
        lookup under its own namespace.
        """
        await coord._connection.pool.set(  # type: ignore[misc]
            "other_run:archive", "payload"
        )
        # Read directly without applying the coordinator's prefix.
        out = await coord.raw_get("other_run:archive")
        assert isinstance(out, dp.Ok)
        assert out.value == "payload"
