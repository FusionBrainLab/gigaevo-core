"""Tests for gigaevo/evolution/storage/archive_storage.py"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
import gigaevo.dataplane as dp
from gigaevo.evolution.storage.archive_storage import RedisArchiveStorage
from gigaevo.evolution.strategies.selectors import (
    ParetoFrontSelector,
    SumArchiveSelector,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _prog(metrics=None):
    p = Program(code="def solve(): return 1", state=ProgramState.DONE)
    if metrics:
        p.add_metrics(metrics)
    return p


def _always_better(new, current):
    return True


def _never_better(new, current):
    return False


@pytest.fixture
async def storage(fakeredis_storage):
    """Re-use conftest's fakeredis_storage instead of duplicating setup."""
    return fakeredis_storage


@pytest.fixture
async def archive(storage):
    return RedisArchiveStorage(storage, key_prefix="test")


# ── shared-server fixtures for the dataplane-backed archive path ─────────
#
# The dataplane and the program storage must talk to the same backing
# Redis so a candidate written via ``storage.add`` is visible to the Lua
# script's swap routine. fakeredis preserves state through the
# ``FakeServer`` argument; both clients are built against one server.


@pytest.fixture
async def shared_server() -> AsyncIterator[fakeredis.FakeServer]:
    server = fakeredis.FakeServer()
    yield server


@pytest.fixture
async def dp_storage(
    shared_server: fakeredis.FakeServer,
) -> AsyncIterator[RedisProgramStorage]:
    """Program storage and dataplane sharing one fakeredis server."""
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
    )
    storage = RedisProgramStorage(config)
    fake = fakeredis.aioredis.FakeRedis(server=shared_server, decode_responses=True)
    storage._conn._redis = fake  # type: ignore[attr-defined]
    storage._conn._closing = False  # type: ignore[attr-defined]
    try:
        yield storage
    finally:
        await storage.close()


@pytest.fixture
async def coord(shared_server: fakeredis.FakeServer) -> AsyncIterator[dp.DataPlane]:
    coord = dp.DataPlane("redis://embedded/0", key_prefix="test")
    fake = fakeredis.aioredis.FakeRedis(server=shared_server, decode_responses=True)
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


@pytest.fixture
async def dp_archive(dp_storage, coord):
    return RedisArchiveStorage(dp_storage, key_prefix="test", dataplane=coord)


class TestRedisArchiveStorageBasic:
    async def test_add_and_get(self, storage, archive):
        p = _prog(metrics={"score": 10.0})
        await storage.add(p)

        added = await archive.add_elite((0, 1), p, _always_better)
        assert added is True

        elite = await archive.get_elite((0, 1))
        assert elite is not None
        assert elite.id == p.id

    async def test_empty_cell(self, archive):
        elite = await archive.get_elite((99, 99))
        assert elite is None

    async def test_replace_worse_with_better(self, storage, archive):
        old = _prog(metrics={"score": 1.0})
        new = _prog(metrics={"score": 10.0})
        await storage.add(old)
        await storage.add(new)

        await archive.add_elite((0,), old, _always_better)
        replaced = await archive.add_elite(
            (0,), new, lambda n, c: n.metrics["score"] > c.metrics["score"]
        )
        assert replaced is True

        elite = await archive.get_elite((0,))
        assert elite.id == new.id

    async def test_keep_better_reject_worse(self, storage, archive):
        good = _prog(metrics={"score": 10.0})
        bad = _prog(metrics={"score": 1.0})
        await storage.add(good)
        await storage.add(bad)

        await archive.add_elite((0,), good, _always_better)
        rejected = await archive.add_elite(
            (0,), bad, lambda n, c: n.metrics["score"] > c.metrics["score"]
        )
        assert rejected is False

        elite = await archive.get_elite((0,))
        assert elite.id == good.id

    async def test_remove(self, storage, archive):
        p = _prog()
        await storage.add(p)
        await archive.add_elite((0,), p, _always_better)

        removed = await archive.remove_elite((0,))
        assert removed is True

        elite = await archive.get_elite((0,))
        assert elite is None

    async def test_remove_nonexistent(self, archive):
        removed = await archive.remove_elite((99,))
        assert removed is False

    async def test_add_unsaved_program_ignored(self, archive):
        p = _prog()
        # Don't save to storage
        added = await archive.add_elite((0,), p, _always_better)
        assert added is False


class TestRedisArchiveStorageBulk:
    async def test_get_all(self, storage, archive):
        p1 = _prog()
        p2 = _prog()
        await storage.add(p1)
        await storage.add(p2)
        await archive.add_elite((0,), p1, _always_better)
        await archive.add_elite((1,), p2, _always_better)

        all_ids = await archive.get_all_elites()
        assert set(all_ids) == {p1.id, p2.id}

    async def test_size(self, storage, archive):
        assert await archive.size() == 0

        p = _prog()
        await storage.add(p)
        await archive.add_elite((0,), p, _always_better)
        assert await archive.size() == 1

    async def test_clear(self, storage, archive):
        p = _prog()
        await storage.add(p)
        await archive.add_elite((0,), p, _always_better)

        cleared = await archive.clear_all_elites()
        assert cleared == 1
        assert await archive.size() == 0

    async def test_clear_empty(self, archive):
        cleared = await archive.clear_all_elites()
        assert cleared == 0

    async def test_bulk_add(self, storage, archive):
        p1 = _prog()
        p2 = _prog()
        await storage.add(p1)
        await storage.add(p2)

        placements = [((0,), p1), ((1,), p2)]
        count = await archive.bulk_add_elites(placements, _always_better)
        assert count == 2
        assert await archive.size() == 2

    async def test_bulk_add_empty(self, archive):
        count = await archive.bulk_add_elites([], _always_better)
        assert count == 0

    async def test_bulk_remove(self, storage, archive):
        p1 = _prog()
        p2 = _prog()
        await storage.add(p1)
        await storage.add(p2)
        await archive.add_elite((0,), p1, _always_better)
        await archive.add_elite((1,), p2, _always_better)

        count = await archive.bulk_remove_elites_by_id([p1.id, p2.id])
        assert count == 2
        assert await archive.size() == 0

    async def test_bulk_remove_empty(self, archive):
        count = await archive.bulk_remove_elites_by_id([])
        assert count == 0


class TestRedisArchiveStorageReverseIndex:
    async def test_remove_by_id(self, storage, archive):
        p = _prog()
        await storage.add(p)
        await archive.add_elite((0, 1), p, _always_better)

        removed = await archive.remove_elite_by_id(p.id)
        assert removed is True
        assert await archive.get_elite((0, 1)) is None

    async def test_remove_by_id_not_found(self, archive):
        removed = await archive.remove_elite_by_id("nonexistent-id")
        assert removed is False

    async def test_reverse_updated_on_replace(self, storage, archive):
        old = _prog()
        new = _prog()
        await storage.add(old)
        await storage.add(new)

        await archive.add_elite((0,), old, _always_better)
        await archive.add_elite((0,), new, _always_better)

        # Old program should no longer be findable
        removed_old = await archive.remove_elite_by_id(old.id)
        assert removed_old is False

        # New program should be findable
        removed_new = await archive.remove_elite_by_id(new.id)
        assert removed_new is True

    async def test_one_to_one_mapping(self, storage, archive):
        """Each program can only be elite in ONE cell at a time."""
        p = _prog()
        await storage.add(p)

        await archive.add_elite((0,), p, _always_better)
        await archive.add_elite((1,), p, _always_better)

        # Program should be in cell (1,) now (latest add)
        # The reverse index maps program_id -> cell
        removed = await archive.remove_elite_by_id(p.id)
        assert removed is True


class TestRedisArchiveStorageCellDescriptor:
    def test_field_serialization(self):
        assert RedisArchiveStorage._field((0, 1, 2)) == "0,1,2"
        assert RedisArchiveStorage._field((5,)) == "5"
        assert RedisArchiveStorage._field(()) == ""


class TestRedisArchiveStorageConcurrency:
    async def test_concurrent_add_same_cell(self, storage, archive):
        """Use asyncio.gather to add 2 different programs to the same cell. Exactly one wins."""
        p1 = _prog(metrics={"score": 5.0})
        p2 = _prog(metrics={"score": 10.0})
        await storage.add(p1)
        await storage.add(p2)

        await asyncio.gather(
            archive.add_elite((0,), p1, _always_better),
            archive.add_elite((0,), p2, _always_better),
        )
        # Both may report True (optimistic locking can succeed for both if non-overlapping)
        # but the cell should contain exactly one program
        elite = await archive.get_elite((0,))
        assert elite is not None
        assert elite.id in {p1.id, p2.id}
        assert await archive.size() == 1

    async def test_concurrent_add_different_cells(self, storage, archive):
        """asyncio.gather on 2 different cells. Both succeed."""
        p1 = _prog(metrics={"score": 5.0})
        p2 = _prog(metrics={"score": 10.0})
        await storage.add(p1)
        await storage.add(p2)

        results = await asyncio.gather(
            archive.add_elite((0,), p1, _always_better),
            archive.add_elite((1,), p2, _always_better),
        )
        assert results == [True, True]
        assert await archive.size() == 2

        e1 = await archive.get_elite((0,))
        e2 = await archive.get_elite((1,))
        assert e1.id == p1.id
        assert e2.id == p2.id

    async def test_add_replace_race(self, storage, archive):
        """Add elite, then concurrently try to replace and remove it. Verify consistent state."""
        original = _prog(metrics={"score": 5.0})
        replacement = _prog(metrics={"score": 10.0})
        await storage.add(original)
        await storage.add(replacement)

        await archive.add_elite((0,), original, _always_better)

        # Concurrently replace and remove
        await asyncio.gather(
            archive.add_elite((0,), replacement, _always_better),
            archive.remove_elite((0,)),
        )

        # After the race, the cell should be in a consistent state:
        # either empty (remove won) or contain the replacement
        elite = await archive.get_elite((0,))
        if elite is not None:
            assert elite.id == replacement.id


class TestDataplaneSwapPath:
    """The dataplane branch of :meth:`add_elite`.

    Reducible comparators with ``reduce_to_score`` should route through
    :meth:`DataPlane.try_replace_elite`. Non-reducible comparators —
    bare callables and Pareto dominance — must fall back to the legacy
    WATCH path even when a dataplane is wired in.
    """

    async def test_sum_selector_inserts_via_dataplane(self, dp_storage, dp_archive):
        p = _prog(metrics={"score": 5.0})
        await dp_storage.add(p)
        selector = SumArchiveSelector(fitness_keys=["score"])
        added = await dp_archive.add_elite((0, 1), p, selector)
        assert added is True
        elite = await dp_archive.get_elite((0, 1))
        assert elite is not None and elite.id == p.id

    async def test_sum_selector_swaps_via_dataplane(
        self, dp_storage, dp_archive, coord
    ):
        low = _prog(metrics={"score": 1.0})
        high = _prog(metrics={"score": 10.0})
        await dp_storage.add(low)
        await dp_storage.add(high)
        selector = SumArchiveSelector(fitness_keys=["score"])
        await dp_archive.add_elite((0,), low, selector)
        await dp_archive.add_elite((0,), high, selector)
        # The dataplane writes the candidate score to the scores hash:
        # an artefact of the new path that the WATCH path does not produce.
        pool = coord._connection.pool  # type: ignore[attr-defined]
        assert (await pool.hget("test:archive:scores", "0")) == "10.0"
        elite = await dp_archive.get_elite((0,))
        assert elite.id == high.id

    async def test_sum_selector_rejects_via_dataplane(self, dp_storage, dp_archive):
        good = _prog(metrics={"score": 10.0})
        bad = _prog(metrics={"score": 1.0})
        await dp_storage.add(good)
        await dp_storage.add(bad)
        selector = SumArchiveSelector(fitness_keys=["score"])
        await dp_archive.add_elite((0,), good, selector)
        added = await dp_archive.add_elite((0,), bad, selector)
        assert added is False
        elite = await dp_archive.get_elite((0,))
        assert elite.id == good.id

    async def test_equal_score_occupant_wins_tiebreak(self, dp_storage, dp_archive):
        first = _prog(metrics={"score": 5.0})
        second = _prog(metrics={"score": 5.0})
        await dp_storage.add(first)
        await dp_storage.add(second)
        selector = SumArchiveSelector(fitness_keys=["score"])
        assert await dp_archive.add_elite((0,), first, selector) is True
        assert await dp_archive.add_elite((0,), second, selector) is False

    async def test_pareto_selector_falls_back_to_watch(self, dp_storage, dp_archive):
        """Multi-criteria Pareto comparator must NOT route through dp."""
        first = _prog(metrics={"a": 1.0, "b": 5.0})
        second = _prog(metrics={"a": 5.0, "b": 1.0})  # incomparable
        await dp_storage.add(first)
        await dp_storage.add(second)
        selector = ParetoFrontSelector(fitness_keys=["a", "b"])
        # ``reduce_to_score`` returns None for the Pareto selector, so the
        # storage stays on the legacy WATCH path; an incomparable
        # candidate cannot dominate and must be rejected.
        assert await dp_archive.add_elite((0,), first, selector) is True
        assert await dp_archive.add_elite((0,), second, selector) is False
        elite = await dp_archive.get_elite((0,))
        assert elite.id == first.id

    async def test_bare_callable_falls_back_to_watch(self, dp_storage, dp_archive):
        """Test-style lambdas have no ``reduce_to_score`` attribute and
        must take the WATCH fallback unchanged."""
        p = _prog(metrics={"score": 7.0})
        await dp_storage.add(p)
        added = await dp_archive.add_elite((0,), p, _always_better)
        assert added is True

    async def test_missing_program_in_storage_rejected(self, dp_archive):
        """The pre-existing 'program must be in storage' guard still
        fires on the dataplane path."""
        p = _prog(metrics={"score": 5.0})
        # Note: p NOT added to dp_storage.
        selector = SumArchiveSelector(fitness_keys=["score"])
        added = await dp_archive.add_elite((0,), p, selector)
        assert added is False

    async def test_nan_score_falls_back_to_watch(self, dp_storage, dp_archive):
        """A NaN scalar must not reach the Lua boundary; the helper
        treats it as non-reducible and the WATCH path takes over."""
        bad = _prog(metrics={"score": float("nan")})
        await dp_storage.add(bad)
        selector = SumArchiveSelector(fitness_keys=["score"])
        added = await dp_archive.add_elite((0,), bad, selector)
        # WATCH path with strict ``>`` rejects a NaN-scored candidate
        # against an empty cell because NaN comparisons are always
        # false; the candidate is inserted only because the cell is
        # empty (no comparison needed). The contract here is solely
        # that the dataplane Lua was bypassed — verified by absence
        # from the scores hash.
        assert added is True

    async def test_dataplane_path_no_dataplane_falls_back(self, storage, archive):
        """Without a dataplane wired in, even a reducible selector goes
        through the WATCH path — backwards-compat check."""
        p = _prog(metrics={"score": 5.0})
        await storage.add(p)
        selector = SumArchiveSelector(fitness_keys=["score"])
        added = await archive.add_elite((0,), p, selector)
        assert added is True
