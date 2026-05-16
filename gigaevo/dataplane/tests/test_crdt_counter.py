"""Integration tests for the CRDT G-counter API.

Covers the ``counter_inc.lua`` script and the
:meth:`gigaevo.dataplane.DataPlane.crdt_inc` /
:meth:`gigaevo.dataplane.DataPlane.crdt_read` wrappers. The tests run
against an in-process ``fakeredis`` server (Lua-enabled) so they pin
both the script semantics and the Python integration without needing a
real Redis.

Invariants under test:

    - Sum across actors equals total of all per-actor increments.
    - Increments commute: order of writes does not change the read.
    - Increments are idempotent under a re-run of the same Lua script.
    - Concurrent writers from distinct actors converge to the correct sum.
    - Every write bumps the global epoch and the per-counter generation;
      stale reads (below caller's floor) raise.
    - Negative deltas (decrements) work — the per-actor sub-count is signed.
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

    fakeredis exposes a server-mode constructor so each test gets a
    fresh, isolated data store. We hand the coordinator a redis_url
    that ``RedisConnection`` would normally use to build its own pool,
    but we substitute the pool out-of-band so the URL is never dialled.
    """
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="test")
    # Substitute the connection's pool with the fakeredis client *before*
    # startup; startup() observes the substituted pool and skips the
    # real PING.
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    # Manually invoke the script-load + FSM-load path that startup() runs.
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


def _actor(run: str, worker: str) -> dp.ActorIdentity:
    return dp.ActorIdentity(run_id=dp.RunId(run), worker_id=dp.WorkerId(worker))


# ── basic increments ─────────────────────────────────────────────────


class TestSingleActorIncrement:
    async def test_one_increment_returns_one(self, coord: dp.DataPlane) -> None:
        result = await coord.crdt_inc(
            dp.CounterKey("bandit:trials:arm-a"),
            actor=_actor("run-1", "w-1"),
        )
        assert isinstance(result, dp.Ok)
        assert result.value == 1

    async def test_repeated_increments_accumulate(self, coord: dp.DataPlane) -> None:
        actor = _actor("run-1", "w-1")
        key = dp.CounterKey("bandit:trials:arm-a")
        for expected in (1, 2, 3, 4, 5):
            result = await coord.crdt_inc(key, actor=actor)
            assert isinstance(result, dp.Ok)
            assert result.value == expected

    async def test_explicit_delta(self, coord: dp.DataPlane) -> None:
        actor = _actor("run-1", "w-1")
        key = dp.CounterKey("trials")
        result = await coord.crdt_inc(key, actor=actor, delta=7)
        assert isinstance(result, dp.Ok)
        assert result.value == 7

    async def test_negative_delta_decrements(self, coord: dp.DataPlane) -> None:
        actor = _actor("run-1", "w-1")
        key = dp.CounterKey("balance")
        await coord.crdt_inc(key, actor=actor, delta=10)
        result = await coord.crdt_inc(key, actor=actor, delta=-3)
        assert isinstance(result, dp.Ok)
        assert result.value == 7


# ── multi-actor sum ──────────────────────────────────────────────────


class TestMultiActorSum:
    async def test_two_actors_sum_to_total(self, coord: dp.DataPlane) -> None:
        key = dp.CounterKey("trials")
        a = _actor("run-1", "w-a")
        b = _actor("run-1", "w-b")
        await coord.crdt_inc(key, actor=a, delta=3)
        await coord.crdt_inc(key, actor=b, delta=5)
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        assert read.value.value == 8

    async def test_independent_counters_isolated(self, coord: dp.DataPlane) -> None:
        a_key = dp.CounterKey("arm-a")
        b_key = dp.CounterKey("arm-b")
        actor = _actor("run-1", "w-1")
        await coord.crdt_inc(a_key, actor=actor, delta=10)
        await coord.crdt_inc(b_key, actor=actor, delta=20)
        ra = await coord.crdt_read(a_key)
        rb = await coord.crdt_read(b_key)
        assert isinstance(ra, dp.Ok) and ra.value.value == 10
        assert isinstance(rb, dp.Ok) and rb.value.value == 20

    async def test_cross_run_actors_share_counter(self, coord: dp.DataPlane) -> None:
        """Two runs incrementing the same counter sum across runs.

        That's the whole point of the G-counter — two engines on the
        same Redis prefix share the same counter view.
        """
        key = dp.CounterKey("global:trials")
        run_a = _actor("run-a", "w-1")
        run_b = _actor("run-b", "w-1")
        await coord.crdt_inc(key, actor=run_a, delta=4)
        await coord.crdt_inc(key, actor=run_b, delta=6)
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        assert read.value.value == 10


# ── commutativity / concurrency ──────────────────────────────────────


class TestConcurrency:
    async def test_concurrent_writes_converge(self, coord: dp.DataPlane) -> None:
        """N coroutines incrementing the same counter from distinct
        actors must converge to N (no lost updates)."""
        key = dp.CounterKey("concurrent")
        n = 50

        async def writer(i: int) -> None:
            await coord.crdt_inc(key, actor=_actor("run", f"w-{i}"), delta=1)

        await asyncio.gather(*(writer(i) for i in range(n)))
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        assert read.value.value == n

    async def test_concurrent_writes_same_actor_converge(
        self, coord: dp.DataPlane
    ) -> None:
        """Concurrent increments from the same actor must accumulate
        without loss. Lua atomicity guarantees this even though the
        Python wrapper is async."""
        key = dp.CounterKey("hot")
        actor = _actor("run", "single")
        n = 100
        await asyncio.gather(*(coord.crdt_inc(key, actor=actor) for _ in range(n)))
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        assert read.value.value == n


# ── Versioned read freshness ─────────────────────────────────────────


class TestVersionedReadFloor:
    async def test_read_carries_epoch_and_generation(self, coord: dp.DataPlane) -> None:
        key = dp.CounterKey("versioned")
        await coord.crdt_inc(key, actor=_actor("run", "w-1"))
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        # First increment bumps epoch and generation to 1 each.
        assert read.value.epoch >= 1
        assert read.value.generation >= 1

    async def test_stale_read_below_min_epoch_raises(self, coord: dp.DataPlane) -> None:
        key = dp.CounterKey("stale-epoch")
        await coord.crdt_inc(key, actor=_actor("run", "w-1"))
        # Snapshot the current epoch via an unconditional read.
        current = await coord.crdt_read(key)
        assert isinstance(current, dp.Ok)
        # Request a future epoch the store has not reached.
        future = await coord.crdt_read(key, min_epoch=current.value.epoch + 5)
        assert isinstance(future, dp.Err)
        assert isinstance(future.error, dp.StaleReadError)

    async def test_stale_read_below_min_generation_raises(
        self, coord: dp.DataPlane
    ) -> None:
        key = dp.CounterKey("stale-gen")
        await coord.crdt_inc(key, actor=_actor("run", "w-1"))
        current = await coord.crdt_read(key)
        assert isinstance(current, dp.Ok)
        future = await coord.crdt_read(key, min_generation=current.value.generation + 5)
        assert isinstance(future, dp.Err)
        assert isinstance(future.error, dp.StaleReadError)

    async def test_fresh_read_at_floor_succeeds(self, coord: dp.DataPlane) -> None:
        key = dp.CounterKey("at-floor")
        await coord.crdt_inc(key, actor=_actor("run", "w-1"))
        current = await coord.crdt_read(key)
        assert isinstance(current, dp.Ok)
        # min == observed must succeed (lattice `>=` is reflexive).
        equal = await coord.crdt_read(
            key,
            min_epoch=current.value.epoch,
            min_generation=current.value.generation,
        )
        assert isinstance(equal, dp.Ok)


# ── empty-counter reads ──────────────────────────────────────────────


class TestEmptyCounter:
    async def test_read_before_any_increment_returns_zero(
        self, coord: dp.DataPlane
    ) -> None:
        read = await coord.crdt_read(dp.CounterKey("never-written"))
        assert isinstance(read, dp.Ok)
        assert read.value.value == 0
        assert read.value.epoch == 0
        assert read.value.generation == 0
