"""Concurrent-writer fuzz suite.

Stresses the Lua-CAS primitives under heavy contention to verify that
no two writers can stomp each other:

- ``crdt_inc``: 10 actors × 100 increments must converge to exactly 1000.
- ``transition_program_state``: 20 racers competing for the same
  ``QUEUED → RUNNING`` transition — exactly one wins, the rest see
  ``Err`` (stale or illegal, depending on race ordering).
- ``acquire_instance_lock``: 20 racers competing for the same key —
  exactly one returns ``Ok``, the rest return ``Err(AlreadyHeld)``.
- ``try_replace_elite``: 20 candidates all targeting the same archive
  cell — the highest-scoring candidate wins, never a lower-scoring one.

These tests structurally guard against lost-update regressions on
read-modify-write across the dataplane primitives.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp
from gigaevo.dataplane.coordinator import ProgramPatch
from gigaevo.dataplane.transitions import (
    PROGRAM_STATE_TRANSITIONS,
    load_fsm_table,
)


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
    """One coordinator per test, with FSM table loaded for transitions."""
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="conc")
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    await load_fsm_table(
        fake, key_prefix="conc", name="program_state", table=PROGRAM_STATE_TRANSITIONS
    )
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    try:
        yield coord
    finally:
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await fake.aclose()  # type: ignore[attr-defined]


# ── crdt_inc fuzz ─────────────────────────────────────────────────────


class TestCrdtIncFuzz:
    async def test_many_actors_no_lost_updates(self, coord: dp.DataPlane) -> None:
        """10 actors × 100 increments each → exactly 1000 in the counter."""
        key = dp.CounterKey("fuzz:counter")
        n_actors = 10
        per_actor = 100

        async def writer(i: int) -> None:
            actor = dp.ActorIdentity(
                run_id=dp.RunId("fuzz"), worker_id=dp.WorkerId(f"w{i}")
            )
            for _ in range(per_actor):
                await coord.crdt_inc(key, actor=actor)

        await asyncio.gather(*(writer(i) for i in range(n_actors)))
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        assert read.value.value == n_actors * per_actor

    async def test_same_actor_burst_no_lost_updates(self, coord: dp.DataPlane) -> None:
        """500 concurrent same-actor increments → exactly 500."""
        key = dp.CounterKey("fuzz:hotline")
        actor = dp.ActorIdentity(run_id=dp.RunId("fuzz"), worker_id=dp.WorkerId("only"))
        n = 500
        await asyncio.gather(*(coord.crdt_inc(key, actor=actor) for _ in range(n)))
        read = await coord.crdt_read(key)
        assert isinstance(read, dp.Ok)
        assert read.value.value == n


# ── transition_program_state race ─────────────────────────────────────


async def _put_program(coord: dp.DataPlane, pid: str, state: dp.ProgramState) -> None:
    blob = {"id": pid, "state": state.value, "epoch": 0}
    await coord._connection.pool.set(  # type: ignore[misc]
        f"conc:program:{pid}", json.dumps(blob)
    )


class TestTransitionRace:
    async def test_idempotent_contention_returns_same_epoch(
        self, coord: dp.DataPlane
    ) -> None:
        """20 racers with the same ``(pid, from, to, patch)``.

        The Lua idempotency hash keys off the canonical hash of those
        four values; the first racer through bumps the epoch and writes
        the blob, every subsequent racer's HEXISTS hits and the script
        returns ``duplicate`` carrying the just-written blob. The Python
        wrapper maps ``duplicate`` to :class:`Ok` so every racer's view
        of the post-write state is consistent — and crucially every
        ``Ok`` carries the *same* epoch, i.e. the transition fired
        exactly once on the server.
        """
        pid = "race-idem"
        await _put_program(coord, pid, dp.ProgramState.QUEUED)

        async def racer(_: int) -> dp.Result[dp.Versioned[dict[str, object]], object]:
            return await coord.transition_program_state(
                dp.ProgramId(pid),
                token=dp.mint_root(dp.ProgramId(pid)),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
                patch=ProgramPatch(),
            )

        results = await asyncio.gather(*(racer(i) for i in range(20)))
        assert all(isinstance(r, dp.Ok) for r in results)
        epochs = {r.value.epoch for r in results if isinstance(r, dp.Ok)}
        # Idempotency: every Ok carries the same epoch.
        assert len(epochs) == 1
        # Program landed in RUNNING.
        raw = await coord._connection.pool.get(f"conc:program:{pid}")  # type: ignore[misc]
        assert json.loads(raw)["state"] == "RUNNING"

    async def test_distinct_patches_exactly_one_winner(
        self, coord: dp.DataPlane
    ) -> None:
        """20 racers each carry a unique patch — exactly one transition fires.

        With distinct patches the idempotency hashes differ, so the
        Lua dedup path does not trigger. The expected_from check then
        gates: only the racer whose attempt reaches the script before
        any other's epoch bump sees ``from == QUEUED`` and returns
        ``Ok``. Everyone else hits the expected_from mismatch and
        returns ``Err(stale)``.
        """
        pid = "race-distinct"
        await _put_program(coord, pid, dp.ProgramState.QUEUED)

        async def racer(i: int) -> dp.Result[dp.Versioned[dict[str, object]], object]:
            return await coord.transition_program_state(
                dp.ProgramId(pid),
                token=dp.mint_root(dp.ProgramId(pid)),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
                patch=ProgramPatch(fields={"writer": i}),
            )

        results = await asyncio.gather(*(racer(i) for i in range(20)))
        wins = [r for r in results if isinstance(r, dp.Ok)]
        losses = [r for r in results if isinstance(r, dp.Err)]
        assert len(wins) == 1, f"expected 1 winner, got {len(wins)}"
        assert len(losses) == 19
        # The losing racers all failed because expected_from no longer matched.
        assert all(
            isinstance(loss, dp.Err) and loss.error.kind == "stale"  # type: ignore[union-attr]
            for loss in losses
        )

    async def test_distinct_programs_all_succeed(self, coord: dp.DataPlane) -> None:
        """N independent programs racing in parallel — every transition wins."""
        n = 20
        for i in range(n):
            await _put_program(coord, f"indep-{i}", dp.ProgramState.QUEUED)

        async def racer(i: int) -> dp.Result[dp.Versioned[dict[str, object]], object]:
            pid = f"indep-{i}"
            return await coord.transition_program_state(
                dp.ProgramId(pid),
                token=dp.mint_root(dp.ProgramId(pid)),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
                patch=ProgramPatch(),
            )

        results = await asyncio.gather(*(racer(i) for i in range(n)))
        assert all(isinstance(r, dp.Ok) for r in results)


# ── acquire_instance_lock race ────────────────────────────────────────


class TestLockRace:
    async def test_only_one_acquirer_wins(self, coord: dp.DataPlane) -> None:
        """20 racers, one key: exactly one returns ``Ok``."""
        # The coordinator's key-component validator rejects colons in
        # the prefix (they collide with the ``{key_prefix}:lock:`` Redis
        # namespacing convention), so this fixture passes a flat slug.
        key = "critical-section"

        async def racer() -> dp.Result[dp.InstanceLease, object]:
            return await coord.acquire_instance_lock(key, ttl_s=5.0)

        results = await asyncio.gather(*(racer() for _ in range(20)))
        wins = [r for r in results if isinstance(r, dp.Ok)]
        losses = [r for r in results if isinstance(r, dp.Err)]
        assert len(wins) == 1
        assert len(losses) == 19


# ── try_replace_elite race ────────────────────────────────────────────


class TestArchiveSwapRace:
    async def test_highest_score_wins(self, coord: dp.DataPlane) -> None:
        """20 candidates of varying scores on the same cell.

        Cell winner is whichever candidate landed last among those whose
        score beat the prior occupant — but the *value* must end up at
        the global max score across the batch (lower scores cannot
        displace a higher-scored occupant).
        """
        cell = dp.CellKey("hotspot")
        n = 20

        async def candidate(i: int) -> dp.Result[dp.EliteSwapOutcome, object]:
            return await coord.try_replace_elite(
                cell,
                dp.ProgramId(f"cand-{i:02d}"),
                token=dp.mint_root(cell),
                candidate_score=float(i),
                tiebreak_bit=0,
            )

        await asyncio.gather(*(candidate(i) for i in range(n)))
        pool = coord._connection.pool  # type: ignore[attr-defined]
        occupant = await pool.hget("conc:archive", cell)  # type: ignore[misc]
        score = float(await pool.hget("conc:archive:scores", cell))  # type: ignore[misc]
        # The maximum score in the batch is n-1; the cell must reflect it.
        assert score == float(n - 1)
        assert occupant == f"cand-{n - 1:02d}"
