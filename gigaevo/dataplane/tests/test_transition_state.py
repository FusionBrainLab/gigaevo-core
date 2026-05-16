"""Integration tests for the program-FSM transition API.

Covers ``transition_state.lua`` plus :meth:`DataPlane.transition_program_state`
and :meth:`DataPlane.read_program`. The Lua script does the FSM check,
patch merging, status-set update, and event-stream append in one atomic
round-trip; tests verify each invariant against fakeredis.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json

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
    from gigaevo.dataplane.transitions import (
        PROGRAM_STATE_TRANSITIONS,
        load_fsm_table,
    )

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    await load_fsm_table(
        fake, key_prefix="test", name="program_state", table=PROGRAM_STATE_TRANSITIONS
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


async def _put_program(
    coord: dp.DataPlane, pid: str, state: dp.ProgramState, **extra: object
) -> None:
    """Pre-populate a program blob via the underlying pool.

    The transition Lua expects an existing blob; there is no
    add_program method on DataPlane yet (that lands with the program
    storage rewrite), so tests inject the blob directly here.
    """
    blob: dict[str, object] = {"id": pid, "state": state.value, "epoch": 0, **extra}
    await coord._connection.pool.set(  # type: ignore[misc]
        f"test:program:{pid}", json.dumps(blob)
    )


def _token(pid: str) -> dp.Token[dp.ProgramId]:
    return dp.mint_root(dp.ProgramId(pid))


# ── happy-path transitions ───────────────────────────────────────────


class TestHappyPath:
    async def test_queued_to_running(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-1", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("p-1"),
            token=_token("p-1"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "RUNNING"
        assert result.value.epoch >= 1

    async def test_running_to_done(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-2", dp.ProgramState.RUNNING)
        result = await coord.transition_program_state(
            dp.ProgramId("p-2"),
            token=_token("p-2"),
            expected_from=dp.ProgramState.RUNNING,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "DONE"

    async def test_expected_from_none_matches_any(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-3", dp.ProgramState.RUNNING)
        result = await coord.transition_program_state(
            dp.ProgramId("p-3"),
            token=_token("p-3"),
            expected_from=None,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "DONE"

    async def test_patch_merged_into_blob(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-4", dp.ProgramState.QUEUED, score=0.0)
        result = await coord.transition_program_state(
            dp.ProgramId("p-4"),
            token=_token("p-4"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"score": 1.5, "worker": "w-1"}),
        )
        assert isinstance(result, dp.Ok)
        blob = result.value.value
        assert blob["score"] == 1.5
        assert blob["worker"] == "w-1"
        assert blob["state"] == "RUNNING"


# ── error paths ──────────────────────────────────────────────────────


class TestErrorPaths:
    async def test_missing_program_returns_stale(self, coord: dp.DataPlane) -> None:
        result = await coord.transition_program_state(
            dp.ProgramId("never-existed"),
            token=_token("never-existed"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "stale"
        assert "program not found" in result.error.detail

    async def test_expected_from_mismatch_returns_stale(
        self, coord: dp.DataPlane
    ) -> None:
        await _put_program(coord, "p-5", dp.ProgramState.RUNNING)
        result = await coord.transition_program_state(
            dp.ProgramId("p-5"),
            token=_token("p-5"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "stale"
        assert "expected_from=QUEUED" in result.error.detail

    async def test_illegal_transition_skips_states(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-6", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("p-6"),
            token=_token("p-6"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "illegal"
        assert "QUEUED -> DONE" in result.error.detail

    async def test_discarded_is_terminal(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-7", dp.ProgramState.DISCARDED)
        for target in (
            dp.ProgramState.QUEUED,
            dp.ProgramState.RUNNING,
            dp.ProgramState.DONE,
        ):
            result = await coord.transition_program_state(
                dp.ProgramId("p-7"),
                token=_token("p-7"),
                expected_from=None,
                to=target,
            )
            assert isinstance(result, dp.Err)
            assert result.error.kind == "illegal"

    async def test_token_pid_mismatch_returns_unknown(
        self, coord: dp.DataPlane
    ) -> None:
        await _put_program(coord, "p-8", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("p-8"),
            token=_token("a-different-pid"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "unknown"


# ── idempotency ──────────────────────────────────────────────────────


class TestIdempotency:
    async def test_replay_returns_duplicate_same_blob(
        self, coord: dp.DataPlane
    ) -> None:
        await _put_program(coord, "p-9", dp.ProgramState.QUEUED)
        first = await coord.transition_program_state(
            dp.ProgramId("p-9"),
            token=_token("p-9"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"trial": 1}),
        )
        assert isinstance(first, dp.Ok)
        second = await coord.transition_program_state(
            dp.ProgramId("p-9"),
            token=_token("p-9"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"trial": 1}),
        )
        assert isinstance(second, dp.Ok)
        assert first.value.value == second.value.value
        assert first.value.epoch == second.value.epoch


# ── side effects: status sets + events stream ────────────────────────


class TestSideEffects:
    async def test_status_sets_updated(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-11", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-11"),
            token=_token("p-11"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        queued = await pool.smembers("test:status:QUEUED")  # type: ignore[misc]
        running = await pool.smembers("test:status:RUNNING")  # type: ignore[misc]
        assert "p-11" not in queued
        assert "p-11" in running

    async def test_status_event_emitted(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-12", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-12"),
            token=_token("p-12"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        events = await pool.xrange("test:status_events")  # type: ignore[misc]
        assert len(events) >= 1
        _, last_fields = events[-1]
        assert last_fields["pid"] == "p-12"
        assert last_fields["from"] == "QUEUED"
        assert last_fields["to"] == "RUNNING"


# ── read_program ─────────────────────────────────────────────────────


class TestReadProgram:
    async def test_missing_returns_ok_none(self, coord: dp.DataPlane) -> None:
        result = await coord.read_program(dp.ProgramId("does-not-exist"))
        assert isinstance(result, dp.Ok)
        assert result.value is None

    async def test_existing_returns_versioned(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-13", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-13"),
            token=_token("p-13"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        result = await coord.read_program(dp.ProgramId("p-13"))
        assert isinstance(result, dp.Ok)
        assert result.value is not None
        assert result.value.value["state"] == "RUNNING"
        assert result.value.epoch >= 1

    async def test_stale_floor_raises(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-14", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-14"),
            token=_token("p-14"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        current = await coord.read_program(dp.ProgramId("p-14"))
        assert isinstance(current, dp.Ok) and current.value is not None
        future = await coord.read_program(
            dp.ProgramId("p-14"), min_epoch=current.value.epoch + 10
        )
        assert isinstance(future, dp.Err)
        assert isinstance(future.error, dp.StaleReadError)


# ── token discipline ─────────────────────────────────────────────────


class TestTokenDiscipline:
    async def test_token_consumed_on_call(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-15", dp.ProgramState.QUEUED)
        token = _token("p-15")
        await coord.transition_program_state(
            dp.ProgramId("p-15"),
            token=token,
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert token.consumed

    async def test_reused_token_raises(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-16", dp.ProgramState.QUEUED)
        token = _token("p-16")
        await coord.transition_program_state(
            dp.ProgramId("p-16"),
            token=token,
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        with pytest.raises(dp.TokenAlreadyConsumed):
            await coord.transition_program_state(
                dp.ProgramId("p-16"),
                token=token,
                expected_from=dp.ProgramState.RUNNING,
                to=dp.ProgramState.DONE,
            )


# ── batch transitions ────────────────────────────────────────────────


class TestBatch:
    async def test_empty_batch_returns_ok_empty(self, coord: dp.DataPlane) -> None:
        result = await coord.transition_program_state_batch(items=())
        assert isinstance(result, dp.Ok)
        assert result.value.items == ()

    async def test_batch_applies_all_in_order(self, coord: dp.DataPlane) -> None:
        for i in range(1, 4):
            await _put_program(coord, f"b-{i}", dp.ProgramState.QUEUED)
        items = tuple(
            dp.BatchTransitionItem(
                program_id=dp.ProgramId(f"b-{i}"),
                token=_token(f"b-{i}"),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            )
            for i in range(1, 4)
        )
        result = await coord.transition_program_state_batch(items=items)
        assert isinstance(result, dp.Ok)
        outcomes = result.value.items
        assert len(outcomes) == 3
        for v, expected_pid in zip(outcomes, ("b-1", "b-2", "b-3")):
            assert v.value["id"] == expected_pid
            assert v.value["state"] == "RUNNING"

    async def test_batch_per_item_token_consumed(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "b-x", dp.ProgramState.QUEUED)
        await _put_program(coord, "b-y", dp.ProgramState.QUEUED)
        tx = _token("b-x")
        ty = _token("b-y")
        items = (
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-x"),
                token=tx,
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-y"),
                token=ty,
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
        )
        result = await coord.transition_program_state_batch(items=items)
        assert isinstance(result, dp.Ok)
        assert tx.consumed and ty.consumed

    async def test_batch_partial_failure_returns_err_after_partial_commit(
        self, coord: dp.DataPlane
    ) -> None:
        # b-good is QUEUED and will succeed; b-bad doesn't exist so the
        # second item fails. The first item is already committed when
        # the second fails — the batch is not atomic across items.
        await _put_program(coord, "b-good", dp.ProgramState.QUEUED)
        items = (
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-good"),
                token=_token("b-good"),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-bad"),  # never written
                token=_token("b-bad"),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
        )
        result = await coord.transition_program_state_batch(items=items)
        assert isinstance(result, dp.Err)
        assert result.error.kind == "stale"
        # The first item's commit survived.
        read = await coord.read_program(dp.ProgramId("b-good"))
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value["state"] == "RUNNING"
