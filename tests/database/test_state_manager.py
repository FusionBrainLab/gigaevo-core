"""ProgramStateManager tests with fakeredis."""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageError,
    StageState,
)
from gigaevo.programs.program_state import ProgramState, merge_states
from tests.conftest import MockOutput

# ===================================================================
# Category A: mark_stage_running
# ===================================================================


class TestMarkStageRunning:
    async def test_mark_stage_running_sets_running_status(
        self, state_manager, make_program
    ):
        """Program's stage_results updated in-memory."""
        prog = make_program()
        await state_manager.storage.add(prog)
        await state_manager.mark_stage_running(prog, "my_stage")

        assert "my_stage" in prog.stage_results
        assert prog.stage_results["my_stage"].status == StageState.RUNNING
        assert prog.stage_results["my_stage"].started_at is not None

    async def test_mark_stage_running_preserves_input_hash(
        self, state_manager, make_program
    ):
        """Existing input_hash carried forward."""
        prog = make_program()
        prog.stage_results["my_stage"] = ProgramStageResult(
            status=StageState.COMPLETED, input_hash="abc123"
        )
        await state_manager.storage.add(prog)
        await state_manager.mark_stage_running(prog, "my_stage")

        assert prog.stage_results["my_stage"].status == StageState.RUNNING
        assert prog.stage_results["my_stage"].input_hash == "abc123"

    async def test_mark_stage_running_not_persisted_to_storage(
        self, state_manager, make_program, fakeredis_storage
    ):
        """RUNNING state is in-memory only — not written to Redis.

        DAG reads stage state from in-memory program.stage_results;
        orphaned RUNNING programs are discarded on restart,
        so persisting RUNNING provides no crash-recovery benefit and costs
        4 Redis round-trips per stage launch.
        """
        prog = make_program()
        await fakeredis_storage.add(prog)
        await state_manager.mark_stage_running(prog, "my_stage")

        # In-memory: stage is RUNNING
        assert prog.stage_results["my_stage"].status == StageState.RUNNING

        # Redis: stage result is NOT written (still has no entry for "my_stage")
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert "my_stage" not in fetched.stage_results


# ===================================================================
# Category B: update_stage_result
# ===================================================================


class TestUpdateStageResult:
    async def test_update_stage_result_persists_to_redis(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Result saved; storage.get() returns it."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        result = ProgramStageResult.success(output=MockOutput(value=99))
        await state_manager.update_stage_result(prog, "my_stage", result)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.stage_results["my_stage"].status == StageState.COMPLETED

    async def test_update_stage_result_with_output(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Output (StageIO) round-trips through Redis."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        output = MockOutput(value=77)
        result = ProgramStageResult.success(output=output)
        await state_manager.update_stage_result(prog, "my_stage", result)

        fetched = await fakeredis_storage.get(prog.id)
        fetched_output = fetched.stage_results["my_stage"].output
        assert fetched_output.value == 77

    async def test_update_stage_result_with_error(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Failed result with StageError persists correctly."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        error = StageError(type="RuntimeError", message="boom", stage="TestStage")
        result = ProgramStageResult.failure(error=error)
        await state_manager.update_stage_result(prog, "my_stage", result)

        fetched = await fakeredis_storage.get(prog.id)
        fetched_res = fetched.stage_results["my_stage"]
        assert fetched_res.status == StageState.FAILED
        assert fetched_res.error.message == "boom"
        assert fetched_res.error.type == "RuntimeError"


# ===================================================================
# Category C: set_program_state
# ===================================================================


class TestSetProgramState:
    async def test_valid_state_transition(
        self, state_manager, make_program, fakeredis_storage
    ):
        """QUEUED -> RUNNING succeeds."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING

    async def test_invalid_state_transition_raises(
        self, state_manager, make_program, fakeredis_storage
    ):
        """QUEUED -> DONE raises ValueError."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        with pytest.raises(ValueError, match="Invalid state transition"):
            await state_manager.set_program_state(prog, ProgramState.DONE)

    async def test_done_to_queued_valid(
        self, state_manager, make_program, fakeredis_storage
    ):
        """DONE -> QUEUED is valid (the refresh path)."""
        prog = make_program(state=ProgramState.DONE)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.QUEUED)
        assert prog.state == ProgramState.QUEUED

    async def test_same_state_noop(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Setting same state is a no-op."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.QUEUED)
        assert prog.state == ProgramState.QUEUED

    async def test_full_lifecycle(self, state_manager, make_program, fakeredis_storage):
        """QUEUED -> RUNNING -> DONE."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING

        await state_manager.set_program_state(prog, ProgramState.DONE)
        assert prog.state == ProgramState.DONE

    async def test_discard_from_any_state(
        self, state_manager, make_program, fakeredis_storage
    ):
        """DISCARDED reachable from QUEUED, RUNNING, DONE."""
        for state in (
            ProgramState.QUEUED,
            ProgramState.RUNNING,
            ProgramState.DONE,
        ):
            prog = make_program(state=state)
            await fakeredis_storage.add(prog)
            await state_manager.set_program_state(prog, ProgramState.DISCARDED)
            assert prog.state == ProgramState.DISCARDED


# ===================================================================
# Category D: Locking & Concurrency
# ===================================================================


class TestLocking:
    async def test_concurrent_updates_serialized(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Two concurrent update_stage_result calls don't corrupt."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        result_a = ProgramStageResult.success(output=MockOutput(value=1))
        result_b = ProgramStageResult.success(output=MockOutput(value=2))

        await asyncio.gather(
            state_manager.update_stage_result(prog, "stage_a", result_a),
            state_manager.update_stage_result(prog, "stage_b", result_b),
        )

        assert "stage_a" in prog.stage_results
        assert "stage_b" in prog.stage_results

    async def test_lock_per_program(self, state_manager, make_program):
        """Different program IDs use different locks."""
        prog_a = make_program()
        prog_b = make_program()

        lock_a = state_manager._lock_for(prog_a.id)
        lock_b = state_manager._lock_for(prog_b.id)

        assert lock_a is not lock_b


# ===================================================================
# Category E: update_program
# ===================================================================


class TestUpdateProgram:
    async def test_update_program_persists_metrics(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Updated metrics survive Redis round-trip."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        prog.add_metrics({"score": 95.5})
        await state_manager.update_program(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metrics["score"] == 95.5

    async def test_update_program_persists_metadata(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Updated metadata survives Redis round-trip."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        prog.set_metadata("experiment", "test-001")
        await state_manager.update_program(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.get_metadata("experiment") == "test-001"


# ===================================================================
# Category F: merge_states
# ===================================================================


class TestMergeStates:
    @pytest.mark.parametrize(
        "current, incoming, expected",
        [
            # DONE <-> QUEUED: the bidirectional refresh pair.
            # Both orderings resolve to QUEUED because DONE→QUEUED is a valid
            # transition: the program should be re-queued regardless of which
            # side of the race won the write.
            (ProgramState.DONE, ProgramState.QUEUED, ProgramState.QUEUED),
            (ProgramState.QUEUED, ProgramState.DONE, ProgramState.QUEUED),
            # QUEUED -> RUNNING: forward pipeline transition.
            # RUNNING wins in either order (a DAG already started should not
            # be rolled back to QUEUED by a stale write).
            (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.RUNNING),
            (ProgramState.RUNNING, ProgramState.QUEUED, ProgramState.RUNNING),
        ],
    )
    def test_merge_state_pairs(
        self, current: ProgramState, incoming: ProgramState, expected: ProgramState
    ) -> None:
        assert merge_states(current, incoming) == expected

    def test_same_state_returns_same(self) -> None:
        for state in (
            ProgramState.QUEUED,
            ProgramState.RUNNING,
            ProgramState.DONE,
            ProgramState.DISCARDED,
        ):
            assert merge_states(state, state) == state

    def test_discarded_always_wins_as_incoming(self) -> None:
        for state in (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE):
            assert merge_states(state, ProgramState.DISCARDED) == ProgramState.DISCARDED

    def test_discarded_always_wins_as_current(self) -> None:
        for state in (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE):
            assert merge_states(ProgramState.DISCARDED, state) == ProgramState.DISCARDED


# ===================================================================
# Category G: Audit Finding 3 — State transitions persist to Redis
# ===================================================================


class TestStateTransitionRedisRoundTrip:
    """Audit finding 3: after set_program_state, re-fetch from Redis must show new state."""

    async def test_queued_to_running_persisted(
        self, state_manager, make_program, fakeredis_storage
    ):
        """QUEUED -> RUNNING is persisted and verified by re-fetch from Redis."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.RUNNING)

        # Re-fetch from Redis
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING

    async def test_running_to_done_persisted(
        self, state_manager, make_program, fakeredis_storage
    ):
        """RUNNING -> DONE is persisted and verified by re-fetch from Redis."""
        prog = make_program(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.DONE)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.DONE

    async def test_done_to_queued_persisted(
        self, state_manager, make_program, fakeredis_storage
    ):
        """DONE -> QUEUED (refresh cycle) is persisted and verified by re-fetch."""
        prog = make_program(state=ProgramState.DONE)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.QUEUED)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.QUEUED

    async def test_discard_persisted(
        self, state_manager, make_program, fakeredis_storage
    ):
        """RUNNING -> DISCARDED is persisted and verified by re-fetch."""
        prog = make_program(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.DISCARDED)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.DISCARDED

    async def test_full_lifecycle_persisted_each_step(
        self, state_manager, make_program, fakeredis_storage
    ):
        """QUEUED -> RUNNING -> DONE: each step is persisted and re-fetchable."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.RUNNING)
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.RUNNING

        await state_manager.set_program_state(prog, ProgramState.DONE)
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.DONE

    async def test_status_sets_updated_after_transition(
        self, state_manager, make_program, fakeredis_storage
    ):
        """After set_program_state, the status set reflects the new state (not old)."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.RUNNING)

        # Verify status set membership via the storage
        queued_count = await fakeredis_storage.count_by_status(
            ProgramState.QUEUED.value
        )
        running_count = await fakeredis_storage.count_by_status(
            ProgramState.RUNNING.value
        )
        assert queued_count == 0
        assert running_count == 1


# ===================================================================
# Category H: Audit Finding 2 — Concurrent writes to same stage key
# ===================================================================


class TestConcurrentSameStageKey:
    """Audit finding 2: multiple concurrent writers on the SAME stage_result key."""

    async def test_concurrent_update_stage_result_same_key(
        self, state_manager, make_program, fakeredis_storage
    ):
        """Multiple concurrent update_stage_result calls targeting the same key.
        The result must be consistent (one of the written values, not corrupted)."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        async def write_stage(value: int):
            result = ProgramStageResult.success(output=MockOutput(value=value))
            await state_manager.update_stage_result(prog, "shared_stage", result)

        # Fire 5 concurrent writes to the same key
        values = list(range(10, 15))
        await asyncio.gather(*(write_stage(v) for v in values))

        # The in-memory program should have the last writer's value
        assert "shared_stage" in prog.stage_results
        assert prog.stage_results["shared_stage"].status == StageState.COMPLETED

        # The Redis-persisted version should be consistent
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert "shared_stage" in fetched.stage_results
        assert fetched.stage_results["shared_stage"].status == StageState.COMPLETED
        final_value = fetched.stage_results["shared_stage"].output.value
        assert final_value in values, (
            f"Final value {final_value} is not one of the expected values {values}"
        )
