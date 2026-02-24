"""Tests for state consistency in concurrent scenarios.

Covers:
- atomic_state_transition uses merged state for status sets (Bug #8 fix)
- ProgramStateManager._locks eviction for DONE programs
- Race between concurrent DONE and DISCARDED transitions
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _prog(state=ProgramState.QUEUED):
    return Program(
        code="def solve(): return 42", state=state, atomic_counter=999_999_999
    )


# ---------------------------------------------------------------------------
# Tests: atomic_state_transition uses merged state for status sets
# ---------------------------------------------------------------------------


class TestAtomicStateTransitionMergedState:
    """Verify that atomic_state_transition uses the merged state (not caller's
    new_state) for status set operations, preventing dual-set membership."""

    async def test_done_transition_when_already_discarded_uses_discarded_set(
        self, fakeredis_storage
    ):
        """If the program in Redis is already DISCARDED, a DONE transition should
        resolve to DISCARDED (via merge) and put the program in discarded_set,
        NOT done_set."""
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        # Simulate _maintain setting DISCARDED first
        prog_copy = _prog(state=ProgramState.RUNNING)
        prog_copy.id = prog.id
        prog_copy.state = ProgramState.DISCARDED
        await fakeredis_storage.atomic_state_transition(
            prog_copy, "running", "discarded"
        )

        # Verify program is in discarded set
        discarded_ids = await fakeredis_storage.get_ids_by_status("discarded")
        assert prog.id in discarded_ids

        # Now simulate _execute_dag trying to set DONE (the race)
        prog_done = _prog(state=ProgramState.RUNNING)
        prog_done.id = prog.id
        prog_done.state = ProgramState.DONE
        await fakeredis_storage.atomic_state_transition(prog_done, "running", "done")

        # CRITICAL: program should be in discarded set ONLY, not done set
        discarded_ids = await fakeredis_storage.get_ids_by_status("discarded")
        done_ids = await fakeredis_storage.get_ids_by_status("done")
        running_ids = await fakeredis_storage.get_ids_by_status("running")

        assert prog.id in discarded_ids, "Program should be in discarded set"
        assert prog.id not in done_ids, "Program should NOT be in done set"
        assert prog.id not in running_ids, "Program should NOT be in running set"

        # Verify the stored program data also has DISCARDED state
        stored = await fakeredis_storage.get(prog.id)
        assert stored.state == ProgramState.DISCARDED

    async def test_normal_transition_still_works(self, fakeredis_storage):
        """Normal RUNNING -> DONE transition (no race) should work correctly."""
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        prog.state = ProgramState.DONE
        await fakeredis_storage.atomic_state_transition(prog, "running", "done")

        done_ids = await fakeredis_storage.get_ids_by_status("done")
        running_ids = await fakeredis_storage.get_ids_by_status("running")

        assert prog.id in done_ids
        assert prog.id not in running_ids

    async def test_discarded_transition_cleans_up_running_set(self, fakeredis_storage):
        """RUNNING -> DISCARDED should remove from running set."""
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        prog.state = ProgramState.DISCARDED
        await fakeredis_storage.atomic_state_transition(prog, "running", "discarded")

        discarded_ids = await fakeredis_storage.get_ids_by_status("discarded")
        running_ids = await fakeredis_storage.get_ids_by_status("running")

        assert prog.id in discarded_ids
        assert prog.id not in running_ids

    async def test_concurrent_done_and_discard_exactly_one_set(self, fakeredis_storage):
        """Simulate concurrent DONE and DISCARD transitions with asyncio.gather.
        The program should end up in exactly ONE status set."""
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        # Two concurrent transitions
        prog_done = _prog(state=ProgramState.RUNNING)
        prog_done.id = prog.id
        prog_done.state = ProgramState.DONE

        prog_discard = _prog(state=ProgramState.RUNNING)
        prog_discard.id = prog.id
        prog_discard.state = ProgramState.DISCARDED

        await asyncio.gather(
            fakeredis_storage.atomic_state_transition(prog_done, "running", "done"),
            fakeredis_storage.atomic_state_transition(
                prog_discard, "running", "discarded"
            ),
        )

        done_ids = await fakeredis_storage.get_ids_by_status("done")
        discarded_ids = await fakeredis_storage.get_ids_by_status("discarded")
        running_ids = await fakeredis_storage.get_ids_by_status("running")

        in_done = prog.id in done_ids
        in_discarded = prog.id in discarded_ids
        in_running = prog.id in running_ids

        # Program should be in exactly one set
        assert not in_running, "Should not remain in running set"
        assert in_done != in_discarded, (
            f"Should be in exactly one set: done={in_done}, discarded={in_discarded}"
        )

    async def test_stale_old_state_cleaned_up(self, fakeredis_storage):
        """If existing program in Redis has a different state than old_state,
        both the old_state set and the existing state set get cleaned up."""
        prog = _prog(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Transition QUEUED -> RUNNING
        prog.state = ProgramState.RUNNING
        await fakeredis_storage.atomic_state_transition(prog, "queued", "running")

        running_ids = await fakeredis_storage.get_ids_by_status("running")
        queued_ids = await fakeredis_storage.get_ids_by_status("queued")
        assert prog.id in running_ids
        assert prog.id not in queued_ids

        # Now transition RUNNING -> DONE
        prog.state = ProgramState.DONE
        await fakeredis_storage.atomic_state_transition(prog, "running", "done")

        done_ids = await fakeredis_storage.get_ids_by_status("done")
        running_ids = await fakeredis_storage.get_ids_by_status("running")
        assert prog.id in done_ids
        assert prog.id not in running_ids


# ---------------------------------------------------------------------------
# Tests: ProgramStateManager._locks eviction
# ---------------------------------------------------------------------------


class TestLocksEviction:
    async def test_locks_evicted_on_discarded(self, fakeredis_storage):
        """Lock should be evicted when program reaches DISCARDED state."""
        sm = ProgramStateManager(fakeredis_storage)
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        await sm.set_program_state(prog, ProgramState.DISCARDED)
        assert prog.id not in sm._locks

    async def test_locks_evicted_on_done(self, fakeredis_storage):
        """Lock should be evicted when program reaches DONE state (memory fix)."""
        sm = ProgramStateManager(fakeredis_storage)
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        await sm.set_program_state(prog, ProgramState.DONE)
        assert prog.id not in sm._locks

    async def test_locks_not_evicted_on_running(self, fakeredis_storage):
        """Lock should NOT be evicted when program is in RUNNING state."""
        sm = ProgramStateManager(fakeredis_storage)
        prog = _prog(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await sm.set_program_state(prog, ProgramState.RUNNING)
        assert prog.id in sm._locks

    async def test_locks_dont_grow_unbounded(self, fakeredis_storage):
        """After cycling many programs through DONE, locks dict stays small."""
        sm = ProgramStateManager(fakeredis_storage)

        for i in range(50):
            prog = _prog(state=ProgramState.RUNNING)
            await fakeredis_storage.add(prog)
            await sm.set_program_state(prog, ProgramState.DONE)

        # All locks should be evicted since all programs reached DONE
        assert len(sm._locks) == 0


# ---------------------------------------------------------------------------
# Tests: Full ProgramStateManager state transition consistency
# ---------------------------------------------------------------------------


class TestStateManagerConsistency:
    async def test_set_program_state_idempotent(self, fakeredis_storage):
        """Setting the same state twice is a no-op."""
        sm = ProgramStateManager(fakeredis_storage)
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        await sm.set_program_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING

    async def test_invalid_transition_raises(self, fakeredis_storage):
        """Invalid transition (e.g., DONE -> RUNNING) raises ValueError."""
        sm = ProgramStateManager(fakeredis_storage)
        prog = _prog(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)
        await sm.set_program_state(prog, ProgramState.DONE)

        with pytest.raises(ValueError, match="Invalid state transition"):
            await sm.set_program_state(prog, ProgramState.RUNNING)

    async def test_concurrent_stage_updates_serialized(self, fakeredis_storage):
        """Two concurrent update_stage_result calls for the same program are
        serialized by the lock — no lost updates."""
        from gigaevo.programs.core_types import ProgramStageResult, StageState

        sm = ProgramStateManager(fakeredis_storage)
        prog = _prog(state=ProgramState.RUNNING)
        prog.stage_results = {
            "stage_a": ProgramStageResult(status=StageState.PENDING),
            "stage_b": ProgramStageResult(status=StageState.PENDING),
        }
        await fakeredis_storage.add(prog)

        result_a = ProgramStageResult(status=StageState.COMPLETED)
        result_b = ProgramStageResult(status=StageState.COMPLETED)

        await asyncio.gather(
            sm.update_stage_result(prog, "stage_a", result_a),
            sm.update_stage_result(prog, "stage_b", result_b),
        )

        # Both stage results should be present
        assert prog.stage_results["stage_a"].status == StageState.COMPLETED
        assert prog.stage_results["stage_b"].status == StageState.COMPLETED
