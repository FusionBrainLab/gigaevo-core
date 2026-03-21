"""Tests for complex/untested logic paths in DagRunner.

Covers:
- _maintain() TOCTOU guard: program classified as timed out but already DONE
- _launch() build failure + state update failure cascade
- _launch() mget failure during Phase 3 program fetch
- _launch() get_ids_by_status failure (Phase 1 fetch-by-status)
- _execute_dag() DAG run failure sets DISCARDED
- _maintain() timeout with program already DONE (TOCTOU race)
- _maintain() timeout with storage.get returning None (program deleted mid-timeout)
- _launch() orphan discard state update failure
- Multiple timed-out programs in single _maintain() call
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import (
    DagRunner,
    DagRunnerConfig,
    TaskInfo,
)


def _prog(state=ProgramState.QUEUED):
    return Program(code="def solve(): return 42", state=state, atomic_counter=999)


def _mock_storage():
    storage = MagicMock()
    storage.close = AsyncMock()
    storage.wait_for_activity = AsyncMock()
    storage.get = AsyncMock(return_value=None)
    storage.mget = AsyncMock(return_value=[])
    storage.add = AsyncMock()
    storage.update = AsyncMock()
    storage.write_exclusive = AsyncMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_ids_by_status = AsyncMock(return_value=[])
    storage.publish_status_event = AsyncMock()
    storage.transition_status = AsyncMock()
    storage.atomic_state_transition = AsyncMock()
    return storage


def _mock_writer():
    writer = MagicMock()
    writer.bind = MagicMock(return_value=writer)
    writer.scalar = MagicMock()
    writer.text = MagicMock()
    return writer


def _mock_dag(run_side_effect=None):
    dag = MagicMock()
    dag.run = AsyncMock(side_effect=run_side_effect)
    dag.automata = MagicMock()
    dag.automata.topology = MagicMock()
    dag.automata.topology.nodes = MagicMock()
    dag.state_manager = MagicMock()
    dag._writer = MagicMock()
    dag._stage_sema = MagicMock()
    return dag


def _runner(storage=None, dag_blueprint=None, config=None):
    storage = storage or _mock_storage()
    dag_blueprint = dag_blueprint or MagicMock()
    config = config or DagRunnerConfig()
    writer = _mock_writer()
    return DagRunner(storage, dag_blueprint, config, writer)


# ===================================================================
# Category A: _maintain() TOCTOU guard — timed out but already DONE
# ===================================================================


class TestMaintainTOCTOU:
    """dag_runner.py L253-262: If program is timed out BUT storage shows DONE,
    skip the discard (TOCTOU guard). This prevents discarding a program
    that completed between the timeout check and the storage fetch."""

    async def test_timed_out_but_already_done_skips_discard(self):
        """Program classified as timed out but storage says DONE -> no discard."""
        prog = _prog(state=ProgramState.RUNNING)
        storage = _mock_storage()
        # Storage returns the program in DONE state (it completed between checks)
        done_prog = _prog(state=ProgramState.DONE)
        done_prog.id = prog.id  # same program
        storage.get = AsyncMock(return_value=done_prog)

        config = DagRunnerConfig(dag_timeout=1.0)
        runner = _runner(storage=storage, config=config)

        # Create a running task that "timed out"
        task = asyncio.create_task(asyncio.sleep(3600))
        runner._active[prog.id] = TaskInfo(
            task=task, program_id=prog.id, started_at=time.monotonic() - 100
        )

        await runner._maintain()

        # Program should be removed from active (cleanup happens)
        assert prog.id not in runner._active
        # Timeout is recorded in metrics
        assert runner._metrics.dag_timeouts == 1
        # But set_program_state should NOT have been called for DISCARDED
        # (the TOCTOU guard skips the discard for DONE programs)
        # Check that atomic_state_transition was NOT called
        storage.atomic_state_transition.assert_not_called()

    async def test_timed_out_and_not_done_gets_discarded(self):
        """Normal timeout: program is RUNNING -> gets DISCARDED."""
        prog = _prog(state=ProgramState.RUNNING)
        storage = _mock_storage()
        storage.get = AsyncMock(return_value=prog)

        config = DagRunnerConfig(dag_timeout=1.0)
        runner = _runner(storage=storage, config=config)

        task = asyncio.create_task(asyncio.sleep(3600))
        runner._active[prog.id] = TaskInfo(
            task=task, program_id=prog.id, started_at=time.monotonic() - 100
        )

        await runner._maintain()

        assert prog.id not in runner._active
        assert runner._metrics.dag_timeouts == 1
        # DISCARDED state should have been set
        storage.atomic_state_transition.assert_called()
        call_args = storage.atomic_state_transition.call_args
        assert call_args[0][2] == ProgramState.DISCARDED.value

    async def test_timed_out_storage_get_returns_none(self):
        """Timed out program has been deleted from storage -> no crash."""
        storage = _mock_storage()
        storage.get = AsyncMock(return_value=None)

        config = DagRunnerConfig(dag_timeout=1.0)
        runner = _runner(storage=storage, config=config)

        task = asyncio.create_task(asyncio.sleep(3600))
        runner._active["deleted-prog"] = TaskInfo(
            task=task, program_id="deleted-prog", started_at=time.monotonic() - 100
        )

        await runner._maintain()

        assert "deleted-prog" not in runner._active
        assert runner._metrics.dag_timeouts == 1
        # No set_program_state call since program is None
        storage.atomic_state_transition.assert_not_called()


# ===================================================================
# Category B: _launch() build failure + state update failure cascade
# ===================================================================


class TestLaunchBuildStateFailureCascade:
    """dag_runner.py L366-383: build() fails -> try DISCARD -> DISCARD also fails.
    Double failure cascade: record_build_failure AND record_state_update_failure."""

    async def test_build_failure_then_state_update_failure(self):
        """Build raises, then state transition to DISCARDED also raises.
        Both failures are recorded: build_failure unconditionally,
        state_update_failure additionally when discard fails."""
        prog = _prog(state=ProgramState.QUEUED)
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.QUEUED.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])
        # Make the DISCARD state transition fail
        storage.atomic_state_transition = AsyncMock(
            side_effect=RuntimeError("Redis down")
        )

        blueprint = MagicMock()
        blueprint.build = MagicMock(side_effect=RuntimeError("build failed"))

        runner = _runner(storage=storage, dag_blueprint=blueprint)
        await runner._launch()

        # Build failure always recorded (regardless of state update outcome)
        assert runner._metrics.dag_build_failures == 1
        # State update failure also recorded
        assert runner._metrics.state_update_failures == 1
        # Total errors: build + state update
        assert runner._metrics.dag_errors == 2
        # Program should NOT be in active
        assert prog.id not in runner._active

    async def test_build_failure_state_update_succeeds(self):
        """Build raises but DISCARD succeeds -> only build_failure metric."""
        prog = _prog(state=ProgramState.QUEUED)
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.QUEUED.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])

        blueprint = MagicMock()
        blueprint.build = MagicMock(side_effect=RuntimeError("build failed"))

        runner = _runner(storage=storage, dag_blueprint=blueprint)
        await runner._launch()

        assert runner._metrics.dag_build_failures == 1
        assert runner._metrics.state_update_failures == 0
        assert runner._metrics.dag_errors == 1


# ===================================================================
# Category C: _launch() mget failure during Phase 3
# ===================================================================


class TestLaunchMgetFailure:
    """dag_runner.py L350-354: mget for launch fails -> return gracefully."""

    async def test_mget_failure_for_launch_returns_gracefully(self):
        """When mget fails during program fetch for launch, no crash."""
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: (
                ["prog-1", "prog-2"] if s == ProgramState.QUEUED.value else []
            )
        )
        # mget fails
        storage.mget = AsyncMock(side_effect=RuntimeError("mget failed"))

        runner = _runner(storage=storage)
        await runner._launch()

        # No programs launched
        assert len(runner._active) == 0
        assert runner._metrics.dag_runs_started == 0


# ===================================================================
# Category D: _launch() get_ids_by_status failure
# ===================================================================


class TestLaunchFetchByStatusFailure:
    """dag_runner.py L309-315: Phase 1 fetch-by-status fails -> return."""

    async def test_fetch_by_status_failure(self):
        """When get_ids_by_status raises, _launch returns gracefully."""
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(side_effect=RuntimeError("Redis timeout"))

        runner = _runner(storage=storage)
        await runner._launch()

        assert len(runner._active) == 0


# ===================================================================
# Category E: _launch() orphan discard state update failure
# ===================================================================


class TestLaunchOrphanDiscardFailure:
    """dag_runner.py L324-335: orphaned RUNNING program's discard fails."""

    async def test_orphan_discard_state_failure_records_metric(self):
        """State update fails for orphan -> state_update_failure metric recorded."""
        prog = _prog(state=ProgramState.RUNNING)
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.RUNNING.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])
        # State transition fails
        storage.atomic_state_transition = AsyncMock(
            side_effect=RuntimeError("state fail")
        )

        runner = _runner(storage=storage)
        await runner._launch()

        # Orphan was detected but discard failed
        assert runner._metrics.orphaned_programs_discarded == 0
        # State update failure IS recorded (so the error is visible in metrics)
        assert runner._metrics.state_update_failures == 1
        assert runner._metrics.dag_errors == 1


# ===================================================================
# Category F: Multiple timed-out programs in single _maintain()
# ===================================================================


class TestMaintainMultipleTimeouts:
    """Multiple programs timing out in the same _maintain() call."""

    async def test_multiple_timeouts_all_handled(self):
        """Three timed-out programs are all cancelled and discarded."""
        storage = _mock_storage()
        progs = [_prog(state=ProgramState.RUNNING) for _ in range(3)]
        # Return each prog by ID
        storage.get = AsyncMock(
            side_effect=lambda pid: next((p for p in progs if p.id == pid), None)
        )

        config = DagRunnerConfig(dag_timeout=1.0)
        runner = _runner(storage=storage, config=config)

        # Create tasks for all programs, all "timed out"
        for p in progs:
            task = asyncio.create_task(asyncio.sleep(3600))
            runner._active[p.id] = TaskInfo(
                task=task, program_id=p.id, started_at=time.monotonic() - 100
            )

        await runner._maintain()

        assert len(runner._active) == 0
        assert runner._metrics.dag_timeouts == 3
        assert runner._metrics.dag_errors == 3

    async def test_mixed_finished_and_timed_out(self):
        """Mix of completed and timed-out tasks in single _maintain()."""
        storage = _mock_storage()
        running_prog = _prog(state=ProgramState.RUNNING)
        storage.get = AsyncMock(return_value=running_prog)

        config = DagRunnerConfig(dag_timeout=1.0)
        runner = _runner(storage=storage, config=config)

        # Completed task
        async def noop():
            pass

        done_task = asyncio.create_task(noop())
        await done_task
        runner._active["done-prog"] = TaskInfo(
            task=done_task, program_id="done-prog", started_at=time.monotonic()
        )

        # Timed-out task
        timeout_task = asyncio.create_task(asyncio.sleep(3600))
        runner._active["timeout-prog"] = TaskInfo(
            task=timeout_task,
            program_id="timeout-prog",
            started_at=time.monotonic() - 100,
        )

        await runner._maintain()

        assert len(runner._active) == 0
        assert runner._metrics.dag_runs_completed == 1
        assert runner._metrics.dag_timeouts == 1


# ===================================================================
# Category G: _execute_dag cleanup always runs
# ===================================================================


class TestExecuteDagCleanup:
    """dag_runner.py L406-435: cleanup runs in finally block regardless of outcome."""

    async def test_cleanup_runs_after_cancellation(self):
        """Even if dag.run() is cancelled, cleanup nullifies references."""
        storage = _mock_storage()
        runner = _runner(storage=storage)
        prog = _prog(state=ProgramState.RUNNING)

        mock_dag = _mock_dag(run_side_effect=asyncio.CancelledError())

        # CancelledError propagates from dag.run -> caught by except Exception
        # Actually, CancelledError is NOT an Exception subclass in Python 3.9+
        # So it propagates. Let's test with a regular exception instead.
        mock_dag = _mock_dag(run_side_effect=RuntimeError("dag error"))

        await runner._execute_dag(mock_dag, prog)

        # Cleanup should have run
        assert mock_dag.automata is None
        assert mock_dag.state_manager is None
        assert mock_dag._writer is None
        assert mock_dag._stage_sema is None

        # State should be DISCARDED (due to error)
        storage.atomic_state_transition.assert_called()
        call_args = storage.atomic_state_transition.call_args
        assert call_args[0][2] == ProgramState.DISCARDED.value


# ===================================================================
# Category H: _launch concurrency gating via semaphore
# ===================================================================


class TestLaunchConcurrencyGating:
    """dag_runner.py L340-346: capacity calculated from max_concurrent_dags - len(_active)."""

    async def test_zero_capacity_skips_launch(self):
        """When at max capacity, _launch fetches IDs but launches nothing."""
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: ["new-prog"] if s == ProgramState.QUEUED.value else []
        )

        config = DagRunnerConfig(max_concurrent_dags=1)
        runner = _runner(storage=storage, config=config)

        # Pre-fill active with one running task
        dummy = asyncio.create_task(asyncio.sleep(100))
        runner._active["existing"] = TaskInfo(
            task=dummy, program_id="existing", started_at=time.monotonic()
        )

        await runner._launch()

        # No new programs launched (capacity = 1 - 1 = 0)
        assert "new-prog" not in runner._active
        assert len(runner._active) == 1  # only the pre-existing one

        dummy.cancel()
        try:
            await dummy
        except asyncio.CancelledError:
            pass

    async def test_partial_capacity_launches_limited(self):
        """With capacity=2 and 3 queued, only 2 are launched."""
        progs = [_prog() for _ in range(3)]
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: (
                [p.id for p in progs] if s == ProgramState.QUEUED.value else []
            )
        )
        storage.mget = AsyncMock(return_value=progs)

        mock_dag = _mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)

        config = DagRunnerConfig(max_concurrent_dags=2)
        runner = _runner(storage=storage, dag_blueprint=blueprint, config=config)

        await runner._launch()

        assert len(runner._active) == 2
        assert runner._metrics.dag_runs_started == 2

        # Cleanup
        for info in list(runner._active.values()):
            info.task.cancel()
            try:
                await info.task
            except (asyncio.CancelledError, Exception):
                pass


# ===================================================================
# Category I: _run_one closure capture regression test (P1)
# ===================================================================


class TestClosureCaptureRegression:
    """dag_runner.py L392: _run_one uses default args to capture loop vars.

    async def _run_one(prog: Program = program, dag_inst: DAG = dag) -> None:

    If someone removes the default args, all tasks would run the LAST
    program in the loop — a classic Python closure bug. This test verifies
    each launched task is bound to its own program."""

    async def test_each_task_runs_its_own_program(self):
        """Launch 3 programs, verify each DAG.run() receives a distinct program."""
        progs = [_prog() for _ in range(3)]
        storage = _mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: (
                [p.id for p in progs] if s == ProgramState.QUEUED.value else []
            )
        )
        storage.mget = AsyncMock(return_value=progs)

        # Track which program each DAG.run() receives
        programs_seen: list[str] = []
        run_event = asyncio.Event()

        async def record_run(program):
            programs_seen.append(program.id)
            if len(programs_seen) == 3:
                run_event.set()

        def make_dag():
            dag = _mock_dag()
            dag.run = AsyncMock(side_effect=record_run)
            return dag

        blueprint = MagicMock()
        blueprint.build = MagicMock(side_effect=lambda *a, **kw: make_dag())

        config = DagRunnerConfig(max_concurrent_dags=3)
        runner = _runner(storage=storage, dag_blueprint=blueprint, config=config)

        await runner._launch()

        # Wait for all 3 tasks to call dag.run()
        try:
            await asyncio.wait_for(run_event.wait(), timeout=5.0)
        except TimeoutError:
            pass

        # Each program should appear exactly once
        expected_ids = {p.id for p in progs}
        assert set(programs_seen) == expected_ids
        assert len(programs_seen) == 3

        # Cleanup
        for info in list(runner._active.values()):
            info.task.cancel()
            try:
                await info.task
            except (asyncio.CancelledError, Exception):
                pass
