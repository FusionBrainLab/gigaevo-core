"""Tests for gigaevo/runner/dag_runner.py"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import (
    DagRunner,
    DagRunnerConfig,
    DagRunnerMetrics,
    TaskInfo,
)


class TestDagRunnerConfig:
    def test_defaults(self):
        config = DagRunnerConfig()
        assert config.poll_interval == 0.5
        assert config.max_concurrent_dags == 8
        assert config.dag_timeout == 3600
        assert config.metrics_collection_interval == 1.0

    def test_poll_interval_too_small(self):
        with pytest.raises(ValueError, match="poll_interval"):
            DagRunnerConfig(poll_interval=0.001)

    def test_poll_interval_gt_zero(self):
        with pytest.raises(ValueError):
            DagRunnerConfig(poll_interval=0)

    def test_max_concurrent_dags_gt_zero(self):
        with pytest.raises(ValueError):
            DagRunnerConfig(max_concurrent_dags=0)

    def test_dag_timeout_gt_zero(self):
        with pytest.raises(ValueError):
            DagRunnerConfig(dag_timeout=0)

    def test_poll_interval_upper_bound(self):
        with pytest.raises(ValueError):
            DagRunnerConfig(poll_interval=61.0)


class TestDagRunnerMetrics:
    def test_initial_values(self):
        m = DagRunnerMetrics()
        assert m.loop_iterations == 0
        assert m.dag_runs_started == 0
        assert m.dag_runs_completed == 0
        assert m.dag_errors == 0
        assert m.dag_timeouts == 0
        assert m.orphaned_programs_discarded == 0
        assert m.dag_build_failures == 0
        assert m.state_update_failures == 0

    def test_increment_loop_iterations(self):
        m = DagRunnerMetrics()
        m.increment_loop_iterations()
        assert m.loop_iterations == 1

    def test_increment_dag_runs_started(self):
        m = DagRunnerMetrics()
        m.increment_dag_runs_started()
        assert m.dag_runs_started == 1

    def test_increment_dag_runs_completed(self):
        m = DagRunnerMetrics()
        m.increment_dag_runs_completed()
        assert m.dag_runs_completed == 1

    def test_increment_dag_errors(self):
        m = DagRunnerMetrics()
        m.increment_dag_errors()
        assert m.dag_errors == 1

    def test_success_rate_no_finished(self):
        m = DagRunnerMetrics()
        assert m.success_rate == 1.0

    def test_success_rate_all_success(self):
        m = DagRunnerMetrics()
        m.dag_runs_completed = 10
        assert m.success_rate == pytest.approx(1.0)

    def test_success_rate_with_errors(self):
        m = DagRunnerMetrics()
        m.dag_runs_completed = 8
        m.dag_errors = 2
        assert m.success_rate == pytest.approx(0.8)

    def test_record_timeout(self):
        m = DagRunnerMetrics()
        m.record_timeout()
        assert m.dag_timeouts == 1
        assert m.dag_errors == 1

    def test_record_orphaned(self):
        m = DagRunnerMetrics()
        m.record_orphaned()
        assert m.orphaned_programs_discarded == 1
        assert m.dag_errors == 1

    def test_record_build_failure(self):
        m = DagRunnerMetrics()
        m.record_build_failure()
        assert m.dag_build_failures == 1
        assert m.dag_errors == 1

    def test_record_state_update_failure(self):
        m = DagRunnerMetrics()
        m.record_state_update_failure()
        assert m.state_update_failures == 1
        assert m.dag_errors == 1

    def test_uptime_seconds(self):
        m = DagRunnerMetrics()
        # Should be at least 0 (just created)
        assert m.uptime_seconds >= 0


class TestDagRunnerLifecycle:
    def _make_runner(self):
        from gigaevo.runner.dag_runner import DagRunner

        storage = MagicMock()
        storage.close = AsyncMock()
        storage.wait_for_activity = AsyncMock()

        dag_blueprint = MagicMock()
        config = DagRunnerConfig()
        writer = MagicMock()
        writer.bind = MagicMock(return_value=writer)

        runner = DagRunner(storage, dag_blueprint, config, writer)
        return runner

    async def test_start_creates_task(self):
        runner = self._make_runner()
        runner.start()
        assert runner._task is not None
        assert not runner._stopping
        # Cleanup
        await runner.stop()

    async def test_start_idempotent(self):
        runner = self._make_runner()
        runner.start()
        task1 = runner._task
        runner.start()  # should warn, not create second task
        assert runner._task is task1
        await runner.stop()

    async def test_stop_cancels(self):
        runner = self._make_runner()
        runner.start()
        assert runner._task is not None
        await runner.stop()
        assert runner._task is None

    def test_active_count_initially_zero(self):
        runner = self._make_runner()
        assert runner.active_count() == 0


class TestTaskInfo:
    def test_named_tuple_fields(self):
        mock_task = MagicMock()
        info = TaskInfo(task=mock_task, program_id="abc-123", started_at=1000.0)
        assert info.task is mock_task
        assert info.program_id == "abc-123"
        assert info.started_at == 1000.0


# ---------------------------------------------------------------------------
# Helpers for behavioral tests
# ---------------------------------------------------------------------------


def _make_test_program(state=ProgramState.QUEUED):
    return Program(code="def solve(): return 42", state=state, atomic_counter=999)


def _make_mock_storage():
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


def _make_mock_writer():
    writer = MagicMock()
    writer.bind = MagicMock(return_value=writer)
    writer.scalar = MagicMock()
    writer.text = MagicMock()
    return writer


def _make_mock_dag(run_side_effect=None):
    dag = MagicMock()
    dag.run = AsyncMock(side_effect=run_side_effect)
    dag.automata = MagicMock()
    dag.automata.topology = MagicMock()
    dag.automata.topology.nodes = MagicMock()
    dag.state_manager = MagicMock()
    dag._writer = MagicMock()
    dag._stage_sema = MagicMock()
    return dag


def _make_runner(storage=None, dag_blueprint=None, config=None, writer=None):
    storage = storage or _make_mock_storage()
    dag_blueprint = dag_blueprint or MagicMock()
    config = config or DagRunnerConfig()
    writer = writer or _make_mock_writer()
    return DagRunner(storage, dag_blueprint, config, writer)


# ---------------------------------------------------------------------------
# TestDagRunnerLaunch
# ---------------------------------------------------------------------------


class TestDagRunnerLaunch:
    async def test_queued_program_transitions_to_running(self):
        """Add QUEUED program to storage, call _launch(), verify it becomes active."""
        prog = _make_test_program(state=ProgramState.QUEUED)
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.QUEUED.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])

        mock_dag = _make_mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)

        runner = _make_runner(storage=storage, dag_blueprint=blueprint)
        await runner._launch()

        assert prog.id in runner._active
        assert runner._metrics.dag_runs_started == 1
        # Cleanup
        for info in list(runner._active.values()):
            info.task.cancel()
            try:
                await info.task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_respects_max_concurrent_capacity(self):
        """Set max_concurrent_dags=1, add 3 QUEUED programs, verify only 1 is active."""
        progs = [_make_test_program() for _ in range(3)]
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [p.id for p in progs]
            if s == ProgramState.QUEUED.value
            else []
        )
        storage.mget = AsyncMock(return_value=progs)

        mock_dag = _make_mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)

        config = DagRunnerConfig(max_concurrent_dags=1)
        runner = _make_runner(storage=storage, dag_blueprint=blueprint, config=config)
        await runner._launch()

        assert len(runner._active) == 1
        # Cleanup
        for info in list(runner._active.values()):
            info.task.cancel()
            try:
                await info.task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_skips_already_active_programs(self):
        """Put a program in _active dict, call _launch(), verify it's not re-launched."""
        prog = _make_test_program(state=ProgramState.QUEUED)
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.QUEUED.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])

        mock_dag = _make_mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)

        runner = _make_runner(storage=storage, dag_blueprint=blueprint)
        # Pre-populate _active with this program
        dummy_task = asyncio.create_task(asyncio.sleep(100))
        runner._active[prog.id] = TaskInfo(
            task=dummy_task, program_id=prog.id, started_at=time.monotonic()
        )

        await runner._launch()
        # Should still have only the original task
        assert runner._active[prog.id].task is dummy_task
        assert blueprint.build.call_count == 0

        dummy_task.cancel()
        try:
            await dummy_task
        except asyncio.CancelledError:
            pass

    async def test_orphaned_running_discarded(self):
        """RUNNING program not in _active gets DISCARDED."""
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.RUNNING.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])

        runner = _make_runner(storage=storage)
        await runner._launch()

        assert runner._metrics.orphaned_programs_discarded == 1
        assert runner._metrics.dag_errors == 1
        # atomic_state_transition should have been called to discard with DISCARDED state
        storage.atomic_state_transition.assert_called()
        call_args = storage.atomic_state_transition.call_args
        assert call_args[0][2] == ProgramState.DISCARDED.value

    async def test_mark_running_failure_cancels_task_and_removes_from_active(self):
        """If set_program_state(RUNNING) raises after task creation, task is cancelled."""
        prog = _make_test_program(state=ProgramState.QUEUED)
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.QUEUED.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])
        # Make the RUNNING state transition fail
        storage.atomic_state_transition = AsyncMock(
            side_effect=RuntimeError("Redis timeout")
        )

        mock_dag = _make_mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)

        runner = _make_runner(storage=storage, dag_blueprint=blueprint)
        await runner._launch()

        # After failure, program should NOT be in _active
        assert prog.id not in runner._active

    async def test_mget_returns_partial_results(self):
        """When mget returns [prog, None], only the non-None program is launched."""
        prog = _make_test_program(state=ProgramState.QUEUED)
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id, "deleted-id"]
            if s == ProgramState.QUEUED.value
            else []
        )
        storage.mget = AsyncMock(return_value=[prog, None])

        mock_dag = _make_mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)

        runner = _make_runner(storage=storage, dag_blueprint=blueprint)
        await runner._launch()

        # Only the real program should be launched
        assert prog.id in runner._active
        assert "deleted-id" not in runner._active
        assert len(runner._active) == 1

        # Cleanup
        for info in list(runner._active.values()):
            info.task.cancel()
            try:
                await info.task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_build_failure_discards_program(self):
        """DAGBlueprint.build() raises -> program DISCARDED, metrics incremented."""
        prog = _make_test_program(state=ProgramState.QUEUED)
        storage = _make_mock_storage()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [prog.id] if s == ProgramState.QUEUED.value else []
        )
        storage.mget = AsyncMock(return_value=[prog])

        blueprint = MagicMock()
        blueprint.build = MagicMock(side_effect=RuntimeError("build failed"))

        runner = _make_runner(storage=storage, dag_blueprint=blueprint)
        await runner._launch()

        assert runner._metrics.dag_build_failures == 1
        assert runner._metrics.dag_errors == 1
        assert prog.id not in runner._active


# ---------------------------------------------------------------------------
# TestDagRunnerMaintain
# ---------------------------------------------------------------------------


class TestDagRunnerMaintain:
    async def test_finished_task_harvested(self):
        """Done asyncio.Task in _active gets removed, dag_runs_completed incremented."""
        runner = _make_runner()

        # Create a task that completes immediately
        async def noop():
            pass

        task = asyncio.create_task(noop())
        await task  # ensure it's done

        runner._active["prog-1"] = TaskInfo(
            task=task, program_id="prog-1", started_at=time.monotonic()
        )

        await runner._maintain()

        assert "prog-1" not in runner._active
        assert runner._metrics.dag_runs_completed == 1

    async def test_timed_out_task_cancelled(self):
        """Pending task with started_at in the past gets cancelled, dag_timeouts incremented."""
        storage = _make_mock_storage()
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage.get = AsyncMock(return_value=prog)

        config = DagRunnerConfig(dag_timeout=1.0)
        runner = _make_runner(storage=storage, config=config)

        # Create a task that will take a long time
        task = asyncio.create_task(asyncio.sleep(3600))

        # Set started_at far in the past so it's timed out
        runner._active["prog-1"] = TaskInfo(
            task=task, program_id="prog-1", started_at=time.monotonic() - 100
        )

        await runner._maintain()

        assert "prog-1" not in runner._active
        assert runner._metrics.dag_timeouts == 1
        assert runner._metrics.dag_errors == 1
        # Verify DISCARDED state was set on the program
        storage.atomic_state_transition.assert_called()
        call_args = storage.atomic_state_transition.call_args
        assert call_args[0][2] == ProgramState.DISCARDED.value

    async def test_unharvested_done_tasks_block_capacity(self):
        """Done tasks in _active block capacity until _maintain harvests them."""
        storage = _make_mock_storage()

        async def noop():
            pass

        # Create 2 done tasks
        task1 = asyncio.create_task(noop())
        task2 = asyncio.create_task(noop())
        await task1
        await task2

        config = DagRunnerConfig(max_concurrent_dags=2)
        runner = _make_runner(storage=storage, config=config)

        runner._active["p1"] = TaskInfo(
            task=task1, program_id="p1", started_at=time.monotonic()
        )
        runner._active["p2"] = TaskInfo(
            task=task2, program_id="p2", started_at=time.monotonic()
        )

        # Capacity check: len(_active) == 2, max == 2 -> capacity == 0
        # Even though active_count() == 0 (both tasks done)
        assert runner.active_count() == 0
        assert len(runner._active) == 2

        # New programs queued
        new_prog = _make_test_program()
        storage.get_ids_by_status = AsyncMock(
            side_effect=lambda s: [new_prog.id]
            if s == ProgramState.QUEUED.value
            else []
        )
        storage.mget = AsyncMock(return_value=[new_prog])

        mock_dag = _make_mock_dag()
        blueprint = MagicMock()
        blueprint.build = MagicMock(return_value=mock_dag)
        runner._dag_blueprint = blueprint

        # _launch won't launch because capacity == 0
        await runner._launch()
        assert new_prog.id not in runner._active

        # After _maintain harvests done tasks, capacity is restored
        await runner._maintain()
        assert len(runner._active) == 0
        assert runner._metrics.dag_runs_completed == 2

        await runner._launch()
        assert new_prog.id in runner._active

        # Cleanup
        for info in list(runner._active.values()):
            info.task.cancel()
            try:
                await info.task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_failed_task_increments_errors(self):
        """Task that raised an exception gets dag_errors incremented."""
        runner = _make_runner()

        async def failing():
            raise RuntimeError("boom")

        task = asyncio.create_task(failing())
        # Wait for it to fail
        try:
            await task
        except RuntimeError:
            pass

        runner._active["prog-1"] = TaskInfo(
            task=task, program_id="prog-1", started_at=time.monotonic()
        )

        await runner._maintain()

        assert "prog-1" not in runner._active
        assert runner._metrics.dag_errors == 1


# ---------------------------------------------------------------------------
# TestDagRunnerExecuteDag
# ---------------------------------------------------------------------------


class TestDagRunnerExecuteDag:
    async def test_success_sets_done(self):
        """Mock DAG.run() to succeed, verify state becomes DONE."""
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage = _make_mock_storage()
        runner = _make_runner(storage=storage)

        mock_dag = _make_mock_dag()  # run() succeeds (returns None)

        await runner._execute_dag(mock_dag, prog)

        mock_dag.run.assert_awaited_once_with(prog)
        # set_program_state should be called with DONE
        # It's called via _state_manager which wraps storage
        # Check that atomic_state_transition was called
        storage.atomic_state_transition.assert_called()
        call_args = storage.atomic_state_transition.call_args
        assert call_args[0][2] == ProgramState.DONE.value

    async def test_failure_sets_discarded(self):
        """Mock DAG.run() to raise, verify state becomes DISCARDED."""
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage = _make_mock_storage()
        runner = _make_runner(storage=storage)

        mock_dag = _make_mock_dag(run_side_effect=RuntimeError("dag failed"))

        await runner._execute_dag(mock_dag, prog)

        mock_dag.run.assert_awaited_once_with(prog)
        storage.atomic_state_transition.assert_called()
        call_args = storage.atomic_state_transition.call_args
        assert call_args[0][2] == ProgramState.DISCARDED.value

    async def test_execute_dag_cleanup_after_crash(self):
        """When dag.run() raises, the finally block still cleans up DAG internals."""
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage = _make_mock_storage()
        runner = _make_runner(storage=storage)

        mock_dag = _make_mock_dag(run_side_effect=RuntimeError("dag crashed"))

        await runner._execute_dag(mock_dag, prog)

        # Verify cleanup: DAG internals should be nullified
        assert mock_dag.automata is None
        assert mock_dag.state_manager is None
        assert mock_dag._writer is None
        assert mock_dag._stage_sema is None

    async def test_execute_dag_cleanup_nullifies_references(self):
        """After successful _execute_dag, DAG references are set to None."""
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage = _make_mock_storage()
        runner = _make_runner(storage=storage)

        mock_dag = _make_mock_dag()  # run() succeeds

        await runner._execute_dag(mock_dag, prog)

        assert mock_dag.automata is None
        assert mock_dag.state_manager is None
        assert mock_dag._writer is None
        assert mock_dag._stage_sema is None

    async def test_execute_dag_state_update_failure_logged(self):
        """When set_program_state fails, state_update_failures metric increments."""
        prog = _make_test_program(state=ProgramState.RUNNING)
        storage = _make_mock_storage()
        # Make the state transition fail after DAG run
        storage.atomic_state_transition = AsyncMock(
            side_effect=RuntimeError("Redis down")
        )
        runner = _make_runner(storage=storage)

        mock_dag = _make_mock_dag()  # run() succeeds

        await runner._execute_dag(mock_dag, prog)

        assert runner._metrics.state_update_failures == 1
        assert runner._metrics.dag_errors == 1
