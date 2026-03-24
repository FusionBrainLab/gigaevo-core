"""Tests for DAG internal mechanics and edge-case behaviour.

Covers the low-level contracts that integration tests exercise only
indirectly:

  - Task cancellation handling in _process_finished_task
  - DAGAutomata construction validation (non-Stage nodes rejected)
  - Stall watchdog fires a WARNING when no progress is made
  - Unresolved-stages defensive path logs a WARNING and terminates
  - _write_stage_status emits exactly 5 correct scalar metrics per state
  - build_named_inputs excludes COMPLETED producers whose output is None
  - compute_hash_from_inputs exception falls back to re-execution
  - _diagnose_stage with combined exec-order + data-flow WAIT gates
  - DAGValidator rejects malformed structure (unknown stages, cycles)
  - write_exclusive is NOT called at DAG start (deferred to first stage persist)
  - mark_stage_running updates memory only, never writes to Redis
  - newly_cached stages reset the progress timer (no spurious stall)
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import loguru
import pytest

from gigaevo.programs.core_types import (
    FINAL_STATES,
    ProgramStageResult,
    StageError,
    StageState,
    VoidInput,
)
from gigaevo.programs.dag.automata import (
    DAGAutomata,
    DAGValidator,
    DataFlowEdge,
    ExecutionOrderDependency,
)
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from tests.conftest import (
    ChainedStage,
    FailingStage,
    FastStage,
    MockOutput,
    NullWriter,
    OptionalInputStage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dag(
    nodes: dict,
    edges: list,
    state_manager,
    *,
    exec_deps: dict | None = None,
    **kwargs,
) -> DAG:
    return DAG(
        nodes=nodes,
        data_flow_edges=edges,
        execution_order_deps=exec_deps,
        state_manager=state_manager,
        writer=NullWriter(),
        **kwargs,
    )


def _make_program(
    stage_results: dict[str, ProgramStageResult] | None = None,
) -> Program:
    p = Program(
        code="def solve(): return 1",
        state=ProgramState.RUNNING,
        atomic_counter=999_999_999,
    )
    if stage_results:
        p.stage_results = stage_results
    return p


def _make_result(
    status: StageState,
    *,
    input_hash: str | None = None,
    output: Any = None,
) -> ProgramStageResult:
    now = datetime.now(UTC)
    return ProgramStageResult(
        status=status,
        started_at=now,
        finished_at=now if status in FINAL_STATES else None,
        input_hash=input_hash,
        output=output,
    )


# ---------------------------------------------------------------------------
# Additional stage mocks needed only in this module
# ---------------------------------------------------------------------------


class HangingStage(Stage):
    """Sleeps forever; used to test cancellation paths."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        await asyncio.sleep(3600)
        return MockOutput(value=0)  # pragma: no cover


class SlowLongStage(Stage):
    """Sleeps 1.5 s — long enough to outlast asyncio.wait's 1.0 s poll interval,
    which lets the stall watchdog fire when stall_grace_seconds is tiny."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    async def compute(self, program: Program) -> MockOutput:
        await asyncio.sleep(1.5)
        return MockOutput(value=1)


class GatedStage(Stage):
    """Stage that waits on an asyncio.Event before completing."""

    InputsModel = VoidInput
    OutputModel = MockOutput

    def __init__(self, *, timeout: float):
        super().__init__(timeout=timeout)
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def compute(self, program: Program) -> MockOutput:
        self.started.set()
        await self.gate.wait()
        return MockOutput(value=7)


# ===========================================================================
# Task cancellation handling
# ===========================================================================


class TestTaskCancellation:
    """_process_finished_task must correctly handle a task that raises
    CancelledError when result() is called.

    In Python 3.8+ asyncio.CancelledError is a BaseException, not Exception.
    The handler must use isinstance(outcome, BaseException) — not Exception —
    so that CancelledError is routed to the CANCELLED path and not silently
    cast to ProgramStageResult (which would crash on .status access).
    """

    async def test_cancelled_task_records_cancelled_state(
        self, state_manager, make_program
    ):
        """A task that raises CancelledError produces StageState.CANCELLED
        with StageError(type='Cancelled')."""
        dag = _make_dag({"a": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()

        prog.stage_results["a"] = ProgramStageResult(
            status=StageState.RUNNING,
            started_at=datetime.now(UTC),
        )
        await state_manager.storage.add(prog)

        async def _coro_that_raises_cancelled():
            raise asyncio.CancelledError()

        task = asyncio.create_task(_coro_that_raises_cancelled())
        await asyncio.sleep(0)

        running: set[str] = {"a"}
        finished: set[str] = set()

        await dag._process_finished_task(prog, "a", task, running, finished)

        result = prog.stage_results["a"]
        assert result.status == StageState.CANCELLED
        assert result.error is not None
        assert result.error.type == "Cancelled"
        assert "a" not in running
        assert "a" in finished

    async def test_cancelled_task_removes_stage_from_running_set(
        self, state_manager, make_program
    ):
        """After handling a CancelledError task, the stage is removed from
        the running set and added to finished."""
        dag = _make_dag({"stage": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        prog.stage_results["stage"] = ProgramStageResult(
            status=StageState.RUNNING,
            started_at=datetime.now(UTC),
        )
        await state_manager.storage.add(prog)

        async def _cancelled():
            raise asyncio.CancelledError()

        task = asyncio.create_task(_cancelled())
        await asyncio.sleep(0)

        running: set[str] = {"stage"}
        finished: set[str] = set()
        await dag._process_finished_task(prog, "stage", task, running, finished)

        assert "stage" not in running
        assert "stage" in finished

    async def test_dag_wide_cancellation_does_not_hang(
        self, state_manager, make_program
    ):
        """Cancelling the entire DAG task (which holds a hanging stage) must
        propagate CancelledError cleanly without hanging."""
        dag = _make_dag(
            {"hang": HangingStage(timeout=3600.0)},
            [],
            state_manager,
            dag_timeout=None,
        )
        prog = make_program()
        await state_manager.storage.add(prog)

        dag_task = asyncio.create_task(dag.run(prog))
        await asyncio.sleep(0.1)

        dag_task.cancel()
        with suppress(asyncio.CancelledError):
            await dag_task

        assert dag_task.done()


# ===========================================================================
# DAGAutomata construction validation
# ===========================================================================


class TestDAGAutomataConstruction:
    """DAGAutomata.build() must validate its inputs and raise ValueError for
    anything that is not a Stage instance."""

    def test_rejects_string_node(self):
        """A plain string value is rejected."""
        with pytest.raises(ValueError, match="Non-Stage objects"):
            DAGAutomata.build({"not_a_stage": "some_string"}, [], None)

    def test_rejects_integer_node(self):
        """An integer value is rejected."""
        with pytest.raises(ValueError, match="Non-Stage objects"):
            DAGAutomata.build({"bad": 42}, [], None)

    def test_rejects_dict_with_one_invalid_node(self):
        """A dict containing one valid Stage and one invalid value is rejected."""
        with pytest.raises(ValueError, match="Non-Stage objects"):
            DAGAutomata.build(
                {"good": FastStage(timeout=5.0), "bad": 42},
                [],
                None,
            )

    def test_accepts_valid_stage_instances(self):
        """A dict of proper Stage instances builds successfully."""
        automata = DAGAutomata.build({"a": FastStage(timeout=5.0)}, [], None)
        assert automata is not None
        assert "a" in automata.topology.nodes

    def test_rejects_stage_class_instead_of_instance(self):
        """Passing the Stage class (not an instance) is rejected."""
        with pytest.raises(ValueError, match="Non-Stage objects"):
            DAGAutomata.build({"cls": FastStage}, [], None)  # type: ignore[arg-type]

    def test_rejects_none_node_value(self):
        """None as a node value is rejected."""
        with pytest.raises(ValueError, match="Non-Stage objects"):
            DAGAutomata.build({"none_node": None}, [], None)  # type: ignore[arg-type]


# ===========================================================================
# Stall watchdog
# ===========================================================================


class TestStallWatchdog:
    """The stall watchdog in dag.py logs a WARNING when no progress has been
    made for longer than stall_grace_seconds.

    The watchdog can only fire after asyncio.wait() times out (1.0 s poll
    interval) with no task completing.  SlowLongStage sleeps 1.5 s, so after
    the first wait timeout the grace period (set to 0.001 s) is already
    exceeded and the warning fires.
    """

    async def test_stall_watchdog_fires_warning(self, state_manager, make_program):
        """A long-running stage with stall_grace_seconds=0.001 must trigger
        at least one STALLED warning."""
        dag = _make_dag({"slow": SlowLongStage(timeout=10.0)}, [], state_manager)
        dag.stall_grace_seconds = 0.001

        prog = make_program()
        warnings: list[str] = []

        def _capture(msg: Any) -> None:
            if "STALLED" in str(msg):
                warnings.append(str(msg))

        handler_id = loguru.logger.add(_capture, level="WARNING")
        try:
            await dag.run(prog)
        finally:
            loguru.logger.remove(handler_id)

        assert len(warnings) >= 1, (
            f"Expected at least one STALLED warning. Captured: {warnings}"
        )

    async def test_stall_watchdog_message_includes_grace_period(
        self, state_manager, make_program
    ):
        """The stall warning message includes the configured stall_grace_seconds value."""
        dag = _make_dag({"slow": SlowLongStage(timeout=10.0)}, [], state_manager)
        dag.stall_grace_seconds = 0.001

        prog = make_program()
        stall_messages: list[str] = []

        def _capture(msg: Any) -> None:
            if "STALLED" in str(msg):
                stall_messages.append(str(msg))

        handler_id = loguru.logger.add(_capture, level="WARNING")
        try:
            await dag.run(prog)
        finally:
            loguru.logger.remove(handler_id)

        assert any("0.001" in m for m in stall_messages), (
            f"Expected grace period value in stall message. Got: {stall_messages}"
        )


# ===========================================================================
# Unresolved-stages defensive path
# ===========================================================================


class TestUnresolvedStages:
    """dag.py logs a WARNING and terminates (without crashing or hanging) when
    the loop exits with stages that are neither done nor skipped.

    This is a defensive path for edge cases where the automata cannot resolve
    all stages.  We trigger it by patching get_stages_to_skip at the class
    level (Pydantic instances do not allow direct attribute assignment) so
    that the downstream of a failed stage is never skipped.
    """

    async def test_unresolved_stages_warning_is_logged(
        self, state_manager, make_program
    ):
        """When stages remain unresolved at termination, the DAG logs a warning
        mentioning the unresolved stages."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge(source_stage="a", destination_stage="b", input_name="data")],
            state_manager,
        )

        prog = make_program()
        warnings_logged: list[str] = []

        def _capture(msg: Any) -> None:
            if "unresolved" in str(msg).lower():
                warnings_logged.append(str(msg))

        handler_id = loguru.logger.add(_capture, level="WARNING")
        try:
            with patch.object(DAGAutomata, "get_stages_to_skip", return_value=set()):
                await dag.run(prog)
        finally:
            loguru.logger.remove(handler_id)

        assert any("unresolved" in w.lower() for w in warnings_logged), (
            f"Expected unresolved-stages warning. Got: {warnings_logged}"
        )
        assert prog.stage_results["a"].status == StageState.FAILED

    async def test_unresolved_stages_dag_still_terminates(
        self, state_manager, make_program
    ):
        """Even with unresolved stages the DAG must terminate rather than hang."""
        dag = _make_dag(
            {
                "a": FailingStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
            },
            [DataFlowEdge(source_stage="a", destination_stage="b", input_name="data")],
            state_manager,
        )
        prog = make_program()

        with patch.object(DAGAutomata, "get_stages_to_skip", return_value=set()):
            await asyncio.wait_for(dag.run(prog), timeout=10.0)


# ===========================================================================
# Stage status metrics
# ===========================================================================


class TestStageStatusMetrics:
    """_write_stage_status must emit exactly 5 scalar metrics for every final
    stage state: stage_success, stage_failure, stage_skipped, stage_cancelled,
    and stage_duration."""

    def _make_recording_writer(self) -> tuple[Any, list[tuple[str, float, dict]]]:
        from gigaevo.utils.trackers.base import LogWriter

        calls: list[tuple[str, float, dict]] = []

        class RecordingWriter(LogWriter):
            def bind(self, path: list[str] | None = None, **kw: Any) -> RecordingWriter:
                return self

            def scalar(self, metric: str, value: float, **kwargs: Any) -> None:
                calls.append((metric, value, kwargs))

            def hist(self, metric: str, values: list[float], **kwargs: Any) -> None:
                pass

            def text(self, tag: str, text: str, **kwargs: Any) -> None:
                pass

            def close(self) -> None:
                pass

        return RecordingWriter(), calls

    @pytest.mark.parametrize(
        "status,expected_flags",
        [
            (
                StageState.COMPLETED,
                {
                    "stage_success": 1,
                    "stage_failure": 0,
                    "stage_skipped": 0,
                    "stage_cancelled": 0,
                },
            ),
            (
                StageState.FAILED,
                {
                    "stage_success": 0,
                    "stage_failure": 1,
                    "stage_skipped": 0,
                    "stage_cancelled": 0,
                },
            ),
            (
                StageState.SKIPPED,
                {
                    "stage_success": 0,
                    "stage_failure": 0,
                    "stage_skipped": 1,
                    "stage_cancelled": 0,
                },
            ),
            (
                StageState.CANCELLED,
                {
                    "stage_success": 0,
                    "stage_failure": 0,
                    "stage_skipped": 0,
                    "stage_cancelled": 1,
                },
            ),
        ],
    )
    async def test_correct_flag_values_per_state(
        self,
        status: StageState,
        expected_flags: dict[str, int],
        state_manager,
    ):
        """Each of the 4 binary flag metrics has the correct value for the given
        final state; stage_duration must be a non-negative float."""
        writer, calls = self._make_recording_writer()

        dag = DAG(
            nodes={"a": FastStage(timeout=5.0)},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=writer,
        )

        now = datetime.now(UTC)
        result = ProgramStageResult(status=status, started_at=now, finished_at=now)
        await dag._write_stage_status("a", result)

        metric_values: dict[str, float] = {name: val for name, val, _ in calls}

        for metric, expected_value in expected_flags.items():
            assert metric in metric_values, (
                f"Metric '{metric}' missing. Got: {list(metric_values.keys())}"
            )
            assert metric_values[metric] == expected_value, (
                f"Expected {metric}={expected_value} for {status.name}, "
                f"got {metric_values[metric]}"
            )

        assert "stage_duration" in metric_values
        assert metric_values["stage_duration"] >= 0.0

    async def test_exactly_five_scalars_emitted(self, state_manager):
        """_write_stage_status emits exactly 5 scalar calls — no more, no less."""
        writer, calls = self._make_recording_writer()

        dag = DAG(
            nodes={"a": FastStage(timeout=5.0)},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=writer,
        )

        now = datetime.now(UTC)
        result = ProgramStageResult(
            status=StageState.COMPLETED, started_at=now, finished_at=now
        )
        await dag._write_stage_status("a", result)

        assert len(calls) == 5, (
            f"Expected exactly 5 scalar calls, got {len(calls)}: "
            f"{[name for name, _, _ in calls]}"
        )


# ===========================================================================
# build_named_inputs input filtering
# ===========================================================================


class TestBuildNamedInputs:
    """build_named_inputs in automata.py only passes output from COMPLETED
    producers whose output is not None."""

    def test_excludes_completed_stage_with_none_output(
        self, state_manager, make_program
    ):
        """A COMPLETED producer with output=None must not appear in named inputs."""
        dag = _make_dag(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            [DataFlowEdge(source_stage="a", destination_stage="b", input_name="data")],
            state_manager,
        )
        prog = make_program()
        prog.stage_results["a"] = ProgramStageResult(
            status=StageState.COMPLETED,
            output=None,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

        named = dag.automata.build_named_inputs(prog, "b")
        assert "data" not in named

    def test_includes_completed_stage_with_real_output(
        self, state_manager, make_program
    ):
        """A COMPLETED producer with a real output is included in named inputs."""
        dag = _make_dag(
            {"a": FastStage(timeout=5.0), "b": ChainedStage(timeout=5.0)},
            [DataFlowEdge(source_stage="a", destination_stage="b", input_name="data")],
            state_manager,
        )
        prog = make_program()
        output = MockOutput(value=42)
        prog.stage_results["a"] = ProgramStageResult(
            status=StageState.COMPLETED,
            output=output,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

        named = dag.automata.build_named_inputs(prog, "b")
        assert "data" in named
        assert named["data"] is output

    def test_excludes_failed_producer(self, state_manager, make_program):
        """A FAILED producer is excluded — status must be COMPLETED."""
        dag = _make_dag(
            {"a": FailingStage(timeout=5.0), "opt": OptionalInputStage(timeout=5.0)},
            [
                DataFlowEdge(
                    source_stage="a", destination_stage="opt", input_name="data"
                )
            ],
            state_manager,
        )
        prog = make_program()
        prog.stage_results["a"] = ProgramStageResult(
            status=StageState.FAILED,
            error=StageError(type="RuntimeError", message="boom"),
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

        named = dag.automata.build_named_inputs(prog, "opt")
        assert "data" not in named

    def test_no_edges_returns_empty_dict(self, state_manager, make_program):
        """A stage with no incoming edges has an empty named inputs dict."""
        dag = _make_dag({"a": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        assert dag.automata.build_named_inputs(prog, "a") == {}


# ===========================================================================
# Cache hash computation fallback
# ===========================================================================


class TestCacheHashFallback:
    """When compute_hash_from_inputs raises or returns None, the stage must
    be scheduled for re-execution rather than treated as cached."""

    def test_hash_exception_forces_rerun(self, state_manager, make_program):
        """An exception in compute_hash_from_inputs puts the stage in ready,
        not in newly_cached."""
        automata = DAGAutomata.build({"a": FastStage(timeout=5.0)}, [], None)
        prog = _make_program(
            stage_results={"a": _make_result(StageState.COMPLETED, input_hash="stored")}
        )

        with patch.object(
            FastStage,
            "compute_hash_from_inputs",
            side_effect=RuntimeError("hash exploded"),
        ):
            ready, cached = automata.get_ready_stages(
                prog, running=set(), launched_this_run=set(), finished_this_run=set()
            )

        assert "a" in ready
        assert "a" not in cached

    def test_hash_returns_none_forces_rerun(self, state_manager, make_program):
        """When compute_hash_from_inputs returns None, the InputHashCache sees
        a mismatch against any stored hash and triggers a rerun."""
        automata = DAGAutomata.build({"a": FastStage(timeout=5.0)}, [], None)
        prog = _make_program(
            stage_results={"a": _make_result(StageState.COMPLETED, input_hash="stored")}
        )

        with patch.object(FastStage, "compute_hash_from_inputs", return_value=None):
            ready, cached = automata.get_ready_stages(
                prog, running=set(), launched_this_run=set(), finished_this_run=set()
            )

        assert "a" in ready
        assert "a" not in cached


# ===========================================================================
# _diagnose_stage compound gate logic
# ===========================================================================


class TestDiagnoseStageGates:
    """_diagnose_stage combines exec-order and data-flow gate checks.
    Both must be satisfied before a stage is considered READY."""

    def test_combined_exec_and_dataflow_wait(self, state_manager, make_program):
        """Stage 'c' has an always_after exec dep on 'a' (not finished) AND a
        mandatory data dep on 'b' (not finished) — result must be WAIT."""
        automata = DAGAutomata.build(
            nodes={
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            data_flow_edges=[
                DataFlowEdge(source_stage="b", destination_stage="c", input_name="data")
            ],
            execution_order_deps={"c": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program()

        state, _ = automata._diagnose_stage(prog, "c", finished_this_run=set())
        assert state is DAGAutomata.GateState.WAIT

    def test_exec_wait_blocks_even_when_dataflow_ready(
        self, state_manager, make_program
    ):
        """Data dep on 'b' is satisfied this run; exec dep on 'a' is not.
        The exec WAIT must keep the stage blocked."""
        automata = DAGAutomata.build(
            nodes={
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            data_flow_edges=[
                DataFlowEdge(source_stage="b", destination_stage="c", input_name="data")
            ],
            execution_order_deps={"c": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(
            stage_results={
                "b": _make_result(StageState.COMPLETED, output=MockOutput(value=1))
            }
        )

        state, _ = automata._diagnose_stage(prog, "c", finished_this_run={"b"})
        assert state is DAGAutomata.GateState.WAIT

    def test_both_gates_satisfied_returns_ready(self, state_manager, make_program):
        """When both exec dep and data-flow dep are satisfied this run, the
        stage is READY."""
        automata = DAGAutomata.build(
            nodes={
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            data_flow_edges=[
                DataFlowEdge(source_stage="b", destination_stage="c", input_name="data")
            ],
            execution_order_deps={"c": [ExecutionOrderDependency.always_after("a")]},
        )
        prog = _make_program(
            stage_results={
                "a": _make_result(StageState.COMPLETED),
                "b": _make_result(StageState.COMPLETED, output=MockOutput(value=1)),
            }
        )

        state, _ = automata._diagnose_stage(prog, "c", finished_this_run={"a", "b"})
        assert state is DAGAutomata.GateState.READY


# ===========================================================================
# DAGValidator structure validation
# ===========================================================================


class TestDAGValidator:
    """DAGValidator.validate_structure() must return errors for malformed
    topologies and an empty list for valid ones."""

    def test_rejects_unknown_edge_source(self):
        """An edge referencing a non-existent source stage produces an error."""
        errors = DAGValidator.validate_structure(
            {"a": FastStage},  # type: ignore[dict-item]
            [
                DataFlowEdge(
                    source_stage="nonexistent", destination_stage="a", input_name="data"
                )
            ],
        )
        assert any("nonexistent" in e for e in errors)

    def test_rejects_unknown_edge_destination(self):
        """An edge referencing a non-existent destination stage produces an error."""
        errors = DAGValidator.validate_structure(
            {"a": FastStage},  # type: ignore[dict-item]
            [
                DataFlowEdge(
                    source_stage="a", destination_stage="ghost", input_name="data"
                )
            ],
        )
        assert any("ghost" in e for e in errors)

    def test_rejects_unknown_exec_dep_stage(self):
        """An exec dep referencing a non-existent dep stage produces an error."""
        errors = DAGValidator.validate_structure(
            {"a": FastStage},  # type: ignore[dict-item]
            [],
            {"a": [ExecutionOrderDependency(stage_name="ghost", condition="always")]},
        )
        assert any("ghost" in e for e in errors)

    def test_rejects_unknown_exec_dep_target(self):
        """Exec deps with an unknown target stage name produce an error."""
        errors = DAGValidator.validate_structure(
            {"a": FastStage},  # type: ignore[dict-item]
            [],
            {
                "unknown_target": [
                    ExecutionOrderDependency(stage_name="a", condition="always")
                ]
            },
        )
        assert any("unknown_target" in e for e in errors)

    def test_accepts_valid_two_node_dag(self):
        """A fully valid 2-node DAG produces no errors."""
        errors = DAGValidator.validate_structure({"a": FastStage, "b": FastStage}, [])  # type: ignore[dict-item]
        assert errors == []

    def test_accepts_valid_dag_with_exec_deps(self):
        """A valid DAG with exec-order deps produces no errors."""
        errors = DAGValidator.validate_structure(
            {"a": FastStage, "b": FastStage},  # type: ignore[dict-item]
            [],
            {"b": [ExecutionOrderDependency(stage_name="a", condition="success")]},
        )
        assert errors == []

    def test_detects_cycle_in_exec_deps(self):
        """A mutual always_after cycle must be detected."""
        errors = DAGValidator.validate_structure(
            {"a": FastStage, "b": FastStage},  # type: ignore[dict-item]
            [],
            {
                "b": [ExecutionOrderDependency(stage_name="a", condition="always")],
                "a": [ExecutionOrderDependency(stage_name="b", condition="always")],
            },
        )
        assert any("cycle" in e.lower() for e in errors)


# ===========================================================================
# Initial persistence contract
# ===========================================================================


class TestDAGInitialPersistence:
    """The DAG does NOT call write_exclusive at startup — only
    update_stage_result persists (via write_exclusive under the hood).
    This avoids one to_dict() + 2 Redis RT per program."""

    async def test_write_exclusive_not_called_at_startup_three_stage_dag(
        self, state_manager, make_program
    ):
        """A 3-stage DAG never calls write_exclusive directly (only via update_stage_result)."""
        dag = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": FastStage(timeout=5.0),
                "c": FastStage(timeout=5.0),
            },
            [],
            state_manager,
        )
        prog = make_program()

        call_count = 0
        original_write = state_manager.write_exclusive

        async def counting_write(p: Program) -> None:
            nonlocal call_count
            call_count += 1
            return await original_write(p)

        with patch.object(state_manager, "write_exclusive", side_effect=counting_write):
            await dag.run(prog)

        assert call_count == 0

    async def test_write_exclusive_not_called_at_startup_single_stage_dag(
        self, state_manager, make_program
    ):
        """A single-stage DAG also never calls write_exclusive directly."""
        dag = _make_dag({"only": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()

        call_count = 0
        original_write = state_manager.write_exclusive

        async def counting_write(p: Program) -> None:
            nonlocal call_count
            call_count += 1
            return await original_write(p)

        with patch.object(state_manager, "write_exclusive", side_effect=counting_write):
            await dag.run(prog)

        assert call_count == 0


# ===========================================================================
# Running state memory-only contract
# ===========================================================================


class TestRunningStateInMemoryOnly:
    """mark_stage_running() updates the program in memory but must NOT write
    to Redis.  Only update_stage_result() (called on stage completion) is
    allowed to persist stage results."""

    async def test_running_status_not_in_redis_while_stage_active(
        self, state_manager, fakeredis_storage, make_program
    ):
        """While a stage is executing, Redis must not contain RUNNING status
        for that stage.  After completion Redis shows COMPLETED."""
        gated = GatedStage(timeout=10.0)
        dag = _make_dag({"gated": gated}, [], state_manager)

        prog = make_program()
        await fakeredis_storage.add(prog)

        dag_task = asyncio.create_task(dag.run(prog))
        await asyncio.wait_for(gated.started.wait(), timeout=5.0)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        if "gated" in (fetched.stage_results or {}):
            assert fetched.stage_results["gated"].status != StageState.RUNNING

        gated.gate.set()
        await asyncio.wait_for(dag_task, timeout=5.0)

        fetched_after = await fakeredis_storage.get(prog.id)
        assert fetched_after is not None
        assert "gated" in fetched_after.stage_results
        assert fetched_after.stage_results["gated"].status == StageState.COMPLETED

    async def test_mark_stage_running_does_not_write_to_redis(
        self, state_manager, fakeredis_storage, make_program
    ):
        """mark_stage_running() must not call storage.write_exclusive."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        write_count = 0
        original_write = fakeredis_storage.write_exclusive

        async def counting_write(p: Program) -> None:
            nonlocal write_count
            write_count += 1
            return await original_write(p)

        now = datetime.now(UTC)
        with patch.object(
            fakeredis_storage, "write_exclusive", side_effect=counting_write
        ):
            await state_manager.mark_stage_running(prog, "some_stage", started_at=now)

        assert write_count == 0
        assert prog.stage_results["some_stage"].status == StageState.RUNNING


# ===========================================================================
# Cache progress accounting
# ===========================================================================


class TestCacheProgressAccounting:
    """When stages are newly_cached, last_progress_ts is reset in the DAG
    loop.  A fully-cached DAG must therefore complete without triggering the
    stall watchdog, even when stall_grace_seconds is very small."""

    async def test_all_cached_dag_completes_without_stall(
        self, state_manager, make_program
    ):
        """Run the DAG once to populate results.  A second run (all cached)
        with stall_grace_seconds=0.001 must not emit any STALLED warning."""
        dag1 = _make_dag({"a": FastStage(timeout=5.0)}, [], state_manager)
        prog = make_program()
        await dag1.run(prog)

        dag2 = _make_dag({"a": FastStage(timeout=5.0)}, [], state_manager)
        dag2.stall_grace_seconds = 0.001

        stall_warnings: list[str] = []

        def _capture(msg: Any) -> None:
            if "STALLED" in str(msg):
                stall_warnings.append(str(msg))

        handler_id = loguru.logger.add(_capture, level="WARNING")
        try:
            await dag2.run(prog)
        finally:
            loguru.logger.remove(handler_id)

        assert prog.stage_results["a"].status == StageState.COMPLETED
        assert len(stall_warnings) == 0, (
            f"Cached stages should reset progress timer. Got: {stall_warnings}"
        )

    async def test_cached_chain_completes_without_stall(
        self, state_manager, make_program
    ):
        """A 3-stage chain: first run executes all.  Second run (all cached)
        with tiny stall_grace_seconds must not trigger stall warnings."""
        edges = [
            DataFlowEdge(source_stage="a", destination_stage="b", input_name="data"),
            DataFlowEdge(source_stage="b", destination_stage="c", input_name="data"),
        ]

        dag1 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges,
            state_manager,
        )
        prog = make_program()
        await dag1.run(prog)

        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status == StageState.COMPLETED

        dag2 = _make_dag(
            {
                "a": FastStage(timeout=5.0),
                "b": ChainedStage(timeout=5.0),
                "c": ChainedStage(timeout=5.0),
            },
            edges,
            state_manager,
        )
        dag2.stall_grace_seconds = 0.001

        stall_warnings: list[str] = []

        def _capture(msg: Any) -> None:
            if "STALLED" in str(msg):
                stall_warnings.append(str(msg))

        handler_id = loguru.logger.add(_capture, level="WARNING")
        try:
            await dag2.run(prog)
        finally:
            loguru.logger.remove(handler_id)

        assert len(stall_warnings) == 0
        for name in ("a", "b", "c"):
            assert prog.stage_results[name].status == StageState.COMPLETED
