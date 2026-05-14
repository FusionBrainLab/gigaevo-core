"""Verifies the cascade contract the archive-potential gate relies on.

When ``ArchivePotentialGateStage.compute()`` returns
``ProgramStageResult.skipped(...)``, the framework's
``DAGAutomata.get_stages_to_skip`` must auto-skip ``on_success`` dependents
(InsightsStage in production) while leaving ``always_after`` dependents
running (LineageStage, MutationContextStage).

If this test fails, the gate's own unit tests will still pass but the gate
will produce no actual skip in a running pipeline — fix in the framework
contract, not in the gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

from gigaevo.programs.core_types import FINAL_STATES, ProgramStageResult, StageState
from gigaevo.programs.dag.automata import DAGAutomata, ExecutionOrderDependency
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.conftest import FastStage


def _skipped_result() -> ProgramStageResult:
    now = datetime.now(UTC)
    skipped = ProgramStageResult.skipped(
        stage="ArchivePotentialGateStage",
        message="dominated_in_all_islands",
        error_type="ArchiveGateSkip",
    )
    # Tests below rely on finished_at; skipped() already sets it but we
    # normalise here in case ProgramStageResult.skipped() ever changes.
    if skipped.finished_at is None and skipped.status in FINAL_STATES:
        skipped.finished_at = now
    return skipped


def _program_with_gate_skipped() -> Program:
    p = Program(
        code="def solve(): return 1",
        state=ProgramState.RUNNING,
        atomic_counter=1,
    )
    p.stage_results = {"ArchivePotentialGateStage": _skipped_result()}
    return p


def _build(nodes: dict, exec_deps: dict) -> DAGAutomata:
    return DAGAutomata.build(
        nodes=nodes,
        data_flow_edges=[],
        execution_order_deps=exec_deps,
    )


def test_on_success_dependent_cascades_to_skip_when_gate_returns_skipped():
    """The bread-and-butter case: InsightsStage (on_success on the gate)
    must end up in to_skip when the gate emits SKIPPED."""
    automata = _build(
        {
            "ArchivePotentialGateStage": FastStage(timeout=5.0),
            "InsightsStage": FastStage(timeout=5.0),
        },
        exec_deps={
            "InsightsStage": [
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage")
            ]
        },
    )
    to_skip = automata.get_stages_to_skip(
        _program_with_gate_skipped(),
        running=set(),
        launched_this_run=set(),
        finished_this_run={"ArchivePotentialGateStage"},
    )
    assert "InsightsStage" in to_skip


def test_always_after_dependent_does_not_cascade_when_gate_returns_skipped():
    """LineageStage / MutationContextStage are always_after on the gate
    (in production they depend on EnsureMetricsStage, not the gate — but
    if a future wiring change made them depend always_after on the gate,
    they must still run because SKIPPED satisfies the 'always' condition).
    """
    automata = _build(
        {
            "ArchivePotentialGateStage": FastStage(timeout=5.0),
            "LineageStage": FastStage(timeout=5.0),
        },
        exec_deps={
            "LineageStage": [
                ExecutionOrderDependency.always_after("ArchivePotentialGateStage")
            ]
        },
    )
    to_skip = automata.get_stages_to_skip(
        _program_with_gate_skipped(),
        running=set(),
        launched_this_run=set(),
        finished_this_run={"ArchivePotentialGateStage"},
    )
    assert "LineageStage" not in to_skip


def test_skipped_status_is_in_failure_set_used_by_check_dependency_gate():
    """Sanity: ProgramStageResult.skipped() must produce StageState.SKIPPED,
    and the on_success condition must reject SKIPPED. If this changes,
    the gate stops working as designed.
    """
    skipped = _skipped_result()
    assert skipped.status == StageState.SKIPPED

    dep = ExecutionOrderDependency.on_success("ArchivePotentialGateStage")
    assert dep.is_satisfied_historically(skipped) is False

    dep_always = ExecutionOrderDependency.always_after("ArchivePotentialGateStage")
    assert dep_always.is_satisfied_historically(skipped) is True
