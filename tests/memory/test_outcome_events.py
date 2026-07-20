from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_OUTCOME_METADATA_KEY,
    MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY,
)
from gigaevo.evolution.mutation.terminal_failure import (
    MutationTerminalFailureStage,
    set_mutation_terminal_failure,
)
from gigaevo.memory.cards import (
    AssignmentRecord,
    DecisionContext,
    MutationAssignmentRecord,
)
from gigaevo.memory.events import MemoryOutcome, MemoryOutcomeUpdate
from gigaevo.memory.ope.reconcile import reconcile_rows
from gigaevo.memory.outcomes import record_program_memory_outcome
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _metrics_context(*, higher_is_better: bool = True) -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="test fitness",
                is_primary=True,
                higher_is_better=higher_is_better,
            )
        }
    )


def _child(*, child_fitness: float, base_fitness: float) -> Program:
    child = Program(code="def child(): return 1")
    child.metrics = {"is_valid": 1.0, "fitness": child_fitness}
    child.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, "memsel-terminal")
    child.set_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY, "base-program")
    child.set_metadata(
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
        {"is_valid": 1.0, "fitness": base_fitness},
    )
    child.set_metadata(
        MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
        MutationAssignmentRecord(
            mutation_id=child.id,
            used_ids=(),
        ).model_dump(mode="json"),
    )
    return child


def _crossover_child(
    *,
    probe_arms: tuple[str, ...],
    child_fitness: float = 0.8,
    base_fitness: float = 0.5,
) -> Program:
    child = Program(code="def child(): return 1")
    child.metrics = {"is_valid": 1.0, "fitness": child_fitness}
    parent_assignments: dict[str, dict] = {}
    for index, arm in enumerate(probe_arms):
        parent_id = f"parent-{index}"
        is_probe = arm != "none"
        assignment = AssignmentRecord(
            decision_id=f"decision-{index}",
            policy_version="TestPolicy:v1",
            task_key="hover",
            assigned_ids=("offered",) if arm == "treated" else (),
            arm="injected" if arm == "treated" else "none",
            probe_arm=arm,
            randomized=is_probe,
            propensity_kind="probe_bernoulli" if is_probe else "observational",
            propensities={"offered": 0.5} if is_probe else {},
            context=DecisionContext(
                parent_id=parent_id,
                parent_metrics={"is_valid": 1.0, "fitness": base_fitness},
            ),
        )
        parent_assignments[parent_id] = assignment.model_dump(mode="json")
    child.set_metadata(
        MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY, parent_assignments
    )
    child.set_metadata(
        MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
        MutationAssignmentRecord(
            mutation_id=child.id,
            parent_ids=tuple(parent_assignments),
            used_ids=(),
        ).model_dump(mode="json"),
    )
    return child


@pytest.mark.asyncio
async def test_parent_read_metadata_is_not_a_child_terminal() -> None:
    parent = Program(code="def parent(): return 1")
    parent.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, "parent-decision")
    parent.set_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY, parent.id)
    parent.set_metadata(
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
        {"is_valid": 1.0, "fitness": 0.5},
    )
    sink = AsyncMock()

    result = await record_program_memory_outcome(
        parent,
        storage=AsyncMock(),
        metrics_context=_metrics_context(),
        outcome_sink=sink,
    )

    assert result == "not_applicable"
    sink.record_memory_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_multi_probe_crossover_marks_all_terminals_ope_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A child born from >1 randomized-probe decision cannot be attributed to any
    # single probe arm: its one outcome would enter multiple estimator arms. Every
    # terminal it emits must be flagged ope_eligible=False for the DR estimator.
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _crossover_child(probe_arms=("treated", "control"))

    result = await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=_metrics_context()
    )

    assert result == "emitted"
    assert len(emitted) == 2
    assert all(isinstance(event, MemoryOutcome) for event in emitted)
    assert all(event.ope_eligible is False for event in emitted)


@pytest.mark.asyncio
async def test_single_probe_crossover_keeps_terminals_ope_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One probe arm plus one observational (non-probe) parent is a single probe
    # decision — attributable, so its terminals stay eligible.
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _crossover_child(probe_arms=("treated", "none"))

    await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=_metrics_context()
    )

    assert emitted
    assert all(event.ope_eligible is True for event in emitted)


@pytest.mark.asyncio
async def test_terminal_outcome_emits_once_and_reevaluation_is_an_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    storage = AsyncMock()
    child = _child(child_fitness=0.8, base_fitness=0.5)

    first = await record_program_memory_outcome(
        child, storage=storage, metrics_context=_metrics_context()
    )
    duplicate = await record_program_memory_outcome(
        child, storage=storage, metrics_context=_metrics_context()
    )
    child.metrics["fitness"] = 0.9
    update = await record_program_memory_outcome(
        child, storage=storage, metrics_context=_metrics_context()
    )

    assert (first, duplicate, update) == ("emitted", "duplicate", "updated")
    assert isinstance(emitted[0], MemoryOutcome)
    assert emitted[0].decision_id == "memsel-terminal"
    assert emitted[0].fitness_delta == pytest.approx(0.3)
    assert emitted[0].child_id == child.id
    assert emitted[0].base_id == "base-program"
    assert emitted[0].status == "outcome"
    assert emitted[0].invalid is False
    assert isinstance(emitted[1], MemoryOutcomeUpdate)
    assert emitted[1].previous_fitness_delta == pytest.approx(0.3)
    assert emitted[1].fitness_delta == pytest.approx(0.4)
    assert storage.update.await_count == 2

    rows = [
        {"event": type(event).event, **event.model_dump(mode="json")}
        for event in emitted
    ]
    rows.insert(
        0,
        {
            "event": "MEMORY_ASSIGNMENT",
            "decision_id": "memsel-terminal",
            "assignment": {"decision_id": "memsel-terminal"},
        },
    )
    reconciliation = reconcile_rows(rows)
    assert reconciliation.reconciled_ids == ("memsel-terminal",)
    assert reconciliation.dupes == {}
    (terminal,) = reconciliation.terminals["memsel-terminal"]
    assert terminal.outcome == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_cited_outcome_marker_remains_duplicate_after_json_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _child(child_fitness=0.8, base_fitness=0.5)
    child.set_metadata(
        MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
        MutationAssignmentRecord(
            mutation_id=child.id,
            delivered_ids=("card",),
            used_ids=("card",),
        ).model_dump(mode="json"),
    )

    first = await record_program_memory_outcome(
        child,
        storage=AsyncMock(),
        metrics_context=_metrics_context(),
    )
    restarted = Program.from_dict(json.loads(json.dumps(child.to_dict())))
    duplicate = await record_program_memory_outcome(
        restarted,
        storage=AsyncMock(),
        metrics_context=_metrics_context(),
    )

    assert (first, duplicate) == ("emitted", "duplicate")
    assert len(emitted) == 1
    assert isinstance(emitted[0], MemoryOutcome)
    assert emitted[0].used_card_ids == ("card",)


@pytest.mark.asyncio
async def test_terminal_outcome_orients_minimize_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _child(child_fitness=0.2, base_fitness=0.5)

    await record_program_memory_outcome(
        child,
        storage=AsyncMock(),
        metrics_context=_metrics_context(higher_is_better=False),
    )

    (event,) = emitted
    assert isinstance(event, MemoryOutcome)
    assert event.fitness_delta == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_failed_terminal_claim_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _child(child_fitness=0.8, base_fitness=0.5)
    storage = AsyncMock()
    storage.update.side_effect = RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        await record_program_memory_outcome(
            child, storage=storage, metrics_context=_metrics_context()
        )

    assert child.get_metadata(MUTATION_MEMORY_OUTCOME_METADATA_KEY) is None
    assert emitted == []
    storage.update.side_effect = None
    result = await record_program_memory_outcome(
        child, storage=storage, metrics_context=_metrics_context()
    )
    assert result == "emitted"
    assert len(emitted) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "expected_status"),
    [
        (MutationTerminalFailureStage.DAG_TIMEOUT, "invalid"),
        (MutationTerminalFailureStage.DAG_EXECUTION, "invalid"),
        (MutationTerminalFailureStage.DAG_BUILD, "censored"),
        (MutationTerminalFailureStage.SCHEDULER_ORPHAN, "censored"),
    ],
)
async def test_discarded_child_uses_typed_failure_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: MutationTerminalFailureStage,
    expected_status: str,
) -> None:
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _child(child_fitness=0.0, base_fitness=0.5)
    child.state = ProgramState.DISCARDED
    set_mutation_terminal_failure(child, failure_stage)

    await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=_metrics_context()
    )

    (event,) = emitted
    assert isinstance(event, MemoryOutcome)
    assert event.status == expected_status
    assert event.invalid is (expected_status == "invalid")
    assert event.failure_stage == failure_stage.value
    assert event.censor_reason == (
        failure_stage.value if expected_status == "censored" else ""
    )


@pytest.mark.asyncio
async def test_unclassified_discard_is_censored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    child = _child(child_fitness=0.0, base_fitness=0.5)
    child.state = ProgramState.DISCARDED

    await record_program_memory_outcome(
        child, storage=AsyncMock(), metrics_context=_metrics_context()
    )

    (event,) = emitted
    assert event.status == "censored"
    assert event.invalid is False
    assert event.censor_reason == "child_discarded_unclassified"
