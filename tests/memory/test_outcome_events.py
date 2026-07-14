from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from experiments.hover.diff_memory.reconcile_assignments import reconcile_rows
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_OUTCOME_METADATA_KEY,
)
from gigaevo.memory.events import MemoryOutcome, MemoryOutcomeUpdate
from gigaevo.memory.outcomes import record_program_memory_outcome
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program


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
    return child


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
