from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.cards import (
    AssignmentRecord,
    Card,
    CardAssignmentSource,
    DecisionContext,
    MutationAssignmentRecord,
)
from gigaevo.memory.events import MemoryOutcome
from gigaevo.memory.outcomes import record_program_memory_outcome
from gigaevo.memory.write.stats import (
    CardStatsStamper,
    compute_contextual_gains,
    injection_outcomes_from_programs,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness",
                is_primary=True,
                higher_is_better=True,
            ),
            "x": MetricSpec(description="cell", higher_is_better=True),
        }
    )


def _parent(
    *,
    card_id: str,
    decision_id: str,
    fitness: float,
    cell: int,
    sparse_assignment_context: bool = False,
) -> Program:
    metrics = {"is_valid": 1.0, "fitness": fitness, "x": float(cell)}
    parent = Program(code=f"def parent_{cell}(): return {cell}")
    parent.metrics = metrics
    parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, [card_id])
    parent.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, decision_id)
    assignment = AssignmentRecord(
        decision_id=decision_id,
        policy_version="test-policy",
        task_key="test",
        assigned_ids=(card_id,),
        delivered_ids=(card_id,),
        arm="injected",
        context=(
            DecisionContext(task_key="test")
            if sparse_assignment_context
            else DecisionContext(
                task_key="test", parent_metrics=metrics, parent_id=parent.id
            )
        ),
        bd_cell=(cell,),
    )
    parent.set_metadata(
        MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
        assignment.model_dump(mode="json"),
    )
    return parent


class _ZeroBaseline:
    has_evidence = True

    def fit_no_card_baseline(self, outcomes, *, higher_is_better):
        del outcomes, higher_is_better
        return self

    def baseline_for(self, outcome):
        del outcome
        return 0.0

    def baseline_se_for(self, outcome):
        del outcome
        return None


@pytest.mark.asyncio
async def test_crossover_credits_each_card_in_its_parent_context_and_full_slate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _parent(
        card_id="base-card", decision_id="decision-base", fitness=0.5, cell=1
    )
    donor = _parent(
        card_id="donor-card",
        decision_id="decision-donor",
        fitness=0.6,
        cell=8,
        sparse_assignment_context=True,
    )
    mutator = AsyncMock()
    mutator.mutate_single.return_value = MutationSpec(
        code="def child(): return 1",
        parents=[base, donor],
        name="crossover",
        metadata={
            MutationSpec.META_OUTPUT: {
                "base_parent": 1,
                "card_ids_used": ["base-card", "donor-card"],
            }
        },
    )
    storage = AsyncMock()
    storage.get.return_value = None

    child_id = await generate_one_mutation(
        [base, donor],
        mutator=mutator,
        storage=storage,
        state_manager=AsyncMock(),
        iteration=3,
    )

    assert child_id is not None
    child = storage.add.await_args.args[0]
    parent_assignments = child.get_metadata(
        MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY
    )
    assert set(parent_assignments) == {base.id, donor.id}
    assert (
        AssignmentRecord.model_validate(
            parent_assignments[donor.id]
        ).context.parent_metrics
        == donor.metrics
    )
    provenance = {
        card_id: CardAssignmentSource.model_validate(source)
        for card_id, source in child.get_metadata(
            MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY
        ).items()
    }
    assert provenance["base-card"].decision_id == "decision-base"
    assert provenance["base-card"].bd_cell == (1,)
    assert provenance["donor-card"].decision_id == "decision-donor"
    assert provenance["donor-card"].bd_cell == (8,)
    mutation_assignment = MutationAssignmentRecord.model_validate(
        child.get_metadata(MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY)
    )
    assert mutation_assignment.delivered_ids == ("base-card", "donor-card")
    assert mutation_assignment.used_ids == ("base-card", "donor-card")
    assert mutation_assignment.ope_eligible is False

    child.metrics = {"is_valid": 1.0, "fitness": 0.8, "x": 4.0}
    rows = injection_outcomes_from_programs(
        [child], fitness_key="fitness", metrics_context=_metrics_context()
    )
    gains = compute_contextual_gains(
        rows, baseline_estimator=_ZeroBaseline(), task_key="test"
    )
    assert gains["base-card"][0].context.parent_id == base.id
    assert gains["base-card"][0].context.parent_metrics["x"] == 1.0
    assert gains["base-card"][0].gain == pytest.approx(0.3)
    assert gains["donor-card"][0].context.parent_id == donor.id
    assert gains["donor-card"][0].context.parent_metrics["x"] == 8.0
    assert gains["donor-card"][0].gain == pytest.approx(0.2)

    survivor = Card(
        id="base-card",
        description="merged",
        absorbed_ids=("donor-card",),
    )
    stamped = CardStatsStamper().stamp_gain_events(survivor, gains)
    assert {event.context.parent_id for event in stamped.gain_events} == {
        base.id,
        donor.id,
    }

    emitted = []
    monkeypatch.setattr("gigaevo.memory.outcomes.emit_memory_event", emitted.append)
    await record_program_memory_outcome(
        child, storage=storage, metrics_context=_metrics_context()
    )
    terminals = [event for event in emitted if isinstance(event, MemoryOutcome)]
    assert {event.decision_id for event in terminals} == {
        "decision-base",
        "decision-donor",
    }
    assert {event.base_id: event.fitness_delta for event in terminals} == {
        base.id: pytest.approx(0.3),
        donor.id: pytest.approx(0.2),
    }
    # Each terminal is scoped to ONE parent decision, so its cited-card set must
    # carry only that decision's own delivered card — never the global crossover
    # union. The union would violate the ledger guard (a terminal may only cite
    # the card its own decision delivered), hard-erroring cited crossover
    # children under the default num_parents>=2.
    assert {event.decision_id: event.used_card_ids for event in terminals} == {
        "decision-base": ("base-card",),
        "decision-donor": ("donor-card",),
    }
