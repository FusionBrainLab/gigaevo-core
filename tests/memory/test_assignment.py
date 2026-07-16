from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _assignment(*, assigned_ids: tuple[str, ...]) -> AssignmentRecord:
    predicted_help = {card_id: 0.75 for card_id in assigned_ids}
    return AssignmentRecord(
        decision_id="memsel-test",
        policy_version="TestAuctioneer:abc123",
        task_key="hover",
        ordered_eligible_ids=("card-b", "card-a"),
        assigned_ids=assigned_ids,
        delivered_ids=assigned_ids,
        arm="injected" if assigned_ids else "none",
        predicted_help=predicted_help,
        predicted_gain={card_id: 0.3 for card_id in assigned_ids},
        predicted_no_card_gain={card_id: 0.05 for card_id in assigned_ids},
        context=DecisionContext(
            task_key="hover",
            parent_metrics={"fitness": 0.75},
            parent_id="parent-1",
            search_phase="iteration:12",
        ),
        bd_cell=(2, 4),
        timestamp=datetime(2026, 7, 14, tzinfo=UTC),
    )


def test_assignment_record_json_round_trip() -> None:
    assignment = _assignment(assigned_ids=("card-a",))

    restored = AssignmentRecord.model_validate(assignment.model_dump(mode="json"))

    assert restored == assignment
    assert restored.schema_version == 2
    assert restored.delivered_ids == ("card-a",)
    assert restored.ope_eligible is False
    assert restored.probe_arm == "none"
    assert restored.predicted_help == {"card-a": 0.75}
    assert restored.predicted_gain == {"card-a": 0.3}
    assert restored.predicted_no_card_gain == {"card-a": 0.05}


def test_decision_context_validates_legacy_four_field_row() -> None:
    context = DecisionContext.model_validate(
        {
            "task_key": "hover",
            "parent_metrics": {"fitness": 0.5},
            "parent_id": "parent-1",
            "timestamp": None,
        }
    )

    assert context.search_phase == ""
    assert context.parent_quality_quantile is None
    assert context.local_opportunity_count is None
    assert context.local_visit_count is None


@pytest.mark.parametrize("assigned_ids", [("card-a",), ()])
async def test_assignment_rides_onto_born_child(
    assigned_ids: tuple[str, ...],
) -> None:
    assignment = _assignment(assigned_ids=assigned_ids)
    parent = Program(code="def parent(): return 1", state=ProgramState.DONE)
    parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, list(assigned_ids))
    parent.set_metadata(
        MUTATION_MEMORY_DECISION_ID_METADATA_KEY, assignment.decision_id
    )
    parent.set_metadata(
        MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
        assignment.model_dump(mode="json"),
    )
    mutator = AsyncMock()
    mutator.mutate_single.return_value = MutationSpec(
        code="def child(): return 2",
        parents=[parent],
        name="assignment test",
    )
    storage = AsyncMock()
    storage.get.return_value = None
    storage.mget.return_value = []
    state_manager = AsyncMock()

    child_id = await generate_one_mutation(
        [parent],
        mutator=mutator,
        storage=storage,
        state_manager=state_manager,
        iteration=13,
    )

    assert child_id is not None
    child = storage.add.await_args.args[0]
    frozen = AssignmentRecord.model_validate(
        child.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
    )
    assert child.get_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY) == (
        assignment.decision_id
    )
    assert child.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY) == list(
        assigned_ids
    )
    assert frozen == assignment
    assert frozen.arm == ("injected" if assigned_ids else "none")
