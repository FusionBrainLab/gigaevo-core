from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
)
from gigaevo.memory.cards import AssignmentRecord, DecisionContext
from gigaevo.memory.events import MemoryDelivery
from gigaevo.memory.provider import MemoryProvider
from gigaevo.memory.read.reader import (
    MemorySelection,
    _policy_version,
    extend_policy_version,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.memory_context import MemoryContextStage


class _Component(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model_name: str = ""
    embedding_model: str = ""


class _FixedProvider(MemoryProvider):
    def __init__(self, selection: MemorySelection) -> None:
        self._selection = selection

    async def select_cards(self, program: Program, **kwargs) -> MemorySelection:
        del program, kwargs
        return self._selection


def _version(*, model_name: str = "research-v1") -> str:
    return _policy_version(
        shortlister=_Component(
            name="retrieval", model_name=model_name, embedding_model="embed-v3"
        ),
        reputation=_Component(name="reputation"),
        auctioneer=_Component(name="auction"),
        budgeter=_Component(name="budget"),
        context_model=_Component(name="bd-context"),
        candidate_projector=_Component(name="projection/no-card"),
        probe_policy=_Component(name="probe"),
        renderer=_Component(name="render"),
        exclusion_policy=_Component(name="lineage-exclusion"),
        downstream_delivery={
            "no_card_control_probability": 0.1,
            "reverse_repack": False,
        },
        max_cards=2,
    )


def test_policy_version_is_canonical_and_covers_model_identifiers() -> None:
    assert _version() == _version()
    assert _version(model_name="research-v1") != _version(model_name="research-v2")
    assert _version().startswith("MemoryPolicy:")


@pytest.mark.asyncio
async def test_downstream_delivery_logs_assigned_and_withheld_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = []
    monkeypatch.setattr(
        "gigaevo.programs.stages.memory_context.emit_memory_event", emitted.append
    )
    assignment = AssignmentRecord(
        decision_id="memsel-delivery",
        policy_version="read-policy",
        task_key="test",
        assigned_ids=("card-a",),
        delivered_ids=("card-a",),
        arm="injected",
        context=DecisionContext(task_key="test"),
    )
    stage = MemoryContextStage(
        memory_provider=_FixedProvider(
            MemorySelection(
                cards=("card text",),
                card_ids=("card-a",),
                decision_id=assignment.decision_id,
                assignment=assignment,
            )
        ),
        task_description="task",
        metrics_description="metrics",
        no_card_control_probability=1.0,
        reverse_repack=True,
        timeout=5.0,
    )
    stage.attach_inputs({})
    program = Program(code="def solve(): return 1")

    result = await stage.compute(program)

    final = AssignmentRecord.model_validate(
        program.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
    )
    assert result.data == ""
    assert final.assigned_ids == ("card-a",)
    assert final.delivered_ids == ()
    assert final.ope_eligible is False
    assert final.policy_version == extend_policy_version(
        "read-policy",
        downstream_delivery={
            "fresh_context_reorder": True,
            "no_card_control_probability": 1.0,
            "reverse_repack": True,
        },
    )
    assert program.get_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY) is True
    (event,) = emitted
    assert isinstance(event, MemoryDelivery)
    assert event.assigned_ids == ("card-a",)
    assert event.delivered_ids == ()
    assert event.withheld_for_control is True
    assert event.assignment == final


@pytest.mark.asyncio
async def test_probe_arm_is_never_routed_into_no_card_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A randomized cold-probe assignment must NEVER be withheld by the no-card
    # control lane: dropping a probe arm's card would leave probe_arm labelling a
    # unit that received nothing, biasing the DR-AIPW probe-ITT. The probe lane
    # wins even at no_card_control_probability=1.0 (would withhold any non-probe).
    emitted = []
    monkeypatch.setattr(
        "gigaevo.programs.stages.memory_context.emit_memory_event", emitted.append
    )
    assignment = AssignmentRecord(
        decision_id="memsel-probe",
        policy_version="read-policy",
        task_key="test",
        assigned_ids=("card-a",),
        delivered_ids=("card-a",),
        arm="injected",
        probe_arm="treated",
        randomized=True,
        propensity_kind="probe_bernoulli",
        propensities={"card-a": 0.5},
        ope_eligible=True,
        context=DecisionContext(task_key="test"),
    )
    stage = MemoryContextStage(
        memory_provider=_FixedProvider(
            MemorySelection(
                cards=("card text",),
                card_ids=("card-a",),
                decision_id=assignment.decision_id,
                assignment=assignment,
            )
        ),
        task_description="task",
        metrics_description="metrics",
        no_card_control_probability=1.0,
        reverse_repack=True,
        timeout=5.0,
    )
    stage.attach_inputs({})
    program = Program(code="def solve(): return 1")

    result = await stage.compute(program)

    final = AssignmentRecord.model_validate(
        program.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
    )
    assert result.data != ""
    assert final.delivered_ids == ("card-a",)
    assert program.get_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY) is False
    (event,) = emitted
    assert isinstance(event, MemoryDelivery)
    assert event.delivered_ids == ("card-a",)
    assert event.withheld_for_control is False
