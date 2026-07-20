from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.llm.agents.equivalence import (
    AxisVerdict,
    EquivalenceResponse,
    ProgramAxisComparison,
)
from gigaevo.llm.agents.factories import create_equivalence_agent
from gigaevo.llm.schema_compat import nonportable_keys
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.write.decisions import WriteDecision


class FakeStructuredLlm:
    def __init__(self, response: EquivalenceResponse) -> None:
        self.response = response
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.response


class FakeLlm:
    def __init__(self, response: EquivalenceResponse) -> None:
        self.structured = FakeStructuredLlm(response)

    def with_structured_output(self, schema, **kwargs):
        assert schema["title"] == "EquivalenceResponse"
        assert nonportable_keys(schema) == set()
        assert list(schema["properties"]) == [
            "comparison_summary",
            "program_axes",
            "decision",
            "target_id",
        ]
        assert schema["required"] == [
            "comparison_summary",
            "program_axes",
            "decision",
        ]
        return self.structured


def program_axes(
    neighbor_id: str,
    *,
    output_policy: AxisVerdict = AxisVerdict.MATCH,
) -> ProgramAxisComparison:
    return ProgramAxisComparison(
        neighbor_id=neighbor_id,
        applicability=AxisVerdict.MATCH,
        representation_or_state=AxisVerdict.MATCH,
        core_procedure=AxisVerdict.MATCH,
        decision_logic=AxisVerdict.MATCH,
        update_or_output_policy=output_policy,
        essential_constraints=AxisVerdict.MATCH,
    )


@pytest.mark.asyncio
async def test_equivalence_renders_authored_candidate_and_neighbor_ids() -> None:
    llm = FakeLlm(
        EquivalenceResponse(
            comparison_summary="same action",
            program_axes=None,
            decision=WriteDecision.EQUIVALENT,
            target_id="mem-existing",
        )
    )
    agent = create_equivalence_agent(llm, task_description="task")
    candidate = Card(
        id="",
        description="When C holds, try A because M.",
        explanation_summary="candidate why",
    )
    neighbor = Card(
        id="mem-existing",
        description="When C holds, try A because M.",
        explanation_summary="existing why",
    )

    result = await agent.arun(candidate=candidate, neighbors=[neighbor])

    assert result.target_id == neighbor.id
    prompt = str(llm.structured.calls[0])
    assert "candidate why" in prompt
    assert "mem-existing" in prompt
    assert "existing why" in prompt


@pytest.mark.asyncio
async def test_equivalence_can_return_new_without_rewriting_payload() -> None:
    agent = create_equivalence_agent(
        FakeLlm(
            EquivalenceResponse(
                comparison_summary="different action",
                program_axes=None,
                decision=WriteDecision.NEW,
            )
        ),
        task_description="task",
    )
    result = await agent.arun(
        candidate=Card(id="", description="candidate", explanation_summary="why"),
        neighbors=[
            Card(
                id="mem-existing",
                description="different neighbor",
                explanation_summary="different why",
            )
        ],
    )
    assert result == EquivalenceResponse(
        comparison_summary="different action",
        program_axes=None,
        decision=WriteDecision.NEW,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    [AxisVerdict.DIFFERENT, AxisVerdict.UNSPECIFIED],
)
async def test_program_equivalence_requires_every_factorized_axis_to_match(
    mismatch: AxisVerdict,
) -> None:
    neighbor_id = "program-existing"
    agent = create_equivalence_agent(
        FakeLlm(
            EquivalenceResponse(
                decision=WriteDecision.EQUIVALENT,
                target_id=neighbor_id,
                comparison_summary="output policies differ",
                program_axes=program_axes(
                    neighbor_id,
                    output_policy=mismatch,
                ),
            )
        ),
        task_description="task",
    )
    candidate = Card(
        id="program-candidate",
        kind=CardKind.PROGRAM,
        program_id="candidate",
        description="candidate strategy",
    )
    neighbor = Card(
        id=neighbor_id,
        kind=CardKind.PROGRAM,
        program_id="existing",
        description="existing strategy",
    )

    result = await agent.arun(candidate=candidate, neighbors=[neighbor])

    assert result.decision is WriteDecision.NEW
    assert result.target_id == ""


@pytest.mark.asyncio
async def test_matching_program_axes_override_a_conservative_model_decision() -> None:
    neighbor_id = "program-existing"
    agent = create_equivalence_agent(
        FakeLlm(
            EquivalenceResponse(
                decision=WriteDecision.NEW,
                comparison_summary="all load-bearing axes match",
                program_axes=program_axes(neighbor_id),
            )
        ),
        task_description="task",
    )
    candidate = Card(
        id="program-candidate",
        kind=CardKind.PROGRAM,
        program_id="candidate",
        description="candidate strategy",
    )
    neighbor = Card(
        id=neighbor_id,
        kind=CardKind.PROGRAM,
        program_id="existing",
        description="existing strategy",
    )

    result = await agent.arun(candidate=candidate, neighbors=[neighbor])

    assert result.decision is WriteDecision.EQUIVALENT
    assert result.target_id == neighbor_id


@pytest.mark.asyncio
async def test_equivalent_program_rejects_target_axes_disagreement() -> None:
    agent = create_equivalence_agent(
        FakeLlm(
            EquivalenceResponse(
                comparison_summary="same family",
                program_axes=program_axes("program-axes"),
                decision=WriteDecision.EQUIVALENT,
                target_id="program-target",
            )
        ),
        task_description="task",
    )
    candidate = Card(
        id="program-candidate",
        kind=CardKind.PROGRAM,
        program_id="candidate",
        description="candidate strategy",
    )
    neighbors = [
        Card(
            id=neighbor_id,
            kind=CardKind.PROGRAM,
            program_id=neighbor_id,
            description="neighbor strategy",
        )
        for neighbor_id in ("program-axes", "program-target")
    ]

    with pytest.raises(ValueError, match="disagrees"):
        await agent.arun(candidate=candidate, neighbors=neighbors)


@pytest.mark.asyncio
async def test_equivalence_rejects_empty_neighbor_slate_without_calling_llm() -> None:
    llm = FakeLlm(
        EquivalenceResponse(
            comparison_summary="different action",
            program_axes=None,
            decision=WriteDecision.NEW,
        )
    )
    agent = create_equivalence_agent(llm, task_description="task")

    with pytest.raises(ValueError, match="at least one"):
        await agent.arun(
            candidate=Card(id="", description="candidate"),
            neighbors=[],
        )

    assert llm.structured.calls == []


def test_equivalence_schema_rejects_drop_and_bad_targets() -> None:
    schema = EquivalenceResponse.model_json_schema()
    assert list(schema["properties"]) == [
        "comparison_summary",
        "program_axes",
        "decision",
        "target_id",
    ]
    assert schema["required"] == [
        "comparison_summary",
        "program_axes",
        "decision",
    ]
    decision_schema = schema["properties"]["decision"]
    assert decision_schema["enum"] == ["NEW", "EQUIVALENT"]
    with pytest.raises(ValidationError):
        EquivalenceResponse(
            comparison_summary="invalid decision",
            program_axes=None,
            decision=WriteDecision.DROP,
        )
    with pytest.raises(ValidationError):
        EquivalenceResponse(
            comparison_summary="same action",
            program_axes=None,
            decision=WriteDecision.EQUIVALENT,
        )
    with pytest.raises(ValidationError):
        EquivalenceResponse(
            comparison_summary="different action",
            program_axes=None,
            decision=WriteDecision.NEW,
            target_id="mem-x",
        )
    with pytest.raises(ValidationError):
        EquivalenceResponse(
            comparison_summary="missing explicit nullable axes",
            decision=WriteDecision.NEW,
        )
