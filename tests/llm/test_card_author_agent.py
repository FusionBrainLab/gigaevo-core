from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.llm.agents.card_author import AuthoredCard, CardAuthorResponse
from gigaevo.llm.agents.factories import create_card_author_agent
from gigaevo.llm.schema_compat import nonportable_keys
from gigaevo.memory.write.decisions import ArchiveStatus, WriteDecision


class FakeStructuredLlm:
    def __init__(self, response: CardAuthorResponse) -> None:
        self.response = response
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.response


class FakeLlm:
    def __init__(self, response: CardAuthorResponse) -> None:
        self.structured = FakeStructuredLlm(response)

    def with_structured_output(self, schema, **kwargs):
        assert schema["title"] == "CardAuthorResponse"
        assert nonportable_keys(schema) == set()
        return self.structured


def response(decision: WriteDecision) -> CardAuthorResponse:
    card = (
        AuthoredCard(
            description="When updates overshoot, try bounding the step because "
            "large gradients can leave the useful basin.",
            explanation_summary="A bounded step retains progress near steep regions.",
        )
        if decision is WriteDecision.NEW
        else None
    )
    return CardAuthorResponse(decision=decision, card=card)


@pytest.mark.asyncio
async def test_author_returns_at_most_one_candidate_and_renders_outcome() -> None:
    llm = FakeLlm(response(WriteDecision.NEW))
    agent = create_card_author_agent(llm, task_description="maximize quality")

    result = await agent.arun(
        base_parent_code="x = 1",
        child_code="x = 2",
        mutation_report="Change: bounded update\nMutator explanation: avoid overshoot",
        parent_fitness=0.4,
        child_fitness=0.6,
        signed_gain=0.2,
        higher_is_better=True,
        archive_status=ArchiveStatus.ARCHIVED,
    )

    assert result.decision is WriteDecision.NEW
    assert result.card is not None
    prompt = str(llm.structured.calls[0])
    for marker in (
        "0.4",
        "0.6",
        "0.2",
        "higher is better",
        "archived",
        "avoid overshoot",
    ):
        assert marker in prompt
    assert "--- base_parent.py" in prompt
    assert "+++ child.py" in prompt


@pytest.mark.asyncio
async def test_drop_has_no_card() -> None:
    agent = create_card_author_agent(
        FakeLlm(response(WriteDecision.DROP)), task_description="task"
    )
    result = await agent.arun(
        base_parent_code="x = 1",
        child_code="x = 1",
        mutation_report="rename",
        parent_fitness=None,
        child_fitness=0.1,
        signed_gain=None,
        higher_is_better=False,
        archive_status=ArchiveStatus.REJECTED,
    )
    assert result == CardAuthorResponse(decision=WriteDecision.DROP)


def test_author_schema_rejects_equivalent_and_inconsistent_payloads() -> None:
    decision_schema = CardAuthorResponse.model_json_schema()["properties"]["decision"]
    assert decision_schema["enum"] == ["DROP", "NEW"]
    with pytest.raises(ValidationError):
        CardAuthorResponse(decision=WriteDecision.EQUIVALENT)
    with pytest.raises(ValidationError):
        CardAuthorResponse(decision=WriteDecision.NEW)
    with pytest.raises(ValidationError):
        CardAuthorResponse(
            decision=WriteDecision.DROP,
            card=AuthoredCard(description="x", explanation_summary="y"),
        )
