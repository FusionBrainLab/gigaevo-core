from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.llm.agents.equivalence import EquivalenceResponse
from gigaevo.llm.agents.factories import create_equivalence_agent
from gigaevo.memory.cards import Card
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
        assert schema is EquivalenceResponse
        return self.structured


@pytest.mark.asyncio
async def test_equivalence_renders_authored_candidate_and_neighbor_ids() -> None:
    llm = FakeLlm(
        EquivalenceResponse(decision=WriteDecision.EQUIVALENT, target_id="mem-existing")
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
        FakeLlm(EquivalenceResponse(decision=WriteDecision.NEW)),
        task_description="task",
    )
    result = await agent.arun(
        candidate=Card(id="", description="candidate", explanation_summary="why"),
        neighbors=[],
    )
    assert result == EquivalenceResponse(decision=WriteDecision.NEW)


def test_equivalence_schema_rejects_drop_and_bad_targets() -> None:
    with pytest.raises(ValidationError):
        EquivalenceResponse(decision=WriteDecision.DROP)
    with pytest.raises(ValidationError):
        EquivalenceResponse(decision=WriteDecision.EQUIVALENT)
    with pytest.raises(ValidationError):
        EquivalenceResponse(decision=WriteDecision.NEW, target_id="mem-x")
