from __future__ import annotations

import pytest

from gigaevo.llm.agents.card_author import AuthoredCard
from gigaevo.llm.agents.factories import create_program_author_agent
from gigaevo.llm.agents.program_author import ProgramAuthorResponse
from gigaevo.memory.write.decisions import WriteDecision


class FakeStructuredLlm:
    def __init__(self, response: ProgramAuthorResponse) -> None:
        self.response = response
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.response


class FakeLlm:
    def __init__(self, response: ProgramAuthorResponse) -> None:
        self.structured = FakeStructuredLlm(response)

    def with_structured_output(self, schema, **kwargs):
        assert schema is ProgramAuthorResponse
        return self.structured


@pytest.mark.asyncio
async def test_program_author_returns_one_holistic_hypothesis() -> None:
    expected = ProgramAuthorResponse(
        decision=WriteDecision.NEW,
        card=AuthoredCard(
            description="When a constructive seed is brittle, try a guarded local "
            "search because feasible swaps refine it without restarting.",
            explanation_summary="The seed reaches a useful basin and guarded swaps "
            "exploit it while retaining feasibility.",
        ),
    )
    llm = FakeLlm(expected)
    agent = create_program_author_agent(llm, task_description="task")

    result = await agent.arun(code="def solve(): ...", fitness=0.53, archive_rank=2)

    assert result == expected
    rendered = str(llm.structured.calls[0])
    assert "0.53" in rendered
    assert "2" in rendered


@pytest.mark.asyncio
async def test_program_author_can_drop_uninformative_program() -> None:
    expected = ProgramAuthorResponse(decision=WriteDecision.DROP)
    agent = create_program_author_agent(FakeLlm(expected), task_description="task")
    assert (
        await agent.arun(code="pass", fitness=None, archive_rank=None)
    ).decision is WriteDecision.DROP
