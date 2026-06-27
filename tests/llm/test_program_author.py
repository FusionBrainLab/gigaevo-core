"""Tests for ProgramAuthorAgent: exemplar prose from a top program.

The agent is built through ``create_program_author_agent`` so the externalized
prompt files and the ``{task_description}`` bake are exercised end-to-end.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.factories import create_program_author_agent
from gigaevo.llm.agents.program_author import ProgramAuthorResponse


class _FakeStructuredLLM:
    def __init__(self, response: ProgramAuthorResponse) -> None:
        self._response = response
        self.calls: list = []

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return self._response


class _FakeLLM:
    def __init__(self, response: ProgramAuthorResponse) -> None:
        self._structured = _FakeStructuredLLM(response)

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        return self._structured


def _agent(llm: _FakeLLM, task: str = "maximize min area"):  # noqa: ANN202
    return create_program_author_agent(llm, task_description=task)


@pytest.mark.asyncio
async def test_author_program_describes_what_it_does() -> None:
    agent = _agent(
        _FakeLLM(
            ProgramAuthorResponse(
                description="greedy spectral placement; scores well by maximizing "
                "the min pairwise area",
                keywords=["spectral", "greedy"],
            )
        )
    )
    out = await agent.arun(code="def solve(): ...", fitness=0.53)
    assert "spectral" in out.keywords
    assert out.description


@pytest.mark.asyncio
async def test_author_program_handles_unknown_fitness() -> None:
    fake = _FakeLLM(ProgramAuthorResponse(description="does a thing", keywords=["k"]))
    agent = _agent(fake)
    out = await agent.arun(code="def s(): ...", fitness=None)
    rendered = str(fake._structured.calls[0])
    assert "(unknown)" in rendered
    assert out.description == "does a thing"
