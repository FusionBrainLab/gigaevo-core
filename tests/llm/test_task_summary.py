"""Tests for TaskSummaryAgent: a one-line condensation of the task description."""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.task_summary import TaskSummaryAgent, TaskSummaryResponse


class _FakeStructuredLLM:
    def __init__(self, response: TaskSummaryResponse) -> None:
        self._response = response
        self.calls: list = []

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return self._response


class _FakeLLM:
    def __init__(self, response: TaskSummaryResponse) -> None:
        self._structured = _FakeStructuredLLM(response)

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        return self._structured


@pytest.mark.asyncio
async def test_summary_condenses_task_description() -> None:
    fake = _FakeLLM(
        TaskSummaryResponse(summary="maximize the min pairwise triangle area")
    )
    agent = TaskSummaryAgent(fake)
    out = await agent.arun(
        task_description="Place N points in the unit square to maximize the minimum "
        "area over all triangles formed by triples of points. Report the layout."
    )
    assert out.summary == "maximize the min pairwise triangle area"
    rendered = str(fake._structured.calls[0])
    assert "Place N points" in rendered
