"""Tests for TaskSummaryAgent: a one-line condensation of the task description.

The agent is built through ``create_task_summary_agent`` so the externalized
prompt files and the ``{task_description}`` injection are exercised end-to-end.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.factories import create_task_summary_agent
from gigaevo.llm.agents.task_summary import TaskSummaryResponse


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
    agent = create_task_summary_agent(fake)
    out = await agent.arun(
        task_description="Place N points in the unit square to maximize the minimum "
        "area over all triangles formed by triples of points. Report the layout."
    )
    assert out.summary == "maximize the min pairwise triangle area"
    rendered = str(fake._structured.calls[0])
    assert "Place N points" in rendered


def test_custom_prompts_dir_overrides_package_defaults(tmp_path) -> None:
    agent_dir = tmp_path / "task_summary"
    agent_dir.mkdir()
    (agent_dir / "system.txt").write_text("CUSTOM SYSTEM", encoding="utf-8")
    (agent_dir / "user.txt").write_text("CUSTOM {task_description}", encoding="utf-8")
    agent = create_task_summary_agent(
        _FakeLLM(TaskSummaryResponse(summary="x")), prompts_dir=tmp_path
    )
    state = agent.build_prompt({"task_description": "do X"})
    messages = state["messages"]
    assert messages[0].content == "CUSTOM SYSTEM"
    assert messages[1].content == "CUSTOM do X"
