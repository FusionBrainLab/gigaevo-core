"""Tests for ReconcileAgent: NEW/DUPLICATE/MERGE authoring over a code diff.

The agent is built through ``create_reconcile_agent`` so the externalized
prompt files and the ``{task_description}`` bake are exercised end-to-end.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.factories import create_reconcile_agent
from gigaevo.llm.agents.reconcile import (
    LibrarianCard,
    ReconcileItem,
    ReconcileResponse,
)
from gigaevo.memory.cards import Card


class _FakeStructuredLLM:
    """Stands in for ``llm.with_structured_output(ReconcileResponse)``."""

    def __init__(self, response: ReconcileResponse) -> None:
        self._response = response
        self.calls: list = []

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return self._response


class _FakeLLM:
    def __init__(self, response: ReconcileResponse) -> None:
        self._structured = _FakeStructuredLLM(response)

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        return self._structured


def _agent(llm: _FakeLLM, task: str = "maximize min area"):  # noqa: ANN202
    return create_reconcile_agent(llm, task_description=task)


@pytest.mark.asyncio
async def test_reconcile_returns_new_card_from_diff() -> None:
    resp = ReconcileResponse(
        items=[
            ReconcileItem(
                decision="NEW",
                card=LibrarianCard(
                    description="widen spectral gap before pruning",
                    keywords=["spectral"],
                ),
            )
        ]
    )
    agent = _agent(_FakeLLM(resp))
    out = await agent.arun(
        base_parent_code="def f(): return 1",
        child_code="def f(): return 2",
        note="bumped k",
        neighbors=[],
    )
    assert len(out.items) == 1
    assert out.items[0].decision == "NEW"
    assert "spectral" in out.items[0].card.keywords


@pytest.mark.asyncio
async def test_reconcile_empty_items_means_drop() -> None:
    agent = _agent(_FakeLLM(ReconcileResponse(items=[])))
    out = await agent.arun(
        base_parent_code="x",
        child_code="x",
        note="cosmetic",
        neighbors=[],
    )
    assert out.items == []


@pytest.mark.asyncio
async def test_reconcile_passes_neighbor_ids_into_prompt() -> None:
    fake = _FakeLLM(ReconcileResponse(items=[]))
    agent = _agent(fake)
    neighbor = Card(id="mem-N", description="existing lever", keywords=[])
    await agent.arun(
        base_parent_code="a",
        child_code="b",
        note="n",
        neighbors=[neighbor],
    )
    rendered = str(fake._structured.calls[0])
    assert "mem-N" in rendered


@pytest.mark.asyncio
async def test_reconcile_prompt_includes_unified_diff() -> None:
    fake = _FakeLLM(ReconcileResponse(items=[]))
    agent = _agent(fake)
    await agent.arun(
        base_parent_code="def f():\n    return 1\n",
        child_code="def f():\n    return 2\n",
        note="changed return",
        neighbors=[],
    )

    rendered = str(fake._structured.calls[0])
    assert "## UNIFIED DIFF" in rendered
    assert "--- base_parent.py" in rendered
    assert "+++ child.py" in rendered
    assert "-    return 1" in rendered
    assert "+    return 2" in rendered


@pytest.mark.asyncio
async def test_reconcile_passes_neighbor_why_and_keywords_into_prompt() -> None:
    fake = _FakeLLM(ReconcileResponse(items=[]))
    agent = _agent(fake)
    neighbor = Card(
        id="mem-N",
        description="existing lever",
        explanation_summary="WHY_MARKER it escapes the trap",
        keywords=["KW_MARKER"],
    )
    await agent.arun(
        base_parent_code="a",
        child_code="b",
        note="n",
        neighbors=[neighbor],
    )
    rendered = str(fake._structured.calls[0])
    assert "WHY_MARKER it escapes the trap" in rendered
    assert "KW_MARKER" in rendered


@pytest.mark.asyncio
async def test_reconcile_bakes_task_description_into_system_prompt() -> None:
    fake = _FakeLLM(ReconcileResponse(items=[]))
    agent = _agent(fake, task="PACK_TRIANGLES_TASK_MARKER")
    await agent.arun(base_parent_code="a", child_code="b", note="n", neighbors=[])
    rendered = str(fake._structured.calls[0])
    assert "PACK_TRIANGLES_TASK_MARKER" in rendered
