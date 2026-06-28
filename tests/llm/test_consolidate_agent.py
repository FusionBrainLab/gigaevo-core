"""Tests for ConsolidateAgent: union authoring over two drifted near-dup cards.

Built through ``create_consolidate_agent`` so the externalized prompts and the
``{task_description}`` bake are exercised end-to-end.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.consolidate_cards import ConsolidateDecision
from gigaevo.llm.agents.factories import create_consolidate_agent
from gigaevo.llm.agents.reconcile import LibrarianCard
from gigaevo.memory.shared_memory.models import MemoryCard


class _FakeStructuredLLM:
    """Stands in for ``llm.with_structured_output(ConsolidateDecision)``."""

    def __init__(self, response: ConsolidateDecision) -> None:
        self._response = response
        self.calls: list = []

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return self._response


class _FakeLLM:
    def __init__(self, response: ConsolidateDecision) -> None:
        self._structured = _FakeStructuredLLM(response)

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        return self._structured


def _agent(llm: _FakeLLM, task: str = "maximize min area"):  # noqa: ANN202
    return create_consolidate_agent(llm, task_description=task)


@pytest.mark.asyncio
async def test_consolidate_returns_merge_decision_with_union_card() -> None:
    agent = _agent(
        _FakeLLM(
            ConsolidateDecision(
                merge=True, card=LibrarianCard(description="union lever")
            )
        )
    )
    out = await agent.arun(
        card_a=MemoryCard(id="mem-A", description="lever A"),
        card_b=MemoryCard(id="mem-B", description="lever B"),
    )
    assert out.merge is True
    assert out.card.description == "union lever"


@pytest.mark.asyncio
async def test_consolidate_can_abstain_when_cards_are_distinct() -> None:
    # The candidate gate only surfaces NEAR neighbors; the agent must be able to
    # decline a fold when the two cards name different levers (merge=False), so a
    # loosened candidate eps cannot force-merge distinct cards.
    agent = _agent(_FakeLLM(ConsolidateDecision(merge=False, card=None)))
    out = await agent.arun(
        card_a=MemoryCard(id="mem-A", description="lever A"),
        card_b=MemoryCard(id="mem-B", description="lever B"),
    )
    assert out.merge is False
    assert out.card is None


@pytest.mark.asyncio
async def test_consolidate_passes_card_why_and_keywords_into_prompt() -> None:
    fake = _FakeLLM(
        ConsolidateDecision(merge=True, card=LibrarianCard(description="union"))
    )
    agent = _agent(fake)
    card_a = MemoryCard(
        id="mem-A",
        description="lever A",
        explanation_summary="WHY_A_MARKER",
        keywords=["KW_A_MARKER"],
    )
    card_b = MemoryCard(
        id="mem-B",
        description="lever B",
        explanation_summary="WHY_B_MARKER",
        keywords=["KW_B_MARKER"],
    )
    await agent.arun(card_a=card_a, card_b=card_b)
    rendered = str(fake._structured.calls[0])
    assert "WHY_A_MARKER" in rendered
    assert "WHY_B_MARKER" in rendered
    assert "KW_A_MARKER" in rendered
    assert "KW_B_MARKER" in rendered
