"""Unit tests for the structured-output path in MemorySelectorAgent.

These pin the contract that:
- ``select()`` extracts card IDs from ``ExperimentalDecision.top_ideas[].card_id``
  via Pydantic validation (no regex on prose).
- ``select()`` resolves card text via ``memory.get_card(card_id).description``
  (no regex on prose).
- Invalid ``raw_memory`` shapes degrade to an empty selection, not a crash.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gigaevo.llm.agents.memory_selector import (
    MemorySelection,
    MemorySelectorAgent,
)
from gigaevo.memory._vendor.GAM_root.gam.schemas.result import ExperimentalDecision
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


class _StubResearchOutput:
    def __init__(self, *, integrated_memory: str = "", raw_memory: Any = None) -> None:
        self.integrated_memory = integrated_memory
        self.raw_memory = raw_memory


class _StubMemory:
    """Minimal memory backend exposing ``research`` + ``get_card`` like AmemGamMemory."""

    def __init__(self, *, raw_memory: Any, cards: dict[str, Any] | None = None) -> None:
        self._raw_memory = raw_memory
        self._cards = cards or {}

    def research(self, request: str, memory_state: str | None = None):
        return _StubResearchOutput(integrated_memory="", raw_memory=self._raw_memory)

    def get_card(self, card_id: str) -> Any:
        return self._cards.get(card_id)


def _make_selector(memory: Any) -> MemorySelectorAgent:
    selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
    selector._search_lock = asyncio.Lock()
    selector._backend_error = None
    selector.memory = memory
    return selector


def _make_program(code: str = "def solve(): return 1") -> Program:
    return Program(code=code, state=ProgramState.DONE)


@pytest.mark.asyncio
async def test_select_pulls_ids_from_top_ideas_card_id():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "idea-A"}, {"card_id": "idea-B"}],
                "additional_queries": [],
            }
        },
        cards={
            "idea-A": {"description": "Try simulated annealing"},
            "idea-B": {"description": "Filter low-confidence hops"},
        },
    )
    selector = _make_selector(memory)

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=2,
    )

    assert selection.card_ids == ["idea-A", "idea-B"]
    assert any("simulated annealing" in c for c in selection.cards)
    assert any("low-confidence" in c for c in selection.cards)


@pytest.mark.asyncio
async def test_select_resolves_description_from_pydantic_card():
    class _PydanticCard:
        description = "Use a heap for sorted retrieval"

    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": "idea-1"}],
                "additional_queries": [],
            }
        },
        cards={"idea-1": _PydanticCard()},
    )
    selector = _make_selector(memory)

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=1,
    )

    assert selection.cards == ["Use a heap for sorted retrieval"]
    assert selection.card_ids == ["idea-1"]


@pytest.mark.asyncio
async def test_select_respects_max_cards_limit():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [{"card_id": f"id-{i}"} for i in range(5)],
                "additional_queries": [],
            }
        },
        cards={f"id-{i}": {"description": f"card {i}"} for i in range(5)},
    )
    selector = _make_selector(memory)

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=2,
    )

    assert selection.card_ids == ["id-0", "id-1"]
    assert selection.cards == ["card 0", "card 1"]


@pytest.mark.asyncio
async def test_select_invalid_raw_memory_returns_empty():
    memory = _StubMemory(
        raw_memory={"final_decision": {"mode": "nope", "top_ideas": "not-a-list"}}
    )
    selector = _make_selector(memory)

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=3,
    )

    assert selection == MemorySelection(cards=[], card_ids=[])


@pytest.mark.asyncio
async def test_select_missing_final_decision_returns_empty():
    memory = _StubMemory(raw_memory={"other_key": "irrelevant"})
    selector = _make_selector(memory)

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=3,
    )

    assert selection == MemorySelection(cards=[], card_ids=[])


@pytest.mark.asyncio
async def test_select_skips_missing_cards_silently():
    memory = _StubMemory(
        raw_memory={
            "final_decision": {
                "mode": "final",
                "top_ideas": [
                    {"card_id": "exists"},
                    {"card_id": "missing"},
                ],
                "additional_queries": [],
            }
        },
        cards={"exists": {"description": "real card"}},
    )
    selector = _make_selector(memory)

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=5,
    )

    assert selection.card_ids == ["exists", "missing"]
    assert selection.cards == ["real card"]


@pytest.mark.asyncio
async def test_select_research_exception_returns_empty():
    class _ThrowingMemory:
        def research(self, request, memory_state=None):
            raise RuntimeError("backend exploded")

        def get_card(self, card_id):
            return None

    selector = _make_selector(_ThrowingMemory())

    selection = await selector.select(
        input=[_make_program()],
        mutation_mode="rewrite",
        task_description="t",
        metrics_description="m",
        memory_text="",
        max_cards=3,
    )

    assert selection == MemorySelection(cards=[], card_ids=[])


def test_parse_final_decision_handles_non_dict_raw_memory():
    decision = MemorySelectorAgent._parse_final_decision(None)
    assert isinstance(decision, ExperimentalDecision)
    assert decision.top_ideas == []

    decision = MemorySelectorAgent._parse_final_decision("not a dict")
    assert decision.top_ideas == []


def test_render_card_handles_dict_pydantic_and_none():
    assert MemorySelectorAgent._render_card(None) == ""
    assert MemorySelectorAgent._render_card({"description": " trim me  "}) == "trim me"
    assert MemorySelectorAgent._render_card({"description": None}) == ""

    class _Obj:
        description = "via attribute"

    assert MemorySelectorAgent._render_card(_Obj()) == "via attribute"
