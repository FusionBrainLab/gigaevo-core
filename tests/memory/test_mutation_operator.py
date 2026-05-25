"""Integration tests: MemorySelectorAgent with real AmemGamMemory.

Memory injection into mutation prompts is now handled by the DAG pipeline
(MemoryContextStage → MutationContextStage), not by LLMMutationOperator.
These tests verify the MemorySelectorAgent search/parse/ID-extraction logic.
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
)
from gigaevo.llm.agents.memory_selector import MemorySelectorAgent
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(code="def solve(): return 1", **metadata):
    p = Program(code=code, state=ProgramState.DONE)
    p.metadata.update(metadata)
    return p


class _FakeResearchResult:
    """Minimal stand-in for the gigaevo.memory red-agent research result."""

    def __init__(self, raw_memory):
        self.integrated_memory = ""
        self.raw_memory = raw_memory


def _final_raw(card_ids):
    return {
        "final_decision": {
            "mode": "final",
            "top_ideas": [{"card_id": cid} for cid in card_ids],
            "additional_queries": [],
        }
    }


# ===========================================================================
# MemorySelectorAgent with real AmemGamMemory
# ===========================================================================


class TestSelectorWithRealMemory:
    """Wire MemorySelectorAgent with pre-filled local AmemGamMemory."""

    def _make_selector(self, tmp_path, ideas):
        mem = make_test_memory(tmp_path)
        for idea in ideas:
            mem.save_card(idea)

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem
        return selector

    @pytest.mark.asyncio
    async def test_search_returns_relevant_cards(self, tmp_path):
        selector = self._make_selector(
            tmp_path,
            [
                {
                    "id": "idea-1",
                    "description": "Sort evidence by relevance score for multi-hop verification",
                    "keywords": [
                        "sort",
                        "relevance",
                        "evidence",
                        "verification",
                        "multi",
                    ],
                    "task_description": "Multi-hop fact verification",
                },
                {
                    "id": "idea-2",
                    "description": "Filter low-confidence hops using threshold for fact checking",
                    "keywords": ["filter", "confidence", "fact", "verification"],
                    "task_description": "Multi-hop fact verification",
                },
            ],
        )
        selector.memory.research = lambda *a, **k: _FakeResearchResult(
            _final_raw(["idea-1"])
        )
        parent = _make_program(code="def solve(x):\n    return x\n")

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy",
            memory_text="",
            max_cards=3,
        )

        assert "idea-1" in selection.card_ids
        assert len(selection.cards) > 0

    @pytest.mark.asyncio
    async def test_build_request_contains_parent_code(self, tmp_path):
        selector = self._make_selector(tmp_path, [])
        parent = _make_program(code="def solve(x):\n    return sorted(x)\n")

        query = selector._build_request(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy on validation set",
            max_cards=3,
        )

        assert "MUTATION INPUTS" in query
        assert "TASK DESCRIPTION:" in query
        assert "Multi-hop fact verification" in query
        assert "AVAILABLE METRICS:" in query
        assert "accuracy on validation set" in query
        assert "MUTATION MODE:" in query
        assert "rewrite" in query
        assert "def solve(x):" in query
        assert "return sorted(x)" in query
        assert "Search your memory database" in query
        assert "pick up to 3 card(s)" in query

    @pytest.mark.asyncio
    async def test_build_request_includes_mutation_context(self, tmp_path):
        """Parent with mutation_context metadata → appears in request."""
        selector = self._make_selector(tmp_path, [])
        parent = _make_program(code="def f(): pass")
        parent.metadata[MUTATION_CONTEXT_METADATA_KEY] = (
            "Previous mutation improved sorting"
        )

        query = selector._build_request(
            parents=[parent],
            mutation_mode="diff",
            task_description="test task",
            metrics_description="fitness",
            max_cards=1,
        )

        assert "Previous mutation improved sorting" in query
        assert "diff" in query

    @pytest.mark.asyncio
    async def test_select_resolves_card_text_from_structured_top_ideas(self, tmp_path):
        """select() pulls card.description for each id in final_decision.top_ideas."""
        selector = self._make_selector(
            tmp_path,
            [
                {
                    "id": "idea-abc-123",
                    "description": "Use simulated annealing for local search",
                    "keywords": ["annealing"],
                },
            ],
        )
        selector.memory.research = lambda *a, **k: _FakeResearchResult(
            _final_raw(["idea-abc-123"])
        )
        parent = _make_program(code="def solve(x):\n    return x\n")

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="search task",
            metrics_description="fitness",
            memory_text="",
            max_cards=3,
        )

        assert "idea-abc-123" in selection.card_ids
        assert any("simulated annealing" in c for c in selection.cards)

    @pytest.mark.asyncio
    async def test_select_invalid_raw_memory_returns_empty(self, tmp_path):
        """raw_memory shape that fails Pydantic validation yields empty selection."""
        selector = self._make_selector(tmp_path, [])

        class _BadRaw:
            integrated_memory = ""
            raw_memory = {"final_decision": {"mode": "nope", "top_ideas": "not-a-list"}}

        selector.memory.research = lambda *a, **k: _BadRaw()  # type: ignore[method-assign]

        parent = _make_program(code="def f(): pass")
        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            memory_text="",
            max_cards=3,
        )

        assert selection.cards == []
        assert selection.card_ids == []
