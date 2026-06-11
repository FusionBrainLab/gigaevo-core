"""Integration tests: MemoryReadPipeline with real AmemGamMemory.

Memory injection into mutation prompts is now handled by the DAG pipeline
(MemoryContextStage → MutationContextStage), not by LLMMutationOperator.
These tests verify the read pipeline's search/parse/ID-extraction logic.
"""

from __future__ import annotations

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
)
from gigaevo.memory.core import LLMCardSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.agentic_memory import make_test_memory
from tests.fakes.read_pipeline import make_read_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SEED = 20260604
_PROVEN_STATS = {"ALL": {"posterior_a": 200.0, "posterior_b": 1.0}}


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
# MemoryReadPipeline with real AmemGamMemory
# ===========================================================================


class TestSelectorWithRealMemory:
    """Wire MemoryReadPipeline with pre-filled local AmemGamMemory."""

    def _make_pipeline(self, tmp_path, ideas):
        mem = make_test_memory(tmp_path)
        for idea in ideas:
            mem.save_card(idea)
        return make_read_pipeline(mem, seed=_SEED), mem

    @pytest.mark.asyncio
    async def test_search_returns_relevant_cards(self, tmp_path):
        pipeline, mem = self._make_pipeline(
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
                    "evolution_statistics": _PROVEN_STATS,
                },
                {
                    "id": "idea-2",
                    "description": "Filter low-confidence hops using threshold for fact checking",
                    "keywords": ["filter", "confidence", "fact", "verification"],
                    "task_description": "Multi-hop fact verification",
                },
            ],
        )
        mem.research = lambda *a, **k: _FakeResearchResult(_final_raw(["idea-1"]))
        parent = _make_program(code="def solve(x):\n    return x\n")

        selection = await pipeline.select(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy",
            max_cards=3,
        )

        assert "idea-1" in selection.card_ids
        assert len(selection.cards) > 0

    @pytest.mark.asyncio
    async def test_build_request_contains_parent_code(self, tmp_path):
        parent = _make_program(code="def solve(x):\n    return sorted(x)\n")

        query = LLMCardSelector().build_query(
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
        parent = _make_program(code="def f(): pass")
        parent.metadata[MUTATION_CONTEXT_METADATA_KEY] = (
            "Previous mutation improved sorting"
        )

        query = LLMCardSelector().build_query(
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
        pipeline, mem = self._make_pipeline(
            tmp_path,
            [
                {
                    "id": "idea-abc-123",
                    "description": "Use simulated annealing for local search",
                    "keywords": ["annealing"],
                    "evolution_statistics": _PROVEN_STATS,
                },
            ],
        )
        mem.research = lambda *a, **k: _FakeResearchResult(_final_raw(["idea-abc-123"]))
        parent = _make_program(code="def solve(x):\n    return x\n")

        selection = await pipeline.select(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="search task",
            metrics_description="fitness",
            max_cards=3,
        )

        assert "idea-abc-123" in selection.card_ids
        assert any("simulated annealing" in c for c in selection.cards)

    @pytest.mark.asyncio
    async def test_select_invalid_raw_memory_returns_empty(self, tmp_path):
        """raw_memory shape that fails Pydantic validation yields empty selection."""
        pipeline, mem = self._make_pipeline(tmp_path, [])

        class _BadRaw:
            integrated_memory = ""
            raw_memory = {"final_decision": {"mode": "nope", "top_ideas": "not-a-list"}}

        mem.research = lambda *a, **k: _BadRaw()  # type: ignore[method-assign]

        parent = _make_program(code="def f(): pass")
        selection = await pipeline.select(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=3,
        )

        assert selection.cards == []
        assert selection.card_ids == []
