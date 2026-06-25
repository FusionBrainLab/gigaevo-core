"""End-to-end integration: memory read pipeline in the mutation loop.

Memory instructions are now injected via the DAG pipeline (MemoryContextStage),
not via explicit engine config flags. This file tests the MemoryReadPipeline
component that provides memory cards to the pipeline.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.program import Program
from tests.fakes.agentic_memory import make_test_memory
from tests.fakes.read_pipeline import make_read_pipeline

_SEED = 20260604
# 200 unblemished wins -> Beta(201, 1): an overwhelmingly-proven gain history so
# the card reliably beats the auction baseline prior and is always selected.
_PROVEN_GAINS = [
    {"context": {"parent_metrics": {"min_area": 0.5}}, "gain": 0.01} for _ in range(200)
]


def _make_memory(tmp_path, **overrides) -> AmemGamMemory:
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# Memory read pipeline in the mutation loop
# ===========================================================================


class TestMemorySelectorInMutationLoop:
    """Wire MemoryReadPipeline with real memory into the mutation flow."""

    @pytest.mark.asyncio
    async def test_selector_returns_cards_from_memory(self, tmp_path) -> None:
        """MemoryReadPipeline.select() returns cards from pre-filled memory."""
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "idea-1",
                "description": "Sort evidence by relevance score for better chain quality",
                "keywords": ["sort", "relevance", "evidence", "chain"],
                "gain_events": _PROVEN_GAINS,
            }
        )
        mem.save_card(
            {
                "id": "idea-2",
                "description": "Filter low-confidence hops using threshold",
                "keywords": ["filter", "confidence", "threshold"],
                "gain_events": _PROVEN_GAINS,
            }
        )

        class _FakeRaw:
            integrated_memory = ""
            raw_memory = {
                "final_decision": {
                    "mode": "final",
                    "top_ideas": [{"card_id": "idea-1"}, {"card_id": "idea-2"}],
                    "additional_queries": [],
                }
            }

        mem.research = lambda *a, **k: _FakeRaw()

        pipeline = make_read_pipeline(mem, seed=_SEED)

        parent = Program(
            code="def solve(x):\n    return x\n",
            metadata={},
        )

        selection = await pipeline.select(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy on validation set",
            max_cards=3,
        )

        # Should find relevant cards
        assert len(selection.cards) > 0, (
            "Pipeline returned no cards from pre-filled memory"
        )

        # Card IDs should be extractable
        assert isinstance(selection.card_ids, list)

    @pytest.mark.asyncio
    async def test_selector_with_empty_memory_returns_empty(self, tmp_path) -> None:
        """Pipeline over an empty card bank returns empty selection."""
        mem = _make_memory(tmp_path)  # Empty

        pipeline = make_read_pipeline(mem, seed=_SEED)

        selection = await pipeline.select(
            parents=[Program(code="def f(): pass", metadata={})],
            mutation_mode="rewrite",
            task_description="test",
            metrics_description="fitness",
            max_cards=3,
        )

        # Empty memory → "No relevant memories" → no cards parsed
        assert selection.cards == []
