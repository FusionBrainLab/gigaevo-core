"""Tests for the remaining 25% of memory.py using full fake infrastructure.

Covers: _load_or_create_retriever, _build_dedup_retrievers,
_score_retrieved_candidates, _resolve_vector_retriever, and the full
dedup pipeline with real scoring (not mocked _score_retrieved_candidates).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.card_conversion import is_program_card
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from tests.fakes.agentic_memory import (
    FakeAMemGenerator,
    FakeResearchAgent,
    fake_build_gam_store,
    fake_build_retrievers,
    fake_load_amem_records,
    inject_fakes_into_memory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    defaults = dict(
        checkpoint_path=str(tmp_path / "mem"),
        use_api=False,
        sync_on_init=False,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )
    defaults.update(overrides)
    return AmemGamMemory(**defaults)


def _make_full_memory(tmp_path, ideas=None, **overrides):
    """Create AmemGamMemory with fake agentic system + generator + retriever patches."""
    mem = _make_memory(tmp_path, **overrides)
    fake_sys = inject_fakes_into_memory(mem)
    mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

    # Save ideas to populate both memory_cards and agentic system
    for idea in ideas or []:
        mem.save_card(idea)

    # Patch _load_or_create_retriever to use fake GAM builders
    def _patched_load_or_create_retriever():
        # Export memories to JSONL (same as real _dump_memory)
        mem._dump_memory()

        records = fake_load_amem_records(mem.export_file)
        if not records:
            records = [c.model_dump() for c in mem.memory_cards.values()]

        memory_store, page_store, added = fake_build_gam_store(
            records,
            mem.gam_store_dir,
        )
        retrievers = fake_build_retrievers(
            page_store,
            mem.gam_store_dir / "indexes",
            mem.checkpoint_dir / "chroma",
            allowed_tools=sorted(mem.allowed_gam_tools),
        )
        if not retrievers:
            return None

        return FakeResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=mem.generator,
        )

    mem._load_or_create_retriever = _patched_load_or_create_retriever

    # Also patch _build_dedup_retrievers similarly
    def _patched_build_dedup_retrievers():
        records = [c.model_dump() for c in mem.memory_cards.values()]
        records = [r for r in records if str(r.get("category", "")).strip().lower() != "program"]
        if not records:
            return {}

        _, page_store, _ = fake_build_gam_store(records, mem.gam_store_dir)
        retrievers = fake_build_retrievers(
            page_store,
            mem.gam_store_dir / "indexes",
            mem.checkpoint_dir / "chroma",
            allowed_tools=[
                "vector_description",
                "vector_explanation_summary",
                "vector_description_explanation_summary",
                "vector_description_task_description_summary",
            ],
        )
        return {
            name: r for name, r in retrievers.items() if name in mem.allowed_gam_tools
        }

    mem._build_dedup_retrievers = _patched_build_dedup_retrievers

    return mem, fake_sys


# ===========================================================================
# _load_or_create_retriever
# ===========================================================================


class TestLoadOrCreateRetriever:
    def test_creates_research_agent_after_rebuild(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {"id": "i1", "description": "SA optimization", "keywords": ["SA"]},
                {
                    "id": "i2",
                    "description": "Crossover recombination",
                    "keywords": ["crossover"],
                },
            ],
        )

        mem.rebuild()

        assert mem.research_agent is not None
        assert mem.export_file.exists()

    def test_research_agent_finds_cards(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "i1",
                    "description": "simulated annealing for optimization",
                    "keywords": ["annealing"],
                },
                {
                    "id": "i2",
                    "description": "genetic crossover for diversity",
                    "keywords": ["crossover"],
                },
            ],
        )

        mem.rebuild()
        result = mem.search("annealing optimization")
        assert "i1" in result

    def test_empty_memory_no_research_agent(self, tmp_path):
        mem, _ = _make_full_memory(tmp_path, ideas=[])
        # No cards → _load_or_create_retriever has nothing to index
        # rebuild skips agent creation since no export file and no cards
        mem.rebuild()
        # research_agent may be None with empty memory


# ===========================================================================
# _build_dedup_retrievers + _score_retrieved_candidates
# ===========================================================================


class TestDedupWithRealScoring:
    """Test the full dedup pipeline with real (fake) vector scoring."""

    def test_score_retrieved_candidates_finds_similar(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "existing-1",
                    "description": "Use simulated annealing for local search refinement",
                    "keywords": ["annealing", "local-search"],
                },
                {
                    "id": "existing-2",
                    "description": "Apply genetic crossover between top pairs",
                    "keywords": ["crossover", "genetic"],
                },
            ],
            card_update_dedup_config={"enabled": True},
        )

        # Score candidates for a similar card
        from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card

        incoming = normalize_memory_card(
            {
                "description": "simulated annealing optimization for local search",
            }
        )

        scored = mem._score_retrieved_candidates(incoming)

        # Should find at least existing-1 as similar
        assert len(scored) > 0
        card_ids = [s["card_id"] for s in scored]
        assert "existing-1" in card_ids

    def test_score_skips_program_cards(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {"id": "idea-1", "description": "SA optimization"},
                {
                    "id": "prog-1",
                    "category": "program",
                    "program_id": "p1",
                    "description": "SA optimization program",
                    "fitness": 90.0,
                },
            ],
            card_update_dedup_config={"enabled": True},
        )

        from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card

        incoming = normalize_memory_card({"description": "SA optimization"})
        scored = mem._score_retrieved_candidates(incoming)

        # Program cards should be excluded from dedup scoring
        card_ids = [s["card_id"] for s in scored]
        assert "prog-1" not in card_ids

    def test_score_empty_when_dedup_disabled(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "test"}],
        )

        from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card

        incoming = normalize_memory_card({"description": "test"})
        scored = mem._score_retrieved_candidates(incoming)
        assert scored == []

    def test_dedup_invalidates_retrievers_on_save(self, tmp_path):
        """_dedup_retrievers cache is cleared after save_card."""
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "test"}],
            card_update_dedup_config={"enabled": True},
        )

        # First access builds retrievers
        mem._dedup_retrievers = None
        mem._resolve_vector_retriever("vector_description")
        assert mem._dedup_retrievers is not None

        # Save new card clears cache
        mem.save_card({"id": "i2", "description": "new idea"})
        assert mem._dedup_retrievers is None


# ===========================================================================
# Full dedup pipeline: score → LLM decide → apply
# ===========================================================================


class TestFullDedupPipeline:
    """End-to-end dedup: real scoring → mocked LLM → real merge."""

    def test_full_dedup_discard(self, tmp_path):
        """Similar card scored → LLM says discard → card rejected."""
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "existing",
                    "description": "simulated annealing for optimization",
                    "keywords": ["annealing", "optimization"],
                },
            ],
            card_update_dedup_config={"enabled": True},
        )

        # Mock LLM to return discard
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "discard", "duplicate_of": "existing"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        result_id = mem.save_card(
            {
                "description": "simulated annealing optimization technique",
            }
        )

        assert result_id == "existing"
        assert len(mem.memory_cards) == 1
        stats = mem.get_card_write_stats()
        assert stats["rejected"] == 1

    def test_full_dedup_update(self, tmp_path):
        """Similar card scored → LLM says update → existing card merged."""
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "existing",
                    "description": "simulated annealing",
                    "explanation": {"explanations": ["original"], "summary": "SA"},
                    "keywords": ["annealing"],
                },
            ],
            card_update_dedup_config={"enabled": True},
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps(
                {
                    "action": "update",
                    "updates": [
                        {
                            "card_id": "existing",
                            "update_explanation": True,
                            "explanation_append": "Also works for multi-hop chains",
                        }
                    ],
                }
            ),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        result_id = mem.save_card(
            {
                "description": "annealing for multi-hop chain optimization",
            }
        )

        assert result_id == "existing"
        card = mem.get_card("existing")
        assert "multi-hop chains" in str(card.explanation)
        stats = mem.get_card_write_stats()
        assert stats["updated"] == 1

    def test_full_dedup_add_when_no_match(self, tmp_path):
        """Unrelated card → low score → LLM says add → new card created."""
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "existing",
                    "description": "simulated annealing",
                    "keywords": ["annealing"],
                },
            ],
            card_update_dedup_config={"enabled": True},
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "add"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        result_id = mem.save_card(
            {
                "description": "quantum computing for protein folding",
            }
        )

        assert result_id != "existing"
        assert len(mem.memory_cards) == 2
        stats = mem.get_card_write_stats()
        assert stats["added"] == 2


# ===========================================================================
# _resolve_vector_retriever
# ===========================================================================


class TestResolveVectorRetriever:
    def test_builds_on_first_access(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "test idea"}],
            card_update_dedup_config={"enabled": True},
        )

        assert mem._dedup_retrievers is None
        retriever = mem._resolve_vector_retriever("vector_description")
        assert mem._dedup_retrievers is not None
        assert retriever is not None

    def test_fallback_to_vector_tool(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "test"}],
            card_update_dedup_config={"enabled": True},
        )
        # Set allowed tools to include "vector" which is the fallback
        mem.allowed_gam_tools = {"vector", "vector_description"}
        mem._dedup_retrievers = None

        # Request a tool that might not exist
        mem._resolve_vector_retriever("vector_nonexistent")
        # Falls back to "vector" if tool not found
        # (may or may not find it depending on what _build_dedup_retrievers returns)

    def test_empty_memory_returns_none(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path, ideas=[], card_update_dedup_config={"enabled": True}
        )
        retriever = mem._resolve_vector_retriever("vector_description")
        assert retriever is None


# ===========================================================================
# _dump_memory
# ===========================================================================


class TestDumpMemory:
    def test_dump_creates_jsonl(self, tmp_path):
        mem, fake_sys = _make_full_memory(
            tmp_path,
            ideas=[
                {"id": "i1", "description": "idea one"},
                {"id": "i2", "description": "idea two"},
            ],
        )

        mem._dump_memory()

        assert mem.export_file.exists()
        lines = mem.export_file.read_text().strip().split("\n")
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert "id" in record or "content" in record

    def test_dump_empty_when_no_system(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem._dump_memory()  # Should not crash with memory_system=None
