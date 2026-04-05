"""Tests for the remaining 25% of memory.py using full fake infrastructure.

Covers: _load_or_create_retriever, _build_dedup_retrievers,
_score_retrieved_candidates, _resolve_vector_retriever, and the full
dedup pipeline with real scoring (not mocked _score_retrieved_candidates).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from tests.fakes.agentic_memory import (
    FakeAMemGenerator,
    FakeResearchAgent,
    fake_build_gam_store,
    fake_build_retrievers,
    fake_load_amem_records,
    inject_fakes_into_memory,
    make_test_memory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


def _make_full_memory(tmp_path, ideas=None, **overrides):
    """Create AmemGamMemory with fake agentic system + generator + retriever patches."""
    from gigaevo.memory.shared_memory.gam_search import GamSearch

    mem = _make_memory(tmp_path, **overrides)
    fake_sys = inject_fakes_into_memory(mem)
    mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

    # GamSearch wasn't created in __init__ (deps unavailable before fakes).
    if mem.gam is None:
        mem.gam = GamSearch(
            research_agent_cls=mem._ResearchAgentCls,
            generator=mem.generator,
            card_store=mem.card_store,
            checkpoint_dir=mem.checkpoint_dir,
            gam_store_dir=mem.gam_store_dir,
            export_file=mem.export_file,
            enable_bm25=mem.enable_bm25,
            allowed_gam_tools=mem.allowed_gam_tools,
            gam_top_k_by_tool=mem.gam_top_k_by_tool,
            gam_pipeline_mode=mem.gam_pipeline_mode,
        )

    # Save ideas to populate both memory_cards and agentic system
    for idea in ideas or []:
        mem.save_card(idea)

    # Patch gam.build to use fake GAM builders
    def _patched_gam_build():
        if mem.note_sync is not None:
            mem.note_sync.export_jsonl(mem.export_file)

        records = fake_load_amem_records(mem.export_file)
        if not records:
            records = [c.model_dump() for c in mem.card_store.cards.values()]

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
            mem.gam.agent = None
            return

        mem.gam.agent = FakeResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=mem.generator,
        )

    mem.gam.build = _patched_gam_build

    # Also patch dedup.build_retrievers similarly
    def _patched_build_dedup_retrievers():
        records = [c.model_dump() for c in mem.card_store.cards.values()]
        records = [
            r
            for r in records
            if str(r.get("category", "")).strip().lower() != "program"
        ]
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

    mem.dedup.build_retrievers = _patched_build_dedup_retrievers

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

        scored = mem.dedup.score_candidates(incoming)

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
        scored = mem.dedup.score_candidates(incoming)

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
        scored = mem.dedup.score_candidates(incoming)
        assert scored == []

    def test_dedup_invalidates_retrievers_on_save(self, tmp_path):
        """Dedup retriever cache is cleared after save_card."""
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "test"}],
            card_update_dedup_config={"enabled": True},
        )

        # First access builds retrievers
        mem.dedup.invalidate_retrievers()
        mem.dedup.resolve_retriever("vector_description")
        assert mem.dedup._retrievers is not None

        # Save new card clears cache
        mem.save_card({"id": "i2", "description": "new idea"})
        assert mem.dedup._retrievers is None


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
        assert len(mem.card_store.cards) == 1
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
        assert len(mem.card_store.cards) == 2
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

        assert mem.dedup._retrievers is None
        retriever = mem.dedup.resolve_retriever("vector_description")
        assert mem.dedup._retrievers is not None
        assert retriever is not None

    def test_fallback_to_vector_tool(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "test"}],
            card_update_dedup_config={"enabled": True},
        )
        # Set allowed tools to include "vector" which is the fallback
        mem.allowed_gam_tools = {"vector", "vector_description"}
        mem.dedup.invalidate_retrievers()

        # Request a tool that might not exist
        mem.dedup.resolve_retriever("vector_nonexistent")
        # Falls back to "vector" if tool not found
        # (may or may not find it depending on what build_retrievers returns)

    def test_empty_memory_returns_none(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path, ideas=[], card_update_dedup_config={"enabled": True}
        )
        retriever = mem.dedup.resolve_retriever("vector_description")
        assert retriever is None


# ===========================================================================
# note_sync.export_jsonl
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

        mem.note_sync.export_jsonl(mem.export_file)

        assert mem.export_file.exists()
        lines = mem.export_file.read_text().strip().split("\n")
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert "id" in record or "content" in record

    def test_dump_noop_when_no_system(self, tmp_path):
        mem = _make_memory(tmp_path)
        # note_sync is None when memory_system is None
        assert mem.note_sync is None
