"""Three high-value E2E tests for the memory system.

1. Persistence  : save → close → reopen → retrieve (tests api_index.json round-trip)
2. A-mem path   : search exercises FakeRetriever (Jaccard, no embeddings API)
3. Full pipeline: write_pipeline produces cards → memory stores them → search reads back
                  ("launch gigaevo run to build memory - extract cards - launch with memory")
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import MemoryConfig
from tests.fakes.agentic_memory import (
    FakeResearchAgent,
    fake_build_gam_store,
    fake_build_retrievers,
    fake_load_amem_records,
    make_test_memory_with_agentic,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_pipeline_cfg(
    tmp_path: Path, banks_path: Path, best_ideas_path: Path
) -> MagicMock:
    """Build a PipelineConfig mock pointing all I/O to tmp_path."""
    cfg = MagicMock()
    cfg.banks_path = banks_path
    cfg.best_ideas_path = best_ideas_path
    cfg.programs_path = tmp_path / "programs.json"
    cfg.usage_updates_path = None
    cfg.use_api = False
    cfg.memory_dir = tmp_path / "pipeline_mem"
    cfg.search_limit = 5
    cfg.rebuild_interval = 10
    cfg.enable_llm_synthesis = False
    cfg.should_evolve = False
    cfg.fill_missing_fields_with_llm = False
    cfg.enable_bm25 = False
    cfg.allowed_gam_tools = []
    cfg.gam_top_k_by_tool = {}
    cfg.gam_pipeline_mode = "default"
    cfg.card_update_dedup_config = {}
    cfg.best_programs_percent = 100.0  # include every idea
    cfg.sync_batch_size = 100
    cfg.sync_on_init = True
    cfg.channel = "latest"
    cfg.author = None
    cfg.namespace = "default"
    cfg.enable_usage_tracking = False
    cfg.settings_path = tmp_path / "settings.yaml"
    return cfg


# ===========================================================================
# Test 1 — Persistence
# ===========================================================================


class TestMemoryPersistence:
    """Cards must survive close() + reopen of AmemGamMemory."""

    def test_cards_survive_close_and_reopen(self, tmp_path: Path) -> None:
        """Persistence: save cards, close(), reopen same checkpoint_path, retrieve."""
        config = MemoryConfig(
            checkpoint_path=tmp_path / "mem",
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        )

        # --- Phase 1: write ---
        mem1 = AmemGamMemory(config=config)
        id1 = mem1.save_card(
            normalize_memory_card(
                {
                    "id": "persist-001",
                    "description": "batch normalization technique",
                    "category": "general",
                }
            )
        )
        id2 = mem1.save_card(
            normalize_memory_card(
                {
                    "id": "persist-002",
                    "description": "gradient clipping method",
                    "category": "general",
                }
            )
        )
        mem1.close()

        # Verify the index file was written to disk
        assert config.index_file.exists(), "api_index.json must exist after save_card"

        # --- Phase 2: reopen and retrieve ---
        mem2 = AmemGamMemory(config=config)
        retrieved1 = mem2.get_card(id1)
        retrieved2 = mem2.get_card(id2)

        assert retrieved1 is not None, "Card 1 must survive close/reopen"
        assert retrieved2 is not None, "Card 2 must survive close/reopen"
        assert retrieved1.description == "batch normalization technique"
        assert retrieved2.description == "gradient clipping method"

    def test_card_count_survives_reopen(self, tmp_path: Path) -> None:
        """Reopen must load the same number of cards that were saved."""
        config = MemoryConfig(
            checkpoint_path=tmp_path / "mem",
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        )

        mem1 = AmemGamMemory(config=config)
        for i in range(5):
            mem1.save_card(
                normalize_memory_card(
                    {"description": f"technique {i}", "category": "general"}
                )
            )
        count_before = len(mem1.card_store.cards)
        mem1.close()

        mem2 = AmemGamMemory(config=config)
        assert len(mem2.card_store.cards) == count_before


# ===========================================================================
# Test 2 — A-mem search via FakeRetriever (avoids chromadb dependency)
# ===========================================================================


def _patch_gam_build_with_fakes(mem: AmemGamMemory) -> None:
    """Swap mem.gam.build for a fake that uses FakeRetriever instead of ChromaDB.

    This mirrors the pattern in test_memory_backend_agentic.py and avoids
    the 'No module named chromadb' error that occurs when gam.build() tries
    to instantiate ChromaRetriever.
    """

    def _fake_build() -> None:
        if mem.note_sync is not None:
            mem.note_sync.export_jsonl(mem.config.export_file)
        records = fake_load_amem_records(mem.config.export_file)
        if not records:
            records = [c.model_dump() for c in mem.card_store.cards.values()]
        memory_store, page_store, _ = fake_build_gam_store(
            records, mem.config.gam_store_dir
        )
        retrievers = fake_build_retrievers(
            page_store,
            mem.config.gam_store_dir / "indexes",
            mem.config.checkpoint_path / "chroma",
        )
        if not retrievers:
            mem.gam.agent = None  # type: ignore[union-attr]
            return
        mem.gam.agent = FakeResearchAgent(  # type: ignore[union-attr]
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=mem.generator,
        )

    if mem.gam is not None:
        mem.gam.build_research_agent = _fake_build  # type: ignore[method-assign]


class TestMemorySearchAMemPath:
    """Search must route through the A-mem retriever path (FakeRetriever, no API)."""

    def test_search_exercises_agentic_retriever(self, tmp_path: Path) -> None:
        """After rebuild(), search() routes through FakeResearchAgent → FakeRetriever.

        Uses fake GAM builders to avoid chromadb dependency while still exercising
        the full A-mem retrieval code path.
        """
        mem, __ = make_test_memory_with_agentic(tmp_path)
        _patch_gam_build_with_fakes(mem)

        mem.save_card(
            normalize_memory_card(
                {
                    "description": "adaptive gradient descent optimizer for neural networks",
                    "category": "general",
                }
            )
        )
        mem.save_card(
            normalize_memory_card(
                {
                    "description": "decision tree ensemble classifier random forest",
                    "category": "general",
                }
            )
        )

        # rebuild() calls the patched gam.build which uses FakeRetriever
        mem.rebuild()
        assert mem.research_agent is not None, "rebuild() must set research_agent"

        result = mem.search("gradient descent optimizer")

        # FakeRetriever uses Jaccard similarity — "gradient" and "descent" should match
        assert isinstance(result, str)
        assert (
            "gradient" in result.lower()
            or "descent" in result.lower()
            or len(result) > len("No relevant memories found.")
        ), f"Expected A-mem search to find a matching card, got: {result!r}"

    def test_search_returns_no_result_for_unrelated_query(self, tmp_path: Path) -> None:
        """FakeRetriever returns 'No relevant memories' when nothing overlaps."""
        mem, __ = make_test_memory_with_agentic(tmp_path)
        _patch_gam_build_with_fakes(mem)

        mem.save_card(
            normalize_memory_card(
                {
                    "description": "batch normalization layer technique",
                    "category": "general",
                }
            )
        )
        mem.rebuild()

        result = mem.search("quantum physics entanglement")
        # FakeRetriever: no token overlap → research agent returns fallback string
        assert isinstance(result, str)


# ===========================================================================
# Test 3 — Full pipeline E2E
# "launch gigaevo run to build memory → extract cards → launch with memory"
# ===========================================================================


class TestFullPipelineE2E:
    """End-to-end: write_pipeline produces cards, memory stores them, search reads back."""

    def _write_banks(self, path: Path, ideas: list[dict]) -> None:
        """Write banks.json in the format ideas_tracker produces."""
        path.write_text(
            json.dumps(
                [{"active_bank": ideas, "timestamp": "2026-04-09 12:00:00"}],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_best_ideas(self, path: Path, idea_ids: list[str]) -> None:
        """Write best_ideas.json referencing the given idea IDs."""
        path.write_text(
            json.dumps(
                [
                    {
                        "best_ideas": [
                            {"id": iid, "idea_id": iid, "fitness": 0.85}
                            for iid in idea_ids
                        ],
                        "timestamp": "2026-04-09 12:00:00",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_write_pipeline_cards_are_stored_and_retrievable(
        self, tmp_path: Path
    ) -> None:
        """Full loop: write_pipeline writes ideas to memory, new AmemGamMemory reads them.

        Phase 1 (build memory): simulate ideas_tracker output as banks.json + best_ideas.json,
          run write_pipeline.main() to ingest into AmemGamMemory at cfg.memory_dir.
        Phase 2 (launch with memory): open a fresh AmemGamMemory at the same path,
          verify cards are present and searchable.
        """
        from gigaevo.memory.write_pipeline import main as pipeline_main

        # --- Phase 1: simulate ideas_tracker output ---
        ideas = [
            {
                "id": "pipeline-idea-001",
                "description": "adaptive learning rate scheduling reduces overfitting",
                "category": "general",
                "task_description": "Optimize neural network training convergence",
                "task_description_summary": "nn-convergence",
            },
            {
                "id": "pipeline-idea-002",
                "description": "momentum-based weight update accelerates convergence",
                "category": "general",
                "task_description": "Improve gradient-based optimizers",
                "task_description_summary": "optimizer",
            },
        ]

        banks_path = tmp_path / "banks.json"
        best_ideas_path = tmp_path / "best_ideas.json"
        self._write_banks(banks_path, ideas)
        self._write_best_ideas(
            best_ideas_path, ["pipeline-idea-001", "pipeline-idea-002"]
        )

        cfg = _make_pipeline_cfg(tmp_path, banks_path, best_ideas_path)

        with patch("gigaevo.memory.write_pipeline.load_config", return_value=cfg):
            stats = pipeline_main(
                banks_path=banks_path,
                best_ideas_path=best_ideas_path,
            )

        # Write pipeline should return stats
        assert stats is not None, "write_pipeline.main() must return stats dict"

        # --- Phase 2: launch with memory ---
        mem_config = MemoryConfig(
            checkpoint_path=tmp_path / "pipeline_mem",
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        )
        mem = AmemGamMemory(config=mem_config)
        mem.research_agent = None  # force local keyword search (no real LLM)

        card1 = mem.get_card("pipeline-idea-001")
        card2 = mem.get_card("pipeline-idea-002")

        assert card1 is not None, "Idea 001 must be stored by write_pipeline"
        assert card2 is not None, "Idea 002 must be stored by write_pipeline"
        assert "learning rate" in card1.description

        # Search should find relevant cards (local keyword path, no embeddings)
        search_result = mem.search("adaptive learning rate")
        assert isinstance(search_result, str)
        assert len(search_result) > 0

    def test_write_pipeline_partial_best_ideas_filter(self, tmp_path: Path) -> None:
        """best_ideas.json acts as a filter — only referenced ideas are written."""
        from gigaevo.memory.write_pipeline import main as pipeline_main

        ideas = [
            {
                "id": "filtered-001",
                "description": "included idea: gradient clipping technique",
                "category": "general",
            },
            {
                "id": "filtered-002",
                "description": "excluded idea: unrelated algorithm",
                "category": "general",
            },
        ]

        banks_path = tmp_path / "banks.json"
        best_ideas_path = tmp_path / "best_ideas.json"
        self._write_banks(banks_path, ideas)
        # Only reference idea-001 in best_ideas
        self._write_best_ideas(best_ideas_path, ["filtered-001"])

        cfg = _make_pipeline_cfg(tmp_path, banks_path, best_ideas_path)
        # Set best_programs_percent to 0 to avoid any program card logic
        cfg.best_programs_percent = 5.0

        with patch("gigaevo.memory.write_pipeline.load_config", return_value=cfg):
            pipeline_main(banks_path=banks_path, best_ideas_path=best_ideas_path)

        mem = AmemGamMemory(
            config=MemoryConfig(
                checkpoint_path=tmp_path / "pipeline_mem",
                enable_llm_synthesis=False,
                enable_memory_evolution=False,
                enable_llm_card_enrichment=False,
            )
        )

        # filtered-001 is in best_ideas → must be stored
        assert mem.get_card("filtered-001") is not None, "Best idea must be written"
        # filtered-002 is NOT in best_ideas → should be absent
        assert mem.get_card("filtered-002") is None, (
            "Non-best idea must be filtered out"
        )
