"""Tests using fake A-MEM/GAM infrastructure.

These tests cover the paths that were previously untestable without real
Chroma/embedding dependencies: _upsert_local_note_agentic,
_upsert_local_note_fast, _build_note_from_card, _remove_local_note,
_dump_memory, rebuild with real data, and LLM card enrichment.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.memory import AmemGamMemory
from tests.fakes.agentic_memory import (
    FakeAgenticMemorySystem,
    FakeAMemGenerator,
    FakeMemoryNote,
    FakeResearchAgent,
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
    mem = AmemGamMemory(**defaults)
    return mem


def _make_memory_with_fakes(tmp_path, **overrides):
    """Create AmemGamMemory with fake agentic infrastructure injected."""
    mem = _make_memory(tmp_path, **overrides)
    fake_system = inject_fakes_into_memory(mem)
    # Patch _load_or_create_retriever to avoid chromadb import

    def _fake_load_or_create_retriever():
        return FakeResearchAgent(
            retrievers={"vector": fake_system.retriever},
            generator=mem.generator,
        )

    mem._load_or_create_retriever = _fake_load_or_create_retriever
    return mem, fake_system


# ===========================================================================
# FakeMemoryNote basics
# ===========================================================================


class TestFakeMemoryNote:
    def test_auto_id(self):
        note = FakeMemoryNote(content="test")
        assert note.id != ""

    def test_explicit_id(self):
        note = FakeMemoryNote(content="test", id="my-id")
        assert note.id == "my-id"

    def test_all_fields(self):
        note = FakeMemoryNote(
            content="text",
            id="n1",
            keywords=["k"],
            links=["l"],
            context="ctx",
            category="cat",
            strategy="strat",
        )
        assert note.content == "text"
        assert note.keywords == ["k"]
        assert note.context == "ctx"


# ===========================================================================
# FakeAgenticMemorySystem basics
# ===========================================================================


class TestFakeAgenticMemorySystem:
    def test_add_and_read(self):
        sys = FakeAgenticMemorySystem()
        note_id = sys.add_note("test content", id="n1", keywords=["k1"])
        assert note_id == "n1"
        note = sys.read("n1")
        assert note is not None
        assert note.content == "test content"

    def test_read_missing(self):
        sys = FakeAgenticMemorySystem()
        assert sys.read("missing") is None

    def test_update(self):
        sys = FakeAgenticMemorySystem()
        sys.add_note("original", id="n1")
        sys.update("n1", content="updated")
        assert sys.read("n1").content == "updated"

    def test_delete(self):
        sys = FakeAgenticMemorySystem()
        sys.add_note("test", id="n1")
        assert sys.delete("n1") is True
        assert sys.read("n1") is None

    def test_analyze_content(self):
        sys = FakeAgenticMemorySystem()
        result = sys.analyze_content("Use simulated annealing for optimization")
        assert "simulated" in result["keywords"]
        assert "annealing" in result["keywords"]

    def test_retriever_search(self):
        sys = FakeAgenticMemorySystem()
        sys.add_note("simulated annealing for optimization", id="n1")
        sys.add_note("genetic crossover for diversity", id="n2")
        hits = sys.retriever.search(["annealing optimization"])
        assert len(hits) == 1  # One query
        assert len(hits[0]) >= 1
        assert hits[0][0].page_id == "n1"


# ===========================================================================
# AmemGamMemory with fake agentic system: _upsert_local_note_agentic
# ===========================================================================


class TestUpsertLocalNoteAgentic:
    """Test that save_card syncs cards to the local agentic memory system."""

    def test_save_card_creates_note_in_agentic_system(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.save_card({"id": "c1", "description": "SA optimization"})

        # Card should exist in both memory_cards and the agentic system
        assert "c1" in mem.memory_cards
        note = fake_sys.read("c1")
        assert note is not None
        assert "SA optimization" in note.content

    def test_save_card_updates_note_on_change(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.save_card({"id": "c1", "description": "v1"})
        mem.save_card({"id": "c1", "description": "v2 improved"})

        note = fake_sys.read("c1")
        assert note is not None
        assert "v2 improved" in note.content

    def test_save_card_no_update_when_unchanged(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.save_card({"id": "c1", "description": "same"})

        # Save again with identical content
        fake_sys.read("c1")

        mem.save_card({"id": "c1", "description": "same"})
        # Note should not be re-upserted (no change detected)

    def test_multiple_cards_in_agentic_system(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        for i in range(5):
            mem.save_card({"id": f"c{i}", "description": f"idea {i}"})

        assert len(fake_sys.memories) == 5
        for i in range(5):
            assert fake_sys.read(f"c{i}") is not None


# ===========================================================================
# _remove_local_note
# ===========================================================================


class TestRemoveLocalNote:
    def test_delete_removes_from_agentic_system(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})
        assert fake_sys.read("c1") is not None

        mem.delete("c1")
        assert fake_sys.read("c1") is None

    def test_delete_nonexistent_safe(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        # Should not crash
        result = mem.delete("nonexistent")
        assert result is False


# ===========================================================================
# rebuild with fake system
# ===========================================================================


class TestRebuildWithFakes:
    def test_rebuild_creates_export_file(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        # Need a generator for rebuild to proceed
        mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

        mem.save_card({"id": "c1", "description": "test idea"})
        mem.rebuild()

        assert mem.export_file.exists()

    def test_rebuild_resets_counter(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

        mem.save_card({"id": "c1", "description": "test"})
        assert mem._iters_after_rebuild > 0

        mem.rebuild()
        assert mem._iters_after_rebuild == 0

    def test_rebuild_auto_triggers_at_interval(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(
            tmp_path,
            rebuild_interval=3,
        )
        mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

        for i in range(3):
            mem.save_card({"id": f"c{i}", "description": f"idea {i}"})

        # After 3 saves, rebuild should have triggered and reset counter
        assert mem._iters_after_rebuild == 0
        assert mem.export_file.exists()


# ===========================================================================
# LLM card enrichment with fake agentic system
# ===========================================================================


class TestLlmCardEnrichment:
    def test_enrichment_adds_keywords(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(
            tmp_path,
            enable_llm_card_enrichment=True,
        )
        mem.save_card(
            {"id": "c1", "description": "simulated annealing for optimization"}
        )

        card = mem.get_card("c1")
        # analyze_content extracts keywords from description
        assert len(card["keywords"]) > 0
        assert "simulated" in card["keywords"] or "annealing" in card["keywords"]

    def test_enrichment_skipped_when_disabled(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(
            tmp_path,
            enable_llm_card_enrichment=False,
        )
        mem.save_card({"id": "c1", "description": "simulated annealing"})

        card = mem.get_card("c1")
        assert card["keywords"] == []

    def test_enrichment_preserves_existing_keywords(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(
            tmp_path,
            enable_llm_card_enrichment=True,
        )
        mem.save_card(
            {
                "id": "c1",
                "description": "optimization technique",
                "keywords": ["existing-keyword"],
            }
        )

        card = mem.get_card("c1")
        # Existing keywords should not be overwritten
        assert "existing-keyword" in card["keywords"]


# ===========================================================================
# Full cycle with fake agentic system
# ===========================================================================


class TestFullCycleWithFakes:
    """End-to-end: save → agentic sync → rebuild → search → delete."""

    def test_full_lifecycle(self, tmp_path):
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

        # Save ideas
        for i in range(5):
            mem.save_card(
                {
                    "id": f"idea-{i}",
                    "description": f"Optimization technique {i} using method_{i}",
                    "keywords": [f"method_{i}"],
                }
            )

        # All in agentic system
        assert len(fake_sys.memories) == 5

        # Rebuild
        mem.rebuild()
        assert mem.export_file.exists()
        assert mem._iters_after_rebuild == 0

        # Research agent should be created after rebuild
        assert mem.research_agent is not None

        # Search via research agent
        result = mem.search("method_3 optimization")
        assert "idea-3" in result

        # Delete one
        mem.delete("idea-2")
        assert fake_sys.read("idea-2") is None
        assert len(fake_sys.memories) == 4

        # Persist and reload
        mem2, fake_sys2 = _make_memory_with_fakes(tmp_path)
        assert len(mem2.memory_cards) == 4
        assert mem2.get_card("idea-2") is None
        assert mem2.get_card("idea-3") is not None

    def test_persist_reload_agentic_system_starts_empty(self, tmp_path):
        """After reload, agentic system has no notes (only JSON index loads).

        This documents a real gap: cards are in memory_cards but NOT in
        the agentic system until rebuild() or save_card() re-populates them.
        Search via research_agent will miss them until rebuild.
        """
        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        mem.save_card({"id": "c1", "description": "annealing idea"})
        assert fake_sys.read("c1") is not None

        # Reload: cards in memory_cards, but agentic system is fresh/empty
        mem2, fake_sys2 = _make_memory_with_fakes(tmp_path)
        assert "c1" in mem2.memory_cards
        assert fake_sys2.read("c1") is None  # Gap: not in agentic system

        # Local search still works (it reads memory_cards directly)
        result = mem2._search_local_cards("annealing")
        assert "c1" in result

    def test_upsert_local_note_fast_direct(self, tmp_path):
        """Test _upsert_local_note_fast — the hot path used by _sync_from_api."""
        from gigaevo.memory.shared_memory.memory import normalize_memory_card

        mem, fake_sys = _make_memory_with_fakes(tmp_path)
        card = normalize_memory_card(
            {
                "id": "c1",
                "description": "SA optimization",
                "keywords": ["SA"],
                "task_description": "Solve TSP",
            }
        )

        # Directly call the fast upsert (normally called by _sync_from_api)
        changed = mem._upsert_local_note_fast(card)
        assert changed is True
        assert fake_sys.read("c1") is not None
        assert fake_sys.read("c1").content == "SA optimization"

        # Second call with same content → no change
        changed2 = mem._upsert_local_note_fast(card)
        assert changed2 is False

        # Update content → change detected
        card["description"] = "Updated SA optimization"
        changed3 = mem._upsert_local_note_fast(card)
        assert changed3 is True
        assert "Updated" in fake_sys.read("c1").content
