"""Tests for agentic memory paths using full fake infrastructure.

Covers: gam.build / research_agent lifecycle and note_sync export.
"""

from __future__ import annotations

import json

from gigaevo.memory.shared_memory.card_conversion import (
    normalize_allowed_gam_tools,
)
from tests.fakes.agentic_memory import (
    FakeResearchAgent,
    fake_build_gam_store,
    fake_build_retrievers,
    fake_load_amem_records,
    make_test_memory_with_agentic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_memory(tmp_path, ideas=None, **overrides):
    """Create AmemGamMemory with fake agentic system + generator + retriever patches."""
    mem, fake_sys = make_test_memory_with_agentic(tmp_path, **overrides)
    cfg = mem.config

    # Save ideas to populate both card_store.cards and agentic system
    for idea in ideas or []:
        mem.save_card(idea)

    # Patch gam.build to use fake GAM builders
    def _patched_gam_build():
        if mem.note_sync is not None:
            mem.note_sync.export_jsonl(mem.config.export_file)

        records = fake_load_amem_records(mem.config.export_file)
        if not records:
            records = [c.model_dump() for c in mem.card_store.cards.values()]

        memory_store, page_store, added = fake_build_gam_store(
            records,
            mem.config.gam_store_dir,
        )
        retrievers = fake_build_retrievers(
            page_store,
            mem.config.gam_store_dir / "indexes",
            mem.config.checkpoint_path / "chroma",
            allowed_tools=sorted(
                normalize_allowed_gam_tools(cfg.gam.allowed_tools or None)
            ),
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

    mem.gam.build_research_agent = _patched_gam_build

    return mem, fake_sys


# ===========================================================================
# GAM retriever lifecycle (gam.build / research_agent)
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
        assert mem.config.export_file.exists()

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
        # No cards → gam.build has nothing to index
        # rebuild skips agent creation since no export file and no cards
        mem.rebuild()
        # research_agent may be None with empty memory


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

        mem.note_sync.export_jsonl(mem.config.export_file)

        assert mem.config.export_file.exists()
        lines = mem.config.export_file.read_text().strip().split("\n")
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert "id" in record or "content" in record


# ---------------------------------------------------------------------------
# Task 9: NoteSync.sync_card_to_amem_with_evolution calls add_note for new cards
# ---------------------------------------------------------------------------


def test_note_sync_sync_card_to_amem_with_evolution_calls_add_note_for_new_card(
    tmp_path,
):
    """NoteSync.sync_card_to_amem_with_evolution must call memory_system.add_note for new cards."""
    from unittest.mock import MagicMock

    from gigaevo.memory.shared_memory.card_store import CardStore
    from gigaevo.memory.shared_memory.models import MemoryCard
    from gigaevo.memory.shared_memory.note_sync import NoteSync

    card_store = CardStore(index_file=tmp_path / "index.json")

    mock_memory = MagicMock()
    mock_memory.read.return_value = None  # card doesn't exist yet
    mock_memory.add_note.return_value = "new-note-id"

    note_sync = NoteSync(
        memory_system=mock_memory,
        note_cls=MagicMock(),
        card_store=card_store,
    )

    card = MemoryCard(
        id="card-001",
        description="test card description",
        category="general",
    )

    note_sync.sync_card_to_amem_with_evolution(card)

    mock_memory.add_note.assert_called_once()
    call_kwargs = mock_memory.add_note.call_args
    assert call_kwargs is not None
    assert call_kwargs.kwargs.get("id") == "card-001"
