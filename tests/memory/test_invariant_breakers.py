"""Adversarial tests targeting CardStore bidirectional map invariants
and persistence/in-memory divergence.

These tests try to BREAK the memory system by exercising fragile code paths.
Tests documenting real bugs use the TestBug* naming convention from
test_edge_cases.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_store import CardStore
from tests.fakes.agentic_memory import (
    make_test_memory,
    make_test_memory_with_agentic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> CardStore:
    return CardStore(index_file=tmp_path / "api_index.json")


def _check_bimap_invariant(store: CardStore) -> None:
    """Assert the bidirectional map invariant holds.

    For every (card_id → entity_id) in entity_by_card_id,
    card_id_by_entity[entity_id] must equal card_id, and vice versa.
    """
    for card_id, entity_id in store.entity_by_card_id.items():
        assert store.card_id_by_entity.get(entity_id) == card_id, (
            f"Forward maps card_id={card_id!r} → entity_id={entity_id!r}, "
            f"but reverse maps entity_id → {store.card_id_by_entity.get(entity_id)!r}"
        )
    for entity_id, card_id in store.card_id_by_entity.items():
        assert store.entity_by_card_id.get(card_id) == entity_id, (
            f"Reverse maps entity_id={entity_id!r} → card_id={card_id!r}, "
            f"but forward maps card_id → {store.entity_by_card_id.get(card_id)!r}"
        )
    assert len(store.entity_by_card_id) == len(store.card_id_by_entity)


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# Category A: Bidirectional Map Invariant (card_store.py)
# ===========================================================================


class TestBimapInvariant:
    """CardStore maintains entity_by_card_id ↔ card_id_by_entity in sync."""

    def test_save_entity_cleans_old_entity_mapping(self, tmp_path):
        """A1: save_entity(card-A, entity-2) when card-A was linked to entity-1
        must remove entity-1 from reverse map."""
        store = _make_store(tmp_path)
        store.link_entity("card-A", "entity-1", "v1")
        store.save_entity("card-A", "entity-2", "v2")

        assert store.entity_by_card_id["card-A"] == "entity-2"
        assert store.card_id_by_entity["entity-2"] == "card-A"
        assert "entity-1" not in store.card_id_by_entity
        assert "entity-1" not in store.entity_version
        _check_bimap_invariant(store)

    def test_save_entity_reuse_entity_across_cards_preserves_invariant(self, tmp_path):
        """A2: save_entity("card-B", "entity-1") when entity-1 was linked to
        card-A cleans up card-A's forward mapping, preserving the invariant."""
        store = _make_store(tmp_path)
        store.link_entity("card-A", "entity-1", "v1")

        # Move entity-1 to card-B
        store.save_entity("card-B", "entity-1", "v2")

        # Reverse map correctly points to card-B
        assert store.card_id_by_entity["entity-1"] == "card-B"
        assert store.entity_by_card_id["card-B"] == "entity-1"

        # card-A's forward mapping was cleaned up
        assert "card-A" not in store.entity_by_card_id
        _check_bimap_invariant(store)

    def test_invariant_after_mixed_operations(self, tmp_path):
        """A3: After a sequence of mixed link/unlink/save/clear operations,
        the invariant must hold."""
        store = _make_store(tmp_path)

        store.link_entity("c1", "e1", "v1")
        store.link_entity("c2", "e2", "v1")
        store.link_entity("c3", "e3", "v1")
        _check_bimap_invariant(store)

        store.save_entity("c1", "e4", "v2")  # c1: e1→e4
        _check_bimap_invariant(store)

        store.unlink_entity("e2")  # remove c2↔e2
        _check_bimap_invariant(store)

        store.clear_entity("c3")  # remove c3↔e3
        _check_bimap_invariant(store)

        store.link_entity("c5", "e5", "v1")
        store.save_entity("c5", "e6", "v2")  # c5: e5→e6
        _check_bimap_invariant(store)

        store.clear_entity("c1")
        store.unlink_entity("e6")
        _check_bimap_invariant(store)

        assert len(store.entity_by_card_id) == 0
        assert len(store.card_id_by_entity) == 0

    def test_clear_entity_on_unlinked_card_noop(self, tmp_path):
        """A4: clear_entity on a card with no entity mapping is a no-op."""
        store = _make_store(tmp_path)
        store.link_entity("card-A", "entity-1")

        result = store.clear_entity("card-B")
        assert result is None
        assert store.entity_by_card_id["card-A"] == "entity-1"
        _check_bimap_invariant(store)

    def test_unlink_nonexistent_entity_noop(self, tmp_path):
        """A5: unlink_entity for a nonexistent entity returns None safely."""
        store = _make_store(tmp_path)
        result = store.unlink_entity("ghost")
        assert result is None
        assert len(store.entity_by_card_id) == 0
        assert len(store.card_id_by_entity) == 0

    def test_persist_reload_preserves_bimap(self, tmp_path):
        """A6: After persist + reload, card_id_by_entity is correctly
        reconstructed from entity_by_card_id in the index file."""
        store1 = _make_store(tmp_path)
        store1.cards["c1"] = normalize_memory_card({"id": "c1", "description": "a"})
        store1.cards["c2"] = normalize_memory_card({"id": "c2", "description": "b"})
        store1.link_entity("c1", "e1", "v1")
        store1.link_entity("c2", "e2", "v2")
        store1.persist()

        store2 = _make_store(tmp_path)
        assert store2.entity_by_card_id == {"c1": "e1", "c2": "e2"}
        assert store2.card_id_by_entity == {"e1": "c1", "e2": "c2"}
        assert store2.entity_version == {"e1": "v1", "e2": "v2"}
        _check_bimap_invariant(store2)

    def test_load_index_skips_dangling_entity_mapping(self, tmp_path):
        """A7: If api_index.json has an entity_by_card_id entry for a
        card_id that doesn't exist in memory_cards, the mapping is skipped."""
        index_file = tmp_path / "api_index.json"
        index_data = {
            "memory_cards": {},
            "entity_by_card_id": {"ghost-card": "entity-1"},
            "entity_version_by_entity": {"entity-1": "v1"},
        }
        index_file.write_text(json.dumps(index_data))

        store = CardStore(index_file=index_file)

        # Dangling mapping was skipped during load
        assert "ghost-card" not in store.cards
        assert "ghost-card" not in store.entity_by_card_id
        assert "entity-1" not in store.card_id_by_entity
        _check_bimap_invariant(store)

    def test_load_index_skips_orphaned_entity_version(self, tmp_path):
        """X1: entity_version entries for entities that have no card mapping
        are skipped during load, preventing stale version matches in sync."""
        index_file = tmp_path / "api_index.json"
        index_data = {
            "memory_cards": {"real-card": {"id": "real-card", "description": "ok"}},
            "entity_by_card_id": {"real-card": "e-real"},
            "entity_version_by_entity": {"e-real": "v1", "e-orphan": "v99"},
        }
        index_file.write_text(json.dumps(index_data))

        store = CardStore(index_file=index_file)

        # Real entity version loaded
        assert store.entity_version.get("e-real") == "v1"
        # Orphaned entity version skipped
        assert "e-orphan" not in store.entity_version
        _check_bimap_invariant(store)


# ===========================================================================
# Category B: Persistence / In-Memory Divergence (memory.py)
# ===========================================================================


class TestPersistenceDivergence:
    """Tests where in-memory state diverges from persisted state."""

    def test_api_sync_save_raises_card_not_in_memory(self, tmp_path):
        """B1: If save_card_to_api raises, the card is NOT stored in
        store.cards because the dict mutation (line 228) comes AFTER
        the sync call (line 225)."""
        mem = _make_memory(tmp_path)
        mock_api = MagicMock()
        mem.api = mock_api

        mock_sync = MagicMock()
        mock_sync.save_card_to_api.side_effect = RuntimeError("API error")
        mem.api_sync = mock_sync

        card = normalize_memory_card({"id": "test-card", "description": "test"})
        with pytest.raises(RuntimeError, match="API error"):
            mem._save_card_core(card)

        # Card is NOT in memory — the exception interrupted before line 228
        assert "test-card" not in mem.card_store.cards

    def test_note_sync_raises_card_in_memory_not_in_vector(self, tmp_path):
        """B2: If upsert_agentic raises AFTER store.cards is updated,
        the card is in memory but NOT in the vector store."""
        mem, _ = make_test_memory_with_agentic(tmp_path)
        assert mem.note_sync is not None

        def failing_upsert(card):
            raise RuntimeError("Chroma write failed")

        mem.note_sync.upsert_agentic = failing_upsert

        card = normalize_memory_card({"id": "orphan", "description": "test"})
        with pytest.raises(RuntimeError, match="Chroma write failed"):
            mem._save_card_core(card)

        # Card IS in memory (line 228 executed before line 231)
        assert "orphan" in mem.card_store.cards
        # But NOT in the vector store
        assert "orphan" not in mem.note_sync.memory_system.memories

    def test_rebuild_triggered_skips_redundant_persist(self, tmp_path):
        """B3: When _save_card_core triggers periodic rebuild, _save_and_persist
        correctly skips the extra persist() call."""
        mem = _make_memory(tmp_path, rebuild_interval=1)

        with patch.object(
            mem.card_store, "persist", wraps=mem.card_store.persist
        ) as spy:
            mem._save_and_persist(
                normalize_memory_card({"id": "c1", "description": "test"})
            )
            # rebuild() calls persist once, _save_and_persist should NOT call again
            # persist is called inside rebuild() via card_store.persist(serialized=...)
            assert spy.call_count == 1

    def test_context_manager_persists_dirty_state_on_exit(self, tmp_path):
        """B4: The __exit__ method calls rebuild when _iters_after_rebuild > 0,
        ensuring dirty state is persisted even if no explicit persist was called."""
        with _make_memory(tmp_path, rebuild_interval=9999) as mem:
            mem.save_card({"id": "c1", "description": "dirty card"})

        # Reload from disk — the card must be present
        mem2 = _make_memory(tmp_path)
        assert mem2.get_card("c1") is not None
        assert mem2.get_card("c1").description == "dirty card"

    def test_context_manager_swallows_rebuild_exception_still_closes(self, tmp_path):
        """B5: If rebuild() raises in __exit__, the exception is swallowed
        and close() is still called."""
        mock_api = MagicMock()

        with _make_memory(tmp_path, rebuild_interval=9999) as mem:
            mem.api = mock_api
            mem._iters_after_rebuild = 5

            # Make rebuild raise
            mem.rebuild = MagicMock(side_effect=RuntimeError("rebuild failed"))

        # close() was called (which calls api.close())
        mock_api.close.assert_called_once()
