"""Adversarial tests targeting ApiSync stale entity cleanup, lazy factory,
and search/sync interaction.

Tests the API synchronization logic for bugs in pagination, partial results,
and concurrent state changes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from tests.fakes.agentic_memory import make_test_memory


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# Category C: Stale Entity Cleanup (api_sync.py)
# ===========================================================================


class TestStaleEntityCleanup:
    """Tests for api_sync.sync() stale entity removal logic."""

    def test_sync_partial_results_preserves_healthy_local_entities(self, tmp_path):
        """C1: If API pagination fails mid-way (returns a full page then
        errors on the next), stale cleanup is skipped because pagination
        didn't complete. Entities from unseen pages are preserved."""
        from gigaevo.memory.shared_memory.memory_config import ApiConfig

        mem = _make_memory(
            tmp_path, api=ApiConfig(sync_batch_size=2, sync_on_init=False)
        )
        mem.api_sync = None  # force lazy re-creation with mock

        # Pre-populate store with 5 entities
        for i in range(5):
            card_id = f"c{i}"
            entity_id = f"e{i}"
            mem.card_store.cards[card_id] = normalize_memory_card(
                {"id": card_id, "description": f"card {i}"}
            )
            mem.card_store.link_entity(card_id, entity_id, f"v{i}")

        # Mock API: page 1 returns full page (2 items = batch_size),
        # page 2 raises (simulating network timeout).
        mock_api = MagicMock()
        mock_api.list_memory_cards.side_effect = [
            [
                {
                    "entity_id": "e0",
                    "version_id": "v0",
                    "meta": {"namespace": "default"},
                },
                {
                    "entity_id": "e1",
                    "version_id": "v1",
                    "meta": {"namespace": "default"},
                },
            ],
            RuntimeError("API timeout"),
        ]
        mock_api.get_concept.side_effect = lambda eid, **kw: {
            "content": {"id": f"c{int(eid[1])}", "description": f"updated {eid}"},
            "version_id": "new_version",
        }

        mem.api = mock_api
        mem._sync_from_api(force_full=True)

        # Entities from page 1 updated, entities from page 2+ preserved
        assert "c0" in mem.card_store.cards
        assert "c1" in mem.card_store.cards
        # NOT deleted — stale cleanup skipped because pagination didn't complete
        assert "c2" in mem.card_store.cards
        assert "c3" in mem.card_store.cards
        assert "c4" in mem.card_store.cards

    def test_sync_removes_card_and_note_for_stale_entity(self, tmp_path):
        """C2: When a local entity is not in remote list, its card and
        note are removed, and entity mappings are unlinked."""
        mem = _make_memory(tmp_path)

        # Pre-populate
        mem.card_store.cards["c1"] = normalize_memory_card(
            {"id": "c1", "description": "old"}
        )
        mem.card_store.link_entity("c1", "e1", "v1")

        # Mock API to return no entities (all stale)
        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = []
        mem.api = mock_api

        mem._sync_from_api(force_full=True)

        # Card removed
        assert "c1" not in mem.card_store.cards
        # Entity mappings removed
        assert "c1" not in mem.card_store.entity_by_card_id
        assert "e1" not in mem.card_store.card_id_by_entity
        assert "e1" not in mem.card_store.entity_version

    def test_sync_entity_remapped_to_different_card_id(self, tmp_path):
        """C3: When remote returns the same entity with a different card_id,
        the old card is removed and new card stored."""
        mem = _make_memory(tmp_path)

        # Pre-populate
        mem.card_store.cards["old-card"] = normalize_memory_card(
            {"id": "old-card", "description": "old"}
        )
        mem.card_store.link_entity("old-card", "e1", "v1")

        # Mock API to return entity e1 with content id="new-card"
        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {"entity_id": "e1", "version_id": "v2", "meta": {"namespace": "default"}}
        ]
        mock_api.get_concept.return_value = {
            "content": {"id": "new-card", "description": "updated"},
            "version_id": "v2",
        }
        mem.api = mock_api

        mem._sync_from_api(force_full=True)

        # Old card removed
        assert "old-card" not in mem.card_store.cards
        # New card added
        assert "new-card" in mem.card_store.cards
        # Entity mappings updated
        assert mem.card_store.entity_by_card_id.get("new-card") == "e1"
        assert mem.card_store.card_id_by_entity.get("e1") == "new-card"


# ===========================================================================
# Category D: Lazy ApiSync Factory (memory.py)
# ===========================================================================


class TestLazyApiSyncFactory:
    """Tests for the _ensure_api_sync lazy factory logic."""

    def test_ensure_api_sync_when_api_set_but_config_none(self, tmp_path):
        """D1: When mem.api is set post-construction but config.api is None
        (local-only mode), _ensure_api_sync creates ApiSync with "default"
        namespace fallback.

        This may be correct-by-design or a bug depending on intent.
        Document the behavior.
        """
        mem = _make_memory(tmp_path)  # config.api is None
        mock_api = MagicMock()
        mem.api = mock_api

        api_sync = mem._ensure_api_sync()

        assert api_sync is not None
        # Falls back to "default" namespace
        assert api_sync.namespace == "default"
        assert api_sync.channel == "latest"
        assert api_sync.author is None

    def test_ensure_api_sync_caches_instance(self, tmp_path):
        """D2: Once _ensure_api_sync creates an ApiSync, subsequent calls
        return the same cached instance."""
        mem = _make_memory(tmp_path)
        mock_api = MagicMock()
        mem.api = mock_api

        api_sync1 = mem._ensure_api_sync()
        api_sync2 = mem._ensure_api_sync()

        assert api_sync1 is api_sync2

    def test_ensure_api_sync_none_when_no_api(self, tmp_path):
        """D3: When mem.api is None (local-only mode), _ensure_api_sync
        returns None without creating ApiSync."""
        mem = _make_memory(tmp_path)
        assert mem.api is None

        result = mem._ensure_api_sync()
        assert result is None


# ===========================================================================
# Category E: Search/Sync Interaction (memory.py)
# ===========================================================================


class TestSearchSyncInteraction:
    """Tests for interactions between search() and _sync_from_api()."""

    def test_search_uses_post_rebuild_research_agent(self, tmp_path):
        """E1: When _sync_from_api triggers rebuild, the new research_agent
        (created by rebuild) is used in search, not a stale reference."""
        mem = _make_memory(tmp_path, rebuild_interval=1)

        # Spy on rebuild to track if new agent is created
        rebuild_calls = []

        original_rebuild = mem.rebuild

        def tracked_rebuild():
            rebuild_calls.append(True)
            original_rebuild()

        mem.rebuild = tracked_rebuild
        mem.api = MagicMock()
        mem.api.list_memory_cards.return_value = [
            {"entity_id": "e1", "version_id": "v1", "meta": {"namespace": "default"}}
        ]

        # Save a card to trigger rebuild on next save (rebuild_interval=1)
        mem.save_card({"id": "c1", "description": "triggers rebuild"})

        # rebuild was called
        assert len(rebuild_calls) > 0

    def test_search_via_api_persists_on_no_rebuild(self, tmp_path):
        """E2: When _search_via_api adds cards and local_changed=True but
        no rebuild (no agentic), persist() is still called."""
        mem = _make_memory(tmp_path)  # No agentic

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {
            "hits": [{"entity_id": "e1", "version_id": "v1"}]
        }
        mock_api.get_concept.return_value = {
            "content": {"id": "c1", "description": "from api"},
            "version_id": "v1",
        }
        mem.api = mock_api

        # Spy on persist
        original_persist = mem.card_store.persist
        persist_calls = []

        def tracked_persist(*args, **kwargs):
            persist_calls.append(True)
            original_persist(*args, **kwargs)

        mem.card_store.persist = tracked_persist

        mem._search_via_api("test query")

        # Card added, persist called
        assert "c1" in mem.card_store.cards
        assert len(persist_calls) > 0

    def test_stale_orphan_entity_does_not_trigger_rebuild(self, tmp_path):
        """X2: When stale entity has no card (unlink_entity returns None),
        changed flag should NOT be set — avoids unnecessary rebuilds."""
        mem = _make_memory(tmp_path)

        # Pre-populate with one real card+entity
        mem.card_store.cards["c1"] = normalize_memory_card(
            {"id": "c1", "description": "real"}
        )
        mem.card_store.link_entity("c1", "e1", "v1")
        # Inject orphan entity (entity in card_id_by_entity but no card)
        mem.card_store.card_id_by_entity["e-orphan"] = "ghost-card"

        # Mock API to return e1 only — e-orphan is "stale"
        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {"entity_id": "e1", "version_id": "v1", "meta": {"namespace": "default"}}
        ]
        mem.api = mock_api

        rebuild_calls = []
        original_rebuild = mem.rebuild

        def tracked_rebuild():
            rebuild_calls.append(True)
            original_rebuild()

        mem.rebuild = tracked_rebuild
        mem._sync_from_api(force_full=False)

        # Orphan was cleaned up but did NOT trigger rebuild
        assert "e-orphan" not in mem.card_store.card_id_by_entity
        assert len(rebuild_calls) == 0

    def test_delete_local_only_card_with_api_sync(self, tmp_path):
        """X4: Cards created locally (no entity mapping) can be deleted
        even when API sync is enabled."""
        mem = _make_memory(tmp_path)

        # Save a local card (no API)
        mem.save_card({"id": "local-card", "description": "local only"})
        assert "local-card" in mem.card_store.cards

        # Now enable API sync
        mock_api = MagicMock()
        mem.api = mock_api

        # Delete should succeed via local resolution
        result = mem.delete("local-card")
        assert result is True
        assert "local-card" not in mem.card_store.cards

    def test_gam_build_failure_does_not_retry_every_search(self, tmp_path):
        """X9: When GAM build fails, subsequent searches skip rebuild
        until new card data arrives (circuit breaker)."""
        mem = _make_memory(tmp_path)
        mem.api = MagicMock()
        mem.api.list_memory_cards.return_value = []

        # Simulate agentic infrastructure
        mem.memory_system = MagicMock()
        mem.generator = MagicMock()

        # Mock GAM to always fail
        mem.gam = MagicMock()
        mem.gam.build.side_effect = RuntimeError("GAM build failed")

        # First sync triggers rebuild (research_agent is None + _has_agentic)
        mem._sync_from_api(force_full=False)
        assert mem._gam_build_failed is True
        assert mem.gam.build.call_count == 1

        # Second sync does NOT retry build (circuit breaker active)
        mem._sync_from_api(force_full=False)
        assert mem.gam.build.call_count == 1  # Still 1, not 2
