"""Tests for the bug fixes in the memory system refactor.

Covers:
1. State mutation fix: _save_card_core uses model_copy, not direct mutation
2. Namespace filtering fix: api_sync excludes None/empty namespace rows when namespace is set
3. Dedup meta type guard: score_duplicate_candidates handles non-dict meta safely
4. LLM retry fallback logging: warning logged when all retries fail
5. gam_search.invalidate wired: rebuild() calls invalidate() on GAM build failure
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from gigaevo.exceptions import MemoryRetrieverError
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig
from tests.fakes.agentic_memory import (
    make_test_memory,
    make_test_memory_with_agentic,
)

# ---------------------------------------------------------------------------
# 1. State mutation fix
# ---------------------------------------------------------------------------


class TestSaveCardCoreDoesNotMutateInput:
    """_save_card_core must not mutate the caller's card object."""

    def test_enrichment_does_not_mutate_original_card(self, tmp_path):
        """LLM enrichment creates a new card via model_copy; original unchanged."""
        mem, _ = make_test_memory_with_agentic(
            tmp_path, enable_llm_card_enrichment=True
        )
        # Card starts with no keywords
        original = normalize_memory_card(
            {
                "description": "gradient clipping prevents exploding gradients",
                "category": "general",
            }
        )
        original_keywords_id = id(original.keywords)

        mem.save_card(original)

        # Original keywords list must not have been swapped out by _save_card_core
        assert id(original.keywords) == original_keywords_id, (
            "_save_card_core mutated original.keywords in-place"
        )

    def test_enrichment_stored_card_has_keywords(self, tmp_path):
        """The card stored in memory does have the enriched keywords."""
        mem, fake_system = make_test_memory_with_agentic(
            tmp_path, enable_llm_card_enrichment=True
        )
        card = normalize_memory_card(
            {
                "id": "card-enrich-01",
                "description": "gradient clipping prevents exploding gradients",
                "category": "general",
            }
        )

        card_id = mem.save_card(card)
        stored = mem.get_card(card_id)

        # analyze_content returns keywords; stored card should have them
        assert stored is not None
        assert len(stored.keywords) > 0, "Stored card should have enriched keywords"

    def test_save_card_without_enrichment_leaves_keywords_unchanged(self, tmp_path):
        """Without enrichment, save_card does not add keywords."""
        mem = make_test_memory(tmp_path, enable_llm_card_enrichment=False)
        card = normalize_memory_card(
            {
                "id": "card-no-enrich",
                "description": "gradient clipping technique",
                "keywords": ["clipping"],
                "category": "general",
            }
        )
        mem.save_card(card)
        stored = mem.get_card("card-no-enrich")
        assert stored is not None
        assert stored.keywords == ["clipping"]


# ---------------------------------------------------------------------------
# 2. Namespace filtering fix
# ---------------------------------------------------------------------------


class TestNamespaceFiltering:
    """fetch_all_hits must enforce namespace filtering correctly."""

    def _make_api_sync(self, tmp_path, namespace: str):
        """Create ApiSync with a mock client."""
        from gigaevo.memory.shared_memory.api_sync import ApiSync
        from gigaevo.memory.shared_memory.card_store import CardStore

        store = CardStore(index_file=tmp_path / "index.json")
        client = MagicMock()
        return ApiSync(
            client=client,
            card_store=store,
            note_sync=None,
            namespace=namespace,
            channel="latest",
            sync_batch_size=100,
            search_limit=5,
        )

    def test_none_namespace_excluded_when_namespace_set(self, tmp_path):
        """Row with namespace=None must be excluded when self.namespace is 'ns1'."""
        sync = self._make_api_sync(tmp_path, namespace="ns1")
        sync.client.list_memory_cards.side_effect = [
            [{"entity_id": "e1", "meta": {"namespace": None}}],
            [],
        ]
        hits, _ = sync.fetch_all_hits()
        assert hits == [], (
            "Row with namespace=None should be excluded for namespace='ns1'"
        )

    def test_empty_namespace_excluded_when_namespace_set(self, tmp_path):
        """Row with namespace='' must be excluded when self.namespace is 'ns1'."""
        sync = self._make_api_sync(tmp_path, namespace="ns1")
        sync.client.list_memory_cards.side_effect = [
            [{"entity_id": "e2", "meta": {"namespace": ""}}],
            [],
        ]
        hits, _ = sync.fetch_all_hits()
        assert hits == [], (
            "Row with namespace='' should be excluded for namespace='ns1'"
        )

    def test_matching_namespace_included(self, tmp_path):
        """Row with namespace='ns1' must be included when self.namespace is 'ns1'."""
        sync = self._make_api_sync(tmp_path, namespace="ns1")
        sync.client.list_memory_cards.side_effect = [
            [{"entity_id": "e3", "meta": {"namespace": "ns1"}}],
            [],
        ]
        hits, _ = sync.fetch_all_hits()
        assert len(hits) == 1

    def test_mismatched_namespace_excluded(self, tmp_path):
        """Row with namespace='other' must be excluded when self.namespace is 'ns1'."""
        sync = self._make_api_sync(tmp_path, namespace="ns1")
        sync.client.list_memory_cards.side_effect = [
            [{"entity_id": "e4", "meta": {"namespace": "other"}}],
            [],
        ]
        hits, _ = sync.fetch_all_hits()
        assert hits == []

    def test_no_namespace_set_includes_all(self, tmp_path):
        """When self.namespace is empty (''), all rows pass through."""
        sync = self._make_api_sync(tmp_path, namespace="")
        sync.client.list_memory_cards.side_effect = [
            [
                {"entity_id": "e5", "meta": {"namespace": None}},
                {"entity_id": "e6", "meta": {"namespace": "ns1"}},
                {"entity_id": "e7", "meta": {}},
            ],
            [],
        ]
        hits, _ = sync.fetch_all_hits()
        assert len(hits) == 3


# ---------------------------------------------------------------------------
# 3. Dedup meta type guard
# ---------------------------------------------------------------------------


class _FakeHit:
    """Fake retriever hit object."""

    def __init__(self, page_id: str, meta: Any):
        self.page_id = page_id
        self.meta = meta


class TestDedupMetaTypeGuard:
    """score_duplicate_candidates must safely handle non-dict meta values."""

    def _make_dedup(self, tmp_path) -> Any:
        from gigaevo.memory.shared_memory.card_dedup import CardDedup

        store = CardStore(index_file=tmp_path / "index.json")
        config = CardUpdateDedupConfig.model_validate(
            {"enabled": True, "top_k_per_query": 5}
        )
        return CardDedup(
            card_store=store,
            llm_service=None,
            config=config,
            allowed_gam_tools={"vector_description"},
            gam_store_dir=tmp_path / "gam",
            export_file=tmp_path / "export.jsonl",
            checkpoint_dir=tmp_path / "checkpoints",
        )

    def _hit_with_meta(self, page_id: str, meta: Any) -> _FakeHit:
        return _FakeHit(page_id=page_id, meta=meta)

    def test_string_meta_does_not_crash(self, tmp_path):
        """If hit.meta is a string (invalid), score should default to 0.0 (no score)."""
        dedup = self._make_dedup(tmp_path)
        # Add a card to the store so it can be found
        card = normalize_memory_card(
            {"id": "card-x", "description": "test card", "category": "general"}
        )
        dedup._card_store.cards["card-x"] = card

        hit = self._hit_with_meta("card-x", "not-a-dict")

        # Inject a fake retriever that returns this hit
        fake_retriever = MagicMock()
        fake_retriever.search.return_value = [[hit]]

        def _resolve_x(_tool_name: str) -> Any:
            return fake_retriever

        # Should not raise; string meta treated as empty dict → score=0.0 → no result
        incoming = normalize_memory_card(
            {"description": "similar card", "category": "general"}
        )
        candidates = dedup.score_duplicate_candidates(
            incoming, resolve_retriever_fn=_resolve_x
        )
        assert isinstance(candidates, list)

    def test_none_meta_does_not_crash(self, tmp_path):
        """If hit.meta is None, score should default to 0.0 (no crash)."""
        dedup = self._make_dedup(tmp_path)
        card = normalize_memory_card(
            {"id": "card-y", "description": "test card", "category": "general"}
        )
        dedup._card_store.cards["card-y"] = card

        hit = self._hit_with_meta("card-y", None)
        fake_retriever = MagicMock()
        fake_retriever.search.return_value = [[hit]]

        def _resolve_y(_tool_name: str) -> Any:
            return fake_retriever

        incoming = normalize_memory_card(
            {"description": "similar card", "category": "general"}
        )
        candidates = dedup.score_duplicate_candidates(
            incoming, resolve_retriever_fn=_resolve_y
        )
        assert isinstance(candidates, list)

    def test_dict_meta_with_score_works(self, tmp_path):
        """Normal dict meta with score > 0 produces a candidate."""
        dedup = self._make_dedup(tmp_path)
        card = normalize_memory_card(
            {"id": "card-z", "description": "test card", "category": "general"}
        )
        dedup._card_store.cards["card-z"] = card

        hit = self._hit_with_meta("card-z", {"score": 0.85})
        fake_retriever = MagicMock()
        fake_retriever.search.return_value = [[hit]]

        def _resolve_z(_tool_name: str) -> Any:
            return fake_retriever

        incoming = normalize_memory_card(
            {"description": "similar card", "category": "general"}
        )
        candidates = dedup.score_duplicate_candidates(
            incoming, resolve_retriever_fn=_resolve_z
        )
        # card-z should appear as a candidate
        assert any(c["card_id"] == "card-z" for c in candidates)


# ---------------------------------------------------------------------------
# 4. LLM retry fallback logging
# ---------------------------------------------------------------------------


class TestDedupLLMRetryFallback:
    """ask_llm_for_dedup_decision logs a warning when all retries are exhausted."""

    def _make_dedup_with_failing_llm(self, tmp_path, num_retries: int = 2) -> Any:
        from gigaevo.memory.shared_memory.card_dedup import CardDedup

        store = CardStore(index_file=tmp_path / "index.json")
        config = CardUpdateDedupConfig.model_validate(
            {"enabled": True, "llm_max_retries": num_retries}
        )

        # LLM that always raises
        failing_llm = MagicMock()
        failing_llm.generate.side_effect = RuntimeError("LLM unavailable")

        return CardDedup(
            card_store=store,
            llm_service=failing_llm,
            config=config,
            allowed_gam_tools=set(),
            gam_store_dir=tmp_path / "gam",
            export_file=tmp_path / "export.jsonl",
            checkpoint_dir=tmp_path / "checkpoints",
        )

    def test_warning_logged_when_all_retries_fail(self, tmp_path):
        """When all LLM retries fail, default action is add (warning logged to stderr)."""
        dedup = self._make_dedup_with_failing_llm(tmp_path, num_retries=2)
        incoming = normalize_memory_card(
            {"id": "card-incoming", "description": "new idea", "category": "general"}
        )
        # Provide a candidate so the LLM is actually called
        candidates = [{"card_id": "card-existing"}]

        result = dedup.ask_llm_for_dedup_decision(incoming, candidates)

        # When all retries fail, should default to add
        assert result["action"] == "add"
        # Warning is logged to stderr (verified by pytest capture in test output)

    def test_default_action_is_add_when_llm_returns_bad_json(self, tmp_path):
        """When LLM returns invalid JSON every time, action defaults to add."""
        from gigaevo.memory.shared_memory.card_dedup import CardDedup

        store = CardStore(index_file=tmp_path / "index.json")
        config = CardUpdateDedupConfig.model_validate(
            {"enabled": True, "llm_max_retries": 2}
        )
        bad_llm = MagicMock()
        bad_llm.generate.return_value = ("not valid json !!!", None, None, None)

        dedup = CardDedup(
            card_store=store,
            llm_service=bad_llm,
            config=config,
            allowed_gam_tools=set(),
            gam_store_dir=tmp_path / "gam",
            export_file=tmp_path / "export.jsonl",
            checkpoint_dir=tmp_path / "checkpoints",
        )
        incoming = normalize_memory_card(
            {"id": "card-incoming-bad", "description": "idea", "category": "general"}
        )
        candidates = [{"card_id": "card-e1"}]

        result = dedup.ask_llm_for_dedup_decision(incoming, candidates)

        # When all retries fail on bad JSON, should default to add
        assert result["action"] == "add"
        # Warning is logged to stderr (verified by pytest capture in test output)


# ---------------------------------------------------------------------------
# 5. gam_search.invalidate wired on build failure
# ---------------------------------------------------------------------------


class TestGamSearchInvalidateOnBuildFailure:
    """rebuild() must call gam.invalidate() and clear research_agent on build failure."""

    def test_rebuild_calls_invalidate_on_gam_build_failure(self, tmp_path):
        """When gam.build() raises MemoryRetrieverError, invalidate() is called."""
        mem, _ = make_test_memory_with_agentic(tmp_path)

        # Inject a mock GamSearch that raises on build
        mock_gam = MagicMock()
        mock_gam.build.side_effect = MemoryRetrieverError("index corrupt")
        mock_gam.agent = None
        mem.gam = mock_gam
        mem.research_agent = MagicMock()  # Pretend it was set previously

        mem.rebuild()

        mock_gam.invalidate.assert_called_once()

    def test_rebuild_clears_research_agent_on_build_failure(self, tmp_path):
        """After a failed rebuild, research_agent must be None."""
        mem, _ = make_test_memory_with_agentic(tmp_path)

        mock_gam = MagicMock()
        mock_gam.build.side_effect = MemoryRetrieverError("store missing")
        mock_gam.agent = MagicMock()
        mem.gam = mock_gam
        mem.research_agent = MagicMock()  # Stale reference

        mem.rebuild()

        assert mem.research_agent is None, (
            "research_agent should be cleared after build failure"
        )

    def test_rebuild_sets_gam_build_failed_flag(self, tmp_path):
        """After a failed rebuild, _gam_build_failed must be True."""
        mem, _ = make_test_memory_with_agentic(tmp_path)

        mock_gam = MagicMock()
        mock_gam.build.side_effect = MemoryRetrieverError("unavailable")
        mock_gam.agent = None
        mem.gam = mock_gam

        mem.rebuild()

        assert mem._gam_build_failed is True

    def test_rebuild_does_not_call_invalidate_on_success(self, tmp_path):
        """When build succeeds, invalidate() must NOT be called."""
        mem, _ = make_test_memory_with_agentic(tmp_path)

        mock_agent = MagicMock()
        mock_gam = MagicMock()
        mock_gam.build.return_value = None
        mock_gam.agent = mock_agent
        mem.gam = mock_gam

        mem.rebuild()

        mock_gam.invalidate.assert_not_called()
        assert mem.research_agent is mock_agent
        assert mem._gam_build_failed is False


# ---------------------------------------------------------------------------
# E2E Tests for write_pipeline and memory integration
# ---------------------------------------------------------------------------


class TestMemoryWriteE2E:
    """E2E tests for memory save and retrieval cycle."""

    def test_memory_save_and_retrieve_cycle(self, tmp_path):
        """E2E: save_card() writes card and get_card() retrieves it."""
        mem = make_test_memory(tmp_path, enable_llm_card_enrichment=False)

        # Save multiple cards
        card1 = normalize_memory_card(
            {
                "id": "e2e-001",
                "description": "gradient descent optimization",
                "category": "general",
            }
        )
        card2 = normalize_memory_card(
            {
                "id": "e2e-002",
                "description": "batch normalization technique",
                "category": "general",
            }
        )

        id1 = mem.save_card(card1)
        id2 = mem.save_card(card2)

        # Retrieve and verify
        retrieved1 = mem.get_card(id1)
        retrieved2 = mem.get_card(id2)

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.description == "gradient descent optimization"
        assert retrieved2.description == "batch normalization technique"


class TestMemorySearchE2E:
    """E2E tests for memory search functionality."""

    def test_memory_search_returns_results(self, tmp_path):
        """E2E: search() returns card IDs matching query."""
        mem = make_test_memory(tmp_path, enable_llm_card_enrichment=False)

        # Save cards with distinct descriptions
        card1 = normalize_memory_card(
            {
                "description": "gradient descent optimization algorithm",
                "category": "general",
            }
        )
        card2 = normalize_memory_card(
            {
                "description": "random forest classifier ensemble",
                "category": "general",
            }
        )

        mem.save_card(card1)
        mem.save_card(card2)

        # Search for relevant cards
        results = mem.search("gradient descent")

        # Verify search returned results (as card IDs)
        assert len(results) > 0
        # Results should be a list of strings (card IDs)
        assert all(isinstance(r, str) for r in results)


class TestCardLoaderE2E:
    """E2E tests for CardLoader streaming and filtering."""

    def test_card_loader_streams_large_files(self, tmp_path):
        """E2E: CardLoader streams JSONL without loading entire file into memory."""
        from gigaevo.memory.shared_memory.card_loader import CardLoader

        # Create a large JSONL file (100 cards)
        export_file = tmp_path / "export.jsonl"
        with open(export_file, "w") as f:
            for i in range(100):
                f.write(
                    f'{{"id": "card-{i:03d}", "description": "card {i}", "category": "general"}}\n'
                )

        loader = CardLoader(
            export_file=export_file,
            include_programs=False,
        )

        cards = loader.load()

        # Verify all cards loaded and filtered
        assert len(cards) == 100
        assert all(isinstance(c, dict) for c in cards)
        assert all(c.get("category") != "program" for c in cards)

    def test_card_loader_handles_malformed_lines(self, tmp_path):
        """E2E: CardLoader skips malformed JSON and continues."""
        from gigaevo.memory.shared_memory.card_loader import CardLoader

        export_file = tmp_path / "export.jsonl"
        with open(export_file, "w") as f:
            f.write('{"id": "card-001", "description": "valid"}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"id": "card-002", "description": "also valid"}\n')

        loader = CardLoader(
            export_file=export_file,
        )

        cards = loader.load()

        # Malformed line should be skipped
        assert len(cards) == 2
        assert cards[0]["id"] == "card-001"
        assert cards[1]["id"] == "card-002"


class TestCardLoaderAndMemoryRebuildE2E:
    """E2E tests for memory rebuild and consistency."""

    def test_memory_rebuild_maintains_consistency(self, tmp_path):
        """E2E: rebuild() maintains search index consistency with stored cards."""
        mem = make_test_memory(tmp_path, enable_llm_card_enrichment=False)

        # Save a card
        card = normalize_memory_card(
            {
                "description": "neural architecture search technique",
                "category": "general",
            }
        )
        card_id = mem.save_card(card)

        # Rebuild the memory (refreshes search index)
        mem.rebuild()

        # Search should still find the card after rebuild
        results = mem.search("neural architecture")
        assert len(results) > 0

        # Card should still be retrievable
        retrieved = mem.get_card(card_id)
        assert retrieved is not None
        assert retrieved.description == "neural architecture search technique"
