"""Tests for the bug fixes in the memory system refactor.

Covers:
1. State mutation fix: _insert_new_card uses model_copy, not direct mutation
2. Namespace filtering fix: api_sync excludes None/empty namespace rows when namespace is set
3. gam rebuild failure: rebuild() clears research_agent and sets the failure flag
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.exceptions import MemoryRetrieverError
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_store import CardStore
from tests.fakes.agentic_memory import (
    make_test_memory,
    make_test_memory_with_agentic,
)

# ---------------------------------------------------------------------------
# 1. State mutation fix
# ---------------------------------------------------------------------------


class TestSaveCardCoreDoesNotMutateInput:
    """_insert_new_card must not mutate the caller's card object."""

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

        # Original keywords list must not have been swapped out by _insert_new_card
        assert id(original.keywords) == original_keywords_id, (
            "_insert_new_card mutated original.keywords in-place"
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
# 3. gam rebuild failure handling
# ---------------------------------------------------------------------------


class TestGamSearchInvalidateOnBuildFailure:
    """rebuild() must clear research_agent and set the failure flag on build failure."""

    def test_rebuild_clears_research_agent_on_build_failure(self, tmp_path):
        """After a failed rebuild, research_agent must be None."""
        mem, _ = make_test_memory_with_agentic(tmp_path)

        mock_gam = MagicMock()
        mock_gam.build_research_agent.side_effect = MemoryRetrieverError(
            "store missing"
        )
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
        mock_gam.build_research_agent.side_effect = MemoryRetrieverError("unavailable")
        mock_gam.agent = None
        mem.gam = mock_gam

        mem.rebuild()

        assert mem._gam_build_failed is True

    def test_rebuild_sets_agent_on_success(self, tmp_path):
        """When build_research_agent succeeds, research_agent is set and _gam_build_failed is False."""
        mem, _ = make_test_memory_with_agentic(tmp_path)

        mock_agent = MagicMock()
        mock_gam = MagicMock()
        mock_gam.build_research_agent.return_value = None
        mock_gam.agent = mock_agent
        mem.gam = mock_gam

        mem.rebuild()

        assert mem.research_agent is mock_agent
        assert mem._gam_build_failed is False


# ---------------------------------------------------------------------------
# 6. parse_string_list helper tests
# ---------------------------------------------------------------------------


def test_parse_string_list_from_list():
    from gigaevo.memory.utils import parse_string_list

    assert parse_string_list(["a", "b"]) == ["a", "b"]


def test_parse_string_list_from_json_string():
    from gigaevo.memory.utils import parse_string_list

    assert parse_string_list('["x", "y"]') == ["x", "y"]


def test_parse_string_list_from_ast_string():
    from gigaevo.memory.utils import parse_string_list

    assert parse_string_list("['x', 'y']") == ["x", "y"]


def test_parse_string_list_bare_string():
    from gigaevo.memory.utils import parse_string_list

    assert parse_string_list("hello") == ["hello"]


def test_parse_string_list_empty():
    from gigaevo.memory.utils import parse_string_list

    assert parse_string_list("") == []
    assert parse_string_list(None) == []
    assert parse_string_list([]) == []


def test_parse_string_list_with_whitespace():
    from gigaevo.memory.utils import parse_string_list

    assert parse_string_list(["  a  ", "b"]) == ["a", "b"]
    assert parse_string_list('[ "x" , "y" ]') == ["x", "y"]


# ---------------------------------------------------------------------------
# 7. Task-summary LLM is condensed once per run, then memoised
# ---------------------------------------------------------------------------


def test_task_summary_llm_is_memoised_across_increments():
    """The one-line task summary is condensed by the LLM at most once per run:
    repeated write sweeps reuse the cached summary rather than re-billing the
    memory model."""
    import asyncio

    from gigaevo.llm.agents.task_summary import TaskSummaryResponse
    from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

    class _CountingStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages):  # noqa: ANN001
            self.calls += 1
            return TaskSummaryResponse(summary="condensed")

    class _CountingLLM:
        def __init__(self) -> None:
            self.structured = _CountingStructured()

        def with_structured_output(self, schema, **kw):  # noqa: ANN001
            return self.structured

    llm = _CountingLLM()
    tracker = IdeaTracker(
        llm=llm,
        memory_write_enabled=False,
        task_description="Maximise the minimum triangle area over N points.",
    )

    asyncio.run(tracker._ensure_task_summary())
    asyncio.run(tracker._ensure_task_summary())

    assert tracker._task_description_summary == "condensed"
    assert llm.structured.calls == 1


# ---------------------------------------------------------------------------
# E2E Tests for memory integration
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
