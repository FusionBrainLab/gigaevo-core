"""Cycle 5: mutation operator memory flow, sync_from_api, API client body
verification, _build_entity_meta content, __del__ behavior.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.card_conversion import (
    build_entity_meta,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.memory_config import ApiConfig
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


def _mock_client(handler):
    client = _ConceptApiClient.__new__(_ConceptApiClient)
    client._http = httpx.Client(
        base_url="http://test:8000", transport=httpx.MockTransport(handler)
    )
    return client


# ===========================================================================
# 2. _sync_from_api with mocked API
# ===========================================================================


class TestSyncFromApi:
    """Test _sync_from_api paginated sync, version checking."""

    def test_sync_disabled_when_no_api(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem._sync_from_api(force_full=False) is False

    def test_sync_adds_new_cards(self, tmp_path):
        """Mocked API returns cards → they appear in card_store.cards."""
        mem = _make_memory(tmp_path)

        # Mock API
        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {
                "entity_id": "eid-1",
                "version_id": "v1",
                "meta": {"namespace": "default"},
            },
        ]
        mock_api.get_concept.return_value = {
            "content": {
                "id": "idea-1",
                "description": "annealing idea",
                "category": "general",
            },
            "version_id": "v1",
        }
        mem.api = mock_api

        result = mem._sync_from_api(force_full=True)
        assert result is True
        assert "idea-1" in mem.card_store.cards
        assert mem.card_store.entity_by_card_id.get("idea-1") == "eid-1"
        assert mem.card_store.card_id_by_entity.get("eid-1") == "idea-1"
        # Verify correct entity_id was passed to get_concept
        mock_api.get_concept.assert_called_once_with("eid-1", channel="latest")

    def test_sync_skips_unchanged_versions(self, tmp_path):
        """Cards with known version are skipped during incremental sync."""
        mem = _make_memory(tmp_path)

        # Pre-populate known state
        mem.card_store.cards["idea-1"] = normalize_memory_card(
            {"id": "idea-1", "description": "known"}
        )
        mem.card_store.entity_by_card_id["idea-1"] = "eid-1"
        mem.card_store.card_id_by_entity["eid-1"] = "idea-1"
        mem.card_store.entity_version["eid-1"] = "v1"

        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {
                "entity_id": "eid-1",
                "version_id": "v1",
                "meta": {"namespace": "default"},
            },
        ]
        mem.api = mock_api

        mem._sync_from_api(force_full=False)
        # get_concept should NOT be called — version unchanged
        mock_api.get_concept.assert_not_called()

    def test_sync_paginates(self, tmp_path):
        """API returns full pages → sync fetches next page."""
        mem = _make_memory(
            tmp_path, api=ApiConfig(sync_batch_size=2, sync_on_init=False)
        )
        mem.api_sync = None  # force lazy re-creation with mock

        mock_api = MagicMock()
        # Page 1: 2 items (full page → fetch more)
        # Page 2: 1 item (partial → stop)
        mock_api.list_memory_cards.side_effect = [
            [
                {
                    "entity_id": "e1",
                    "version_id": "v1",
                    "meta": {"namespace": "default"},
                },
                {
                    "entity_id": "e2",
                    "version_id": "v1",
                    "meta": {"namespace": "default"},
                },
            ],
            [
                {
                    "entity_id": "e3",
                    "version_id": "v1",
                    "meta": {"namespace": "default"},
                },
            ],
        ]
        call_count = [0]

        def get_concept_side_effect(entity_id, channel="latest"):
            call_count[0] += 1
            return {
                "content": {"id": f"idea-{call_count[0]}", "description": "idea"},
                "version_id": "v1",
            }

        mock_api.get_concept.side_effect = get_concept_side_effect
        mem.api = mock_api

        mem._sync_from_api(force_full=True)
        assert mock_api.list_memory_cards.call_count == 2
        assert mock_api.get_concept.call_count == 3  # 3 entities total

    def test_sync_filters_by_namespace(self, tmp_path):
        """Cards from different namespaces are filtered out."""
        mem = _make_memory(
            tmp_path, api=ApiConfig(namespace="my-ns", sync_on_init=False)
        )
        mem.api_sync = None  # force lazy re-creation with mock

        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {"entity_id": "e1", "version_id": "v1", "meta": {"namespace": "my-ns"}},
            {"entity_id": "e2", "version_id": "v1", "meta": {"namespace": "other-ns"}},
        ]
        mock_api.get_concept.return_value = {
            "content": {"id": "idea-ns", "description": "idea"},
            "version_id": "v1",
        }
        mem.api = mock_api

        mem._sync_from_api(force_full=True)
        # Only e1 (my-ns) should be fetched, e2 (other-ns) filtered
        assert mock_api.get_concept.call_count == 1
        mock_api.get_concept.assert_called_once_with("e1", channel="latest")


# ===========================================================================
# 3. API client request body verification (cycle 3 finding fix)
# ===========================================================================


class TestApiClientRequestBody:
    """Verify request bodies contain expected fields."""

    def test_save_concept_body_has_all_fields(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            captured["method"] = request.method
            return httpx.Response(200, json={"entity_id": "e1", "version_id": "v1"})

        client = _mock_client(handler)
        client.save_concept(
            content={"description": "test"},
            name="card-name",
            tags=["t1", "t2"],
            when_to_use="for optimization",
            channel="latest",
            namespace="ns1",
            author="tester",
        )

        body = captured["body"]
        assert body["content"] == {"description": "test"}
        assert body["channel"] == "latest"
        assert body["meta"]["name"] == "card-name"
        assert body["meta"]["tags"] == ["t1", "t2"]
        assert body["meta"]["when_to_use"] == "for optimization"
        assert body["meta"]["namespace"] == "ns1"
        assert body["meta"]["author"] == "tester"

    def test_search_concepts_body_has_all_fields(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"results": [[]]})

        client = _mock_client(handler)
        client.search_concepts(query="test query", limit=10, namespace="ns1")

        body = captured["body"]
        assert body["queries"] == ["test query"]
        assert body["top_k"] == 10
        assert body["namespace"] == "ns1"
        assert body["entity_type"] == "memory_card"

    def test_delete_concept_url_contains_entity_id(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(204)

        client = _mock_client(handler)
        client.delete_concept("eid-123")
        assert "eid-123" in captured["url"]
        assert captured["method"] == "DELETE"

    def test_delete_concept_404_raises(self):
        def handler(request):
            return httpx.Response(404, text="Not Found")

        client = _mock_client(handler)
        with pytest.raises(MemoryStorageError, match="404"):
            client.delete_concept("nonexistent")

    def test_list_memory_cards_params(self):
        captured = {}

        def handler(request):
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json=[])

        client = _mock_client(handler)
        client.list_memory_cards(limit=25, offset=10, channel="draft")
        assert captured["params"]["limit"] == "25"
        assert captured["params"]["offset"] == "10"
        assert captured["params"]["channel"] == "draft"


# ===========================================================================
# 4. _build_entity_meta content assertions (cycle 3 finding fix)
# ===========================================================================


class TestBuildEntityMetaContent:
    """Assert specific content in entity metadata, not just types."""

    def test_name_from_description(self, tmp_path):
        from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card

        _make_memory(tmp_path)
        card = normalize_memory_card(
            {
                "id": "c1",
                "description": "Use simulated annealing for local search refinement",
                "task_description_summary": "TSP solver",
                "keywords": ["annealing", "TSP"],
            }
        )
        name, tags, when_to_use = build_entity_meta(card)

        # Name should contain description text
        assert "simulated annealing" in name.lower() or "local search" in name.lower()
        # Tags should include keywords and category
        assert any("annealing" in t.lower() or "tsp" in t.lower() for t in tags)
        # when_to_use should reference task or description
        assert "TSP" in when_to_use or "annealing" in when_to_use.lower()

    def test_program_card_meta(self, tmp_path):
        from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card

        _make_memory(tmp_path)
        card = normalize_memory_card(
            {
                "category": "program",
                "program_id": "prog-1",
                "description": "Top evolved program",
                "fitness": 95.0,
            }
        )
        name, tags, when_to_use = build_entity_meta(card)
        assert isinstance(name, str)
        assert len(name) > 0
        assert "program" in " ".join(tags).lower() or "program" in name.lower()


# ===========================================================================
# 5. Context manager behavior (replaces __del__)
# ===========================================================================


class TestContextManager:
    def test_with_statement_calls_close(self, tmp_path):
        mock_api = MagicMock()
        with _make_memory(tmp_path) as mem:
            mem.api = mock_api
        mock_api.close.assert_called_once()

    def test_with_statement_without_api(self, tmp_path):
        with _make_memory(tmp_path) as mem:
            mem.save_card({"id": "c1", "description": "test"})
        # Should not raise

    def test_enter_returns_self(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.__enter__() is mem
        mem.close()
