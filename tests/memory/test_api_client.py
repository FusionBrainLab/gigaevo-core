"""Tests for _ConceptApiClient HTTP wrapper and API-mode AmemGamMemory paths.

All HTTP calls are mocked via httpx mock transport.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.utils import truncate_text
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# _ConceptApiClient
# ---------------------------------------------------------------------------


def _mock_client(responses):
    """Create a _ConceptApiClient with mocked HTTP transport."""
    transport = httpx.MockTransport(responses)
    client = _ConceptApiClient.__new__(_ConceptApiClient)
    client._http = httpx.Client(base_url="http://test:8000", transport=transport)
    return client


class TestConceptApiClientSaveConcept:
    def test_create_new(self):
        def handler(request):
            assert request.method == "POST"
            assert "/v1/memory-cards" in str(request.url)
            return httpx.Response(200, json={"entity_id": "eid-1", "version_id": "v1"})

        client = _mock_client(handler)
        result = client.save_concept(
            content={"description": "test"},
            name="card",
            tags=["t"],
            when_to_use="always",
            channel="latest",
            namespace="ns",
            author="me",
        )
        assert result["entity_id"] == "eid-1"

    def test_update_existing(self):
        def handler(request):
            assert request.method == "PUT"
            assert "eid-1" in str(request.url)
            return httpx.Response(200, json={"entity_id": "eid-1", "version_id": "v2"})

        client = _mock_client(handler)
        result = client.save_concept(
            content={},
            name="card",
            tags=[],
            when_to_use="",
            channel="latest",
            namespace=None,
            author=None,
            entity_id="eid-1",
        )
        assert result["version_id"] == "v2"


class TestConceptApiClientGetConcept:
    def test_success(self):
        def handler(request):
            return httpx.Response(200, json={"content": {"description": "hello"}})

        client = _mock_client(handler)
        result = client.get_concept("eid-1")
        assert result["content"]["description"] == "hello"

    def test_empty_response_raises(self):
        def handler(request):
            return httpx.Response(204)

        client = _mock_client(handler)
        with pytest.raises(RuntimeError, match="Unexpected empty response"):
            client.get_concept("eid-1")


class TestConceptApiClientListMemoryCards:
    def test_returns_list(self):
        def handler(request):
            return httpx.Response(200, json=[{"entity_id": "e1"}, {"entity_id": "e2"}])

        client = _mock_client(handler)
        result = client.list_memory_cards(limit=10)
        assert len(result) == 2

    def test_non_list_returns_empty(self):
        def handler(request):
            return httpx.Response(200, json={"error": "bad"})

        client = _mock_client(handler)
        result = client.list_memory_cards(limit=10)
        assert result == []

    def test_filters_non_dicts(self):
        def handler(request):
            return httpx.Response(200, json=[{"entity_id": "e1"}, "bad", None])

        client = _mock_client(handler)
        result = client.list_memory_cards(limit=10)
        assert len(result) == 1


class TestConceptApiClientSearchConcepts:
    def test_success(self):
        def handler(request):
            body = json.loads(request.content)
            assert body["queries"] == ["test query"]
            return httpx.Response(
                200, json={"results": [[{"entity_id": "e1", "score": 0.9}]]}
            )

        client = _mock_client(handler)
        result = client.search_concepts(query="test query", limit=5, namespace="ns")
        assert len(result["hits"]) == 1
        assert result["hits"][0]["entity_id"] == "e1"

    def test_empty_query(self):
        client = _mock_client(lambda r: httpx.Response(200, json={}))
        result = client.search_concepts(query="", limit=5, namespace=None)
        assert result == {"hits": [], "total": 0}

    def test_no_results(self):
        def handler(request):
            return httpx.Response(200, json={"results": []})

        client = _mock_client(handler)
        result = client.search_concepts(query="test", limit=5, namespace=None)
        assert result["hits"] == []


class TestConceptApiClientDeleteConcept:
    def test_success(self):
        def handler(request):
            assert request.method == "DELETE"
            return httpx.Response(204)

        client = _mock_client(handler)
        client.delete_concept("eid-1")  # Should not raise


class TestConceptApiClientErrors:
    def test_connect_error(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        client = _mock_client(handler)
        with pytest.raises(RuntimeError, match="Cannot connect"):
            client.save_concept(
                content={},
                name="",
                tags=[],
                when_to_use="",
                channel="latest",
                namespace=None,
                author=None,
            )

    def test_timeout_error(self):
        def handler(request):
            raise httpx.TimeoutException("timed out")

        client = _mock_client(handler)
        with pytest.raises(RuntimeError, match="timed out"):
            client.get_concept("eid-1")

    def test_http_400_raises(self):
        def handler(request):
            return httpx.Response(400, text="Bad Request")

        client = _mock_client(handler)
        with pytest.raises(RuntimeError, match="400"):
            client.get_concept("eid-1")

    def test_http_500_raises(self):
        def handler(request):
            return httpx.Response(500, text="Internal Server Error")

        client = _mock_client(handler)
        with pytest.raises(RuntimeError, match="500"):
            client.get_concept("eid-1")

    def test_close(self):
        client = _mock_client(lambda r: httpx.Response(200, json={}))
        client.close()  # Should not raise


# ---------------------------------------------------------------------------
# _truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_short_passthrough(self):
        assert truncate_text("hello") == "hello"

    def test_long_truncated(self):
        result = truncate_text("x" * 2000, max_chars=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_none_returns_empty(self):
        assert truncate_text(None) == ""

    def test_exact_boundary(self):
        text = "a" * 1200
        assert truncate_text(text) == text

    def test_one_over_boundary(self):
        text = "a" * 1201
        result = truncate_text(text)
        assert len(result) == 1200
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# _decide_card_action with mocked LLM
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


class TestDecideCardAction:
    def test_no_llm_returns_add(self, tmp_path):
        mem = _make_memory(tmp_path)
        result = mem.dedup.decide_action(
            normalize_memory_card({"description": "test"}), [{"card_id": "c1"}]
        )
        assert result["action"] == "add"

    def test_no_candidates_returns_add(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.llm_service = MagicMock()
        result = mem.dedup.decide_action(
            normalize_memory_card({"description": "test"}), []
        )
        assert result["action"] == "add"
        mem.llm_service.generate.assert_not_called()

    def test_llm_discard_parsed(self, tmp_path):
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "existing", "description": "original"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "discard", "duplicate_of": "existing"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm

        candidates = [{"card_id": "existing", "final_score": 0.9}]
        result = mem.dedup.decide_action(
            normalize_memory_card({"description": "dup"}), candidates
        )
        assert result["action"] == "discard"
        assert result["duplicate_of"] == "existing"

    def test_llm_exception_retries(self, tmp_path):
        mem = _make_memory(
            tmp_path,
            card_update_dedup_config={
                "enabled": True,
                "llm": {"max_retries": 3},
            },
        )
        mem.save_card({"id": "c1", "description": "test"})

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            (json.dumps({"action": "add"}), {}, None, None),
        ]
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm

        candidates = [{"card_id": "c1", "final_score": 0.5}]
        result = mem.dedup.decide_action(
            normalize_memory_card({"description": "new"}), candidates
        )
        assert result["action"] == "add"
        assert mock_llm.generate.call_count == 3


# ---------------------------------------------------------------------------
# _dedup_candidates_for_llm
# ---------------------------------------------------------------------------


class TestDedupCandidatesForLlm:
    def test_builds_payload(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "c1",
                "description": "Use simulated annealing",
                "task_description_summary": "TSP",
                "explanation": {"explanations": ["tried SA"], "summary": "SA works"},
            }
        )

        candidates = [{"card_id": "c1", "final_score": 0.8, "scores": {}}]
        result = mem.dedup.format_for_llm(candidates)
        assert len(result) == 1
        assert result[0]["card_id"] == "c1"
        assert "simulated annealing" in result[0]["description"]
        assert result[0]["explanation_summary"] == "SA works"

    def test_missing_card_skipped(self, tmp_path):
        mem = _make_memory(tmp_path)
        candidates = [{"card_id": "nonexistent", "final_score": 0.5}]
        result = mem.dedup.format_for_llm(candidates)
        assert result == []

    def test_truncates_long_text(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "c1",
                "description": "x" * 5000,
            }
        )
        candidates = [{"card_id": "c1", "final_score": 0.8, "scores": {}}]
        result = mem.dedup.format_for_llm(candidates)
        assert len(result[0]["description"]) <= 1200
