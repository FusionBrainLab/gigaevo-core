"""Tests for _ConceptApiClient HTTP wrapper and API-mode AmemGamMemory paths.

All HTTP calls are mocked via httpx mock transport.
"""

import json

import httpx
import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.utils import truncate_text

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
        with pytest.raises(MemoryStorageError, match="Unexpected empty response"):
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
        with pytest.raises(MemoryStorageError, match="Cannot connect"):
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
        with pytest.raises(MemoryStorageError, match="timed out"):
            client.get_concept("eid-1")

    def test_http_400_raises(self):
        def handler(request):
            return httpx.Response(400, text="Bad Request")

        client = _mock_client(handler)
        with pytest.raises(MemoryStorageError, match="400"):
            client.get_concept("eid-1")

    def test_http_500_raises(self):
        def handler(request):
            return httpx.Response(500, text="Internal Server Error")

        client = _mock_client(handler)
        with pytest.raises(MemoryStorageError, match="500"):
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
