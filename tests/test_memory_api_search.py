"""Cycle 10 (final): API search paths, LLM synthesis, close().

Tests _search_via_api, _synthesize_results, and close() with mocked
self.api and self.llm_service.
"""

import json
from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.memory import AmemGamMemory


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
    return AmemGamMemory(**defaults)


# ===========================================================================
# _search_via_api
# ===========================================================================


class TestSearchViaApi:
    """Test _search_via_api with mocked self.api."""

    def test_no_api_falls_back_to_local(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "c1",
                "description": "annealing optimization",
                "keywords": ["annealing"],
            }
        )
        result = mem._search_via_api("annealing")
        assert "c1" in result

    def test_api_search_returns_cards(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {
            "hits": [{"entity_id": "e1", "version_id": "v1"}],
        }
        mock_api.get_concept.return_value = {
            "content": {
                "id": "idea-1",
                "description": "SA optimization",
                "category": "general",
            },
            "version_id": "v1",
        }
        mem.api = mock_api

        result = mem._search_via_api("optimization")

        assert "idea-1" in result
        mock_api.search_concepts.assert_called_once()
        mock_api.get_concept.assert_called_once_with("e1", channel="latest")

    def test_api_search_no_hits(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {"hits": []}
        mem.api = mock_api

        result = mem._search_via_api("nothing")
        assert "No relevant memories found" in result

    def test_api_search_with_memory_state(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.use_api = True

        captured = {}

        def mock_search(**kwargs):
            captured["query"] = kwargs.get("query")
            return {"hits": []}

        mock_api = MagicMock()
        mock_api.search_concepts.side_effect = mock_search
        mem.api = mock_api

        mem._search_via_api("test query", memory_state="current state info")
        assert "test query" in captured["query"]
        assert "current state info" in captured["query"]

    def test_api_search_updates_entity_maps(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {
            "hits": [{"entity_id": "e1", "version_id": "v1"}],
        }
        mock_api.get_concept.return_value = {
            "content": {"id": "idea-1", "description": "SA"},
            "version_id": "v1",
        }
        mem.api = mock_api

        mem._search_via_api("test")

        assert mem.entity_by_card_id.get("idea-1") == "e1"
        assert mem.card_id_by_entity.get("e1") == "idea-1"
        version = mem.entity_version_by_entity.get("e1")
        assert version is not None and version != "", (
            f"version_id should be set from get_concept response, got {version!r}"
        )
        assert "idea-1" in mem.memory_cards

    def test_api_search_persists_to_index(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {
            "hits": [{"entity_id": "e1"}],
        }
        mock_api.get_concept.return_value = {
            "content": {"id": "idea-1", "description": "test"},
        }
        mem.api = mock_api

        mem._search_via_api("test")

        assert mem.index_file.exists()
        data = json.loads(mem.index_file.read_text())
        assert "idea-1" in data["memory_cards"]

    def test_api_search_skips_empty_entity_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {
            "hits": [
                {"entity_id": "", "version_id": "v1"},
                {"entity_id": "e2", "version_id": "v2"},
            ],
        }
        mock_api.get_concept.return_value = {
            "content": {"id": "idea-2", "description": "valid"},
        }
        mem.api = mock_api

        mem._search_via_api("test")

        # Only e2 should be fetched (e1 has empty entity_id)
        mock_api.get_concept.assert_called_once()

    def test_api_search_with_synthesis_enabled(self, tmp_path):
        """When enable_llm_synthesis=True, _synthesize_results is called."""
        mem = _make_memory(tmp_path)
        mem.use_api = True
        mem.enable_llm_synthesis = True

        mock_api = MagicMock()
        mock_api.search_concepts.return_value = {
            "hits": [{"entity_id": "e1"}],
        }
        mock_api.get_concept.return_value = {
            "content": {"id": "idea-1", "description": "test"},
        }
        mem.api = mock_api

        # No LLM → _synthesize_results falls back to _format_search_results
        result = mem._search_via_api("test")
        assert "idea-1" in result


# ===========================================================================
# _synthesize_results
# ===========================================================================


class TestSynthesizeResults:
    """Test _synthesize_results with mocked llm_service."""

    def test_no_llm_falls_back_to_format(self, tmp_path):
        mem = _make_memory(tmp_path)
        cards = [normalize_memory_card({"id": "c1", "description": "test", "category": "general"})]
        result = mem._synthesize_results("query", None, cards)
        assert "c1" in result
        assert "Query: query" in result

    def test_llm_returns_synthesized_answer(self, tmp_path):
        mem = _make_memory(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "Based on memory card mem-1, you should use SA for optimization.",
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        cards = [
            {"id": "mem-1", "description": "SA optimization", "category": "general"}
        ]
        result = mem._synthesize_results("how to optimize", "current state", cards)

        assert "SA for optimization" in result
        mock_llm.generate.assert_called_once()

        # Verify prompt contains query and card info
        prompt = mock_llm.generate.call_args[0][0]
        assert "how to optimize" in prompt
        assert "current state" in prompt
        assert "mem-1" in prompt

    def test_llm_returns_empty_falls_back(self, tmp_path):
        mem = _make_memory(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("", {}, None, None)
        mem.llm_service = mock_llm

        cards = [normalize_memory_card({"id": "c1", "description": "test", "category": "general"})]
        result = mem._synthesize_results("query", None, cards)
        # Empty LLM response → fallback to _format_search_results
        assert "c1" in result
        assert "Query: query" in result

    def test_llm_exception_falls_back(self, tmp_path):
        mem = _make_memory(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        mem.llm_service = mock_llm

        cards = [normalize_memory_card({"id": "c1", "description": "test", "category": "general"})]
        result = mem._synthesize_results("query", None, cards)
        assert "c1" in result

    def test_synthesize_prompt_includes_card_fields(self, tmp_path):
        mem = _make_memory(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("answer", {}, None, None)
        mem.llm_service = mock_llm

        cards = [
            {
                "id": "c1",
                "description": "SA optimization",
                "category": "retrieval",
                "task_description_summary": "HoVer verification",
                "task_description": "Multi-hop fact verification",
                "keywords": ["SA", "annealing"],
                "explanation": {"summary": "SA works well", "explanations": []},
            }
        ]
        mem._synthesize_results("test", None, cards)

        prompt = mock_llm.generate.call_args[0][0]
        assert "SA optimization" in prompt
        assert "retrieval" in prompt
        assert "HoVer verification" in prompt
        assert "SA works well" in prompt


# ===========================================================================
# search() top-level routing
# ===========================================================================


class TestSearchRouting:
    """Test the search() method's routing logic."""

    def test_search_no_api_no_agent_uses_local(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {"id": "c1", "description": "annealing idea", "keywords": ["annealing"]}
        )
        result = mem.search("annealing")
        assert "c1" in result

    def test_search_with_research_agent_uses_gam(self, tmp_path):
        mem = _make_memory(tmp_path)
        mock_result = MagicMock()
        mock_result.integrated_memory = "1. idea-1: Use SA for optimization"
        mock_agent = MagicMock()
        mock_agent.research.return_value = mock_result
        mem.research_agent = mock_agent

        result = mem.search("optimization")
        assert "SA for optimization" in result
        mock_agent.research.assert_called_once()

    def test_search_gam_failure_falls_back_to_local(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {"id": "c1", "description": "annealing idea", "keywords": ["annealing"]}
        )

        mock_agent = MagicMock()
        mock_agent.research.side_effect = RuntimeError("GAM down")
        mem.research_agent = mock_agent

        result = mem.search("annealing")
        assert "c1" in result  # Fell back to local search

    def test_search_gam_failure_with_api_falls_back_to_api_search(self, tmp_path):
        """GAM fails + use_api=True → falls back to _search_via_api, not local."""
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_agent = MagicMock()
        mock_agent.research.side_effect = RuntimeError("GAM down")
        mem.research_agent = mock_agent

        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = []
        mock_api.search_concepts.return_value = {
            "hits": [{"entity_id": "e1"}],
        }
        mock_api.get_concept.return_value = {
            "content": {"id": "idea-api", "description": "from API"},
            "version_id": "v1",
        }
        mem.api = mock_api

        result = mem.search("test")
        # Should have fallen back to API search after GAM failure
        assert "idea-api" in result
        mock_api.search_concepts.assert_called()

    def test_search_api_mode_syncs_first(self, tmp_path):
        """In API mode, search() calls _sync_from_api before searching."""
        mem = _make_memory(tmp_path)
        mem.use_api = True

        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = []  # Empty sync
        mock_api.search_concepts.return_value = {"hits": []}
        mem.api = mock_api

        mem.search("test")

        # _sync_from_api should have been called (via list_memory_cards)
        mock_api.list_memory_cards.assert_called()


# ===========================================================================
# close()
# ===========================================================================


class TestClose:
    def test_close_with_api(self, tmp_path):
        mem = _make_memory(tmp_path)
        mock_api = MagicMock()
        mem.api = mock_api
        mem.close()
        mock_api.close.assert_called_once()

    def test_close_without_api(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.close()  # Should not raise

    def test_close_idempotent(self, tmp_path):
        """Double close on same api object should not crash."""
        mem = _make_memory(tmp_path)
        mock_api = MagicMock()
        mem.api = mock_api
        mem.close()
        mem.close()  # Second call on same (already closed) api
        assert mock_api.close.call_count == 2
