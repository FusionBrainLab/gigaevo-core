"""Integration tests: full memory cycle (fill → search → use in mutation).

Tests the complete end-to-end flow:
1. Create memory backend, save idea cards (simulating IdeaTracker output)
2. Search memory with queries (simulating MemorySelectorAgent)
3. Verify memory cards influence mutation context
4. Test the write → persist → reload → search roundtrip
5. Test MemorySelectorAgent directly with mocked backend

All tests run without OPENAI_API_KEY, Redis, Chroma, or network.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory, normalize_memory_card
from gigaevo.memory.memory_write_example import load_memory_cards


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
    return AmemGamMemory(**defaults)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _make_idea_card(idea_id, description, task="Solve the task", **extra):
    card = {
        "id": idea_id,
        "description": description,
        "task_description": task,
        "task_description_summary": task[:50],
        "keywords": extra.pop("keywords", []),
        "explanation": extra.pop("explanation", {"explanations": [], "summary": ""}),
        "programs": extra.pop("programs", []),
    }
    card.update(extra)
    return card


def _make_program_card(program_id, fitness, code="def f(): pass", task="Solve"):
    return {
        "id": f"program-{program_id}",
        "category": "program",
        "program_id": program_id,
        "description": f"Top program (fitness={fitness})",
        "task_description": task,
        "task_description_summary": task,
        "fitness": fitness,
        "code": code,
        "connected_ideas": [],
    }


@dataclass
class FakeProgram:
    code: str = "def solve(): return 1"
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# INTEGRATION TEST 1: Memory fill → persist → reload → search
# ===========================================================================


class TestMemoryFillAndSearch:
    """Simulate IdeaTracker output → save to memory → restart → search."""

    def test_full_write_persist_reload_search_cycle(self, tmp_path):
        """The core integration: write ideas, persist, reload, search."""
        # Phase 1: Fill memory with ideas (simulating IdeaTracker output)
        mem = _make_memory(tmp_path)
        ideas = [
            _make_idea_card(
                "idea-1",
                "Use simulated annealing for local search refinement",
                keywords=["annealing", "local-search", "optimization"],
            ),
            _make_idea_card(
                "idea-2",
                "Apply crossover between top-performing program pairs",
                keywords=["crossover", "genetic", "recombination"],
            ),
            _make_idea_card(
                "idea-3",
                "Add boundary-aware repair step after mutation",
                keywords=["repair", "boundary", "validation"],
            ),
            _make_idea_card(
                "idea-4",
                "Increase retrieval depth for multi-hop verification",
                keywords=["retrieval", "multi-hop", "depth"],
            ),
        ]
        for idea in ideas:
            mem.save_card(idea)

        stats = mem.get_card_write_stats()
        assert stats["processed"] == 4
        assert stats["added"] == 4

        # Phase 2: Persist (already done by save_card) and verify on disk
        assert mem.index_file.exists()
        data = json.loads(mem.index_file.read_text())
        assert len(data["memory_cards"]) == 4

        # Phase 3: Reload from scratch (simulating new process)
        mem2 = _make_memory(tmp_path)
        assert len(mem2.memory_cards) == 4
        assert mem2.get_card("idea-1") is not None

        # Phase 4: Search for relevant ideas
        result = mem2.search("annealing optimization")
        assert "idea-1" in result
        assert "simulated annealing" in result.lower()

        result = mem2.search("crossover recombination")
        assert "idea-2" in result

        result = mem2.search("retrieval depth")
        assert "idea-4" in result

    def test_search_returns_no_results_for_unrelated_query(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_idea_card("idea-1", "simulated annealing"))
        result = mem.search("quantum computing")
        assert "No relevant memories found" in result

    def test_mixed_idea_and_program_cards(self, tmp_path):
        """Both idea cards and program cards coexist."""
        mem = _make_memory(tmp_path)
        mem.save_card(_make_idea_card("idea-1", "Use SA for local search"))
        mem.save_card(_make_program_card("prog-1", 95.0, "def solve(): return 42"))

        assert len(mem.memory_cards) == 2
        idea = mem.get_card("idea-1")
        prog = mem.get_card("program-prog-1")
        assert idea["category"] == "general"
        assert prog["category"] == "program"
        assert prog["fitness"] == 95.0

    def test_search_limit_respected_across_reload(self, tmp_path):
        mem = _make_memory(tmp_path, search_limit=2)
        for i in range(10):
            mem.save_card(_make_idea_card(
                f"idea-{i}",
                f"optimization technique variant {i}",
                keywords=["optimization"],
            ))

        mem2 = _make_memory(tmp_path, search_limit=2)
        result = mem2.search("optimization")
        # Count card IDs in result
        found_ids = [f"idea-{i}" for i in range(10) if f"idea-{i}" in result]
        assert len(found_ids) <= 2


# ===========================================================================
# INTEGRATION TEST 2: load_memory_cards → save_card pipeline
# ===========================================================================


class TestMemoryWritePipeline:
    """Simulate the full write pipeline: banks.json + best_ideas → memory."""

    def test_banks_to_memory_roundtrip(self, tmp_path):
        """Load from banks.json → save to memory → verify searchable."""
        # Create input files (simulating IdeaTracker output)
        banks = tmp_path / "banks.json"
        _write_json(banks, [{
            "active_bank": [
                {
                    "id": "idea-1",
                    "description": "Use simulated annealing for local refinement",
                    "task_description": "Solve TSP",
                    "task_description_summary": "TSP solver",
                    "programs": ["prog-1"],
                },
                {
                    "id": "idea-2",
                    "description": "Apply genetic crossover between pairs",
                    "task_description": "Solve TSP",
                    "task_description_summary": "TSP solver",
                    "programs": ["prog-2"],
                },
            ],
        }])

        best_ideas = tmp_path / "best_ideas.json"
        _write_json(best_ideas, [{
            "best_ideas": [
                {"idea_id": "idea-1", "description": "SA for refinement"},
                {"idea_id": "idea-2", "description": "Genetic crossover"},
            ],
        }])

        # Load cards via the write pipeline
        cards = load_memory_cards(banks, best_ideas)
        assert len(cards) == 2

        # Save to memory backend
        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)

        assert len(mem.memory_cards) == 2
        assert mem.get_card("idea-1")["description"] == "Use simulated annealing for local refinement"

        # Search works
        result = mem.search("annealing")
        assert "idea-1" in result

    def test_banks_with_programs_to_memory(self, tmp_path):
        """Full pipeline with program cards."""
        banks = tmp_path / "banks.json"
        _write_json(banks, [{
            "active_bank": [
                {
                    "id": "idea-1",
                    "description": "SA refinement",
                    "task_description": "Solve TSP",
                    "task_description_summary": "TSP",
                    "programs": ["prog-1"],
                },
            ],
        }])

        best_ideas = tmp_path / "best_ideas.json"
        _write_json(best_ideas, [{"best_ideas": [{"idea_id": "idea-1"}]}])

        programs = tmp_path / "programs.json"
        _write_json(programs, [{
            "programs": [
                {
                    "id": "prog-1",
                    "fitness": 90.0,
                    "code": "def solve():\n    return 42\n",
                    "task_description": "Solve TSP",
                    "task_description_summary": "TSP",
                },
            ],
        }])

        cards = load_memory_cards(
            banks, best_ideas,
            programs_path=programs,
            best_programs_percent=100.0,
        )

        # Should have 1 idea + 1 program card
        idea_cards = [c for c in cards if c.get("category") != "program"]
        program_cards = [c for c in cards if c.get("category") == "program"]
        assert len(idea_cards) == 1
        assert len(program_cards) == 1
        assert program_cards[0]["fitness"] == 90.0

        # Save all to memory
        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)

        stats = mem.get_card_write_stats()
        assert stats["processed"] == 2
        assert stats["added"] == 2


# ===========================================================================
# INTEGRATION TEST 3: MemorySelectorAgent with mocked backend
# ===========================================================================


class TestMemorySelectorIntegration:
    """Test MemorySelectorAgent.select() with a real AmemGamMemory backend."""

    def _make_selector_with_memory(self, tmp_path, ideas):
        """Create a MemorySelectorAgent with pre-filled local memory."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)
        for idea in ideas:
            mem.save_card(idea)

        # Create selector with injected memory backend
        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem
        return selector

    @pytest.mark.asyncio
    async def test_select_returns_relevant_cards(self, tmp_path):
        ideas = [
            _make_idea_card("idea-1", "Use simulated annealing for optimization",
                            keywords=["annealing", "optimization"]),
            _make_idea_card("idea-2", "Apply crossover for diversity",
                            keywords=["crossover"]),
        ]
        selector = self._make_selector_with_memory(tmp_path, ideas)
        parent = FakeProgram(code="def solve(): return sum(x)")

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="Optimize the function using annealing",
            metrics_description="fitness: accuracy",
            memory_text="",
            max_cards=3,
        )

        # Must return non-empty cards with actual content
        assert len(selection.cards) > 0
        # Cards should contain actual text, not empty strings
        assert all(len(card.strip()) > 0 for card in selection.cards)

    @pytest.mark.asyncio
    async def test_select_with_no_memory_returns_empty(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = "test: no backend"
        selector.memory = None

        parent = FakeProgram()
        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="test",
            metrics_description="fitness",
            memory_text="",
            max_cards=3,
        )

        assert selection.cards == []
        assert selection.card_ids == []

    @pytest.mark.asyncio
    async def test_select_max_cards_zero_returns_empty(self, tmp_path):
        ideas = [_make_idea_card("idea-1", "test idea")]
        selector = self._make_selector_with_memory(tmp_path, ideas)
        parent = FakeProgram()

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="test",
            metrics_description="fitness",
            memory_text="",
            max_cards=0,
        )

        assert selection.cards == []
        assert selection.card_ids == []

    def test_build_request_format(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = None

        parent = FakeProgram(code="def solve(x):\n    return x + 1")
        query = selector._build_request(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="Solve the optimization problem",
            metrics_description="fitness: accuracy percentage",
            max_cards=3,
        )

        assert "MUTATION INPUTS" in query
        assert "TASK DESCRIPTION:" in query
        assert "Solve the optimization problem" in query
        assert "AVAILABLE METRICS:" in query
        assert "accuracy percentage" in query
        assert "MUTATION MODE:" in query
        assert "rewrite" in query
        assert "def solve(x):" in query
        assert "Return exactly 3 concise ideas" in query

    def test_parse_search_result_numbered(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        result = (
            "Query: test\n\n"
            "1. Use simulated annealing for local search\n"
            "2. Apply crossover between top pairs\n"
            "3. Add repair step after mutation\n"
        )
        cards = selector._parse_search_result(result, max_cards=2)
        assert len(cards) == 2
        assert "simulated annealing" in cards[0]

    def test_parse_search_result_no_relevant(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        result = "Query: test\n\nNo relevant memories found."
        cards = selector._parse_search_result(result, max_cards=3)
        assert cards == []

    def test_extract_card_ids_from_text(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        text = (
            "Top relevant memory cards:\n"
            "1. idea-1 [general] Use SA for optimization\n"
            "2. idea-2 [general] Apply crossover\n"
        )
        ids = selector._extract_card_ids_from_text(text)
        assert "idea-1" in ids
        assert "idea-2" in ids

    def test_extract_card_ids_from_raw_memory(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        raw_memory = {
            "final_decision": {
                "top_ideas": [
                    {"card_id": "idea-1"},
                    {"card_id": "idea-2"},
                ],
            },
        }
        ids = selector._extract_card_ids_from_raw_memory(raw_memory)
        assert ids == ["idea-1", "idea-2"]

    def test_merge_card_ids_dedupes(self, tmp_path):
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        merged = selector._merge_card_ids(
            primary=["idea-1", "idea-2"],
            secondary=["idea-2", "idea-3"],
            max_cards=3,
        )
        assert merged == ["idea-1", "idea-2", "idea-3"]


# ===========================================================================
# INTEGRATION TEST 4: Full cycle — fill + update + search + delete
# ===========================================================================


class TestFullMemoryCycle:
    """End-to-end: create → update → search → delete → verify."""

    def test_crud_lifecycle(self, tmp_path):
        mem = _make_memory(tmp_path)

        # Create
        mem.save_card(_make_idea_card("idea-1", "SA optimization", keywords=["annealing"]))
        mem.save_card(_make_idea_card("idea-2", "Crossover recombination", keywords=["crossover"]))
        assert len(mem.memory_cards) == 2

        # Update
        mem.save_card(_make_idea_card("idea-1", "Enhanced SA with adaptive cooling"))
        assert mem.get_card("idea-1")["description"] == "Enhanced SA with adaptive cooling"

        # Search
        result = mem.search("cooling")
        assert "idea-1" in result

        # Delete
        mem.delete("idea-2")
        assert mem.get_card("idea-2") is None
        assert len(mem.memory_cards) == 1

        # Verify persistence
        mem2 = _make_memory(tmp_path)
        assert len(mem2.memory_cards) == 1
        assert mem2.get_card("idea-1") is not None
        assert mem2.get_card("idea-2") is None

    def test_dedup_integration(self, tmp_path):
        """Dedup enabled: save similar cards, verify LLM-guided dedup works."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card(_make_idea_card("idea-1", "Use simulated annealing"))

        # Mock LLM to say "discard — duplicate of idea-1"
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "discard", "duplicate_of": "idea-1"}),
            {}, None, None,
        )
        mem.llm_service = mock_llm
        mem._score_retrieved_candidates = MagicMock(
            return_value=[{"card_id": "idea-1", "final_score": 0.9}]
        )

        # Try to save duplicate
        result_id = mem.save_card({"description": "SA optimization variant"})
        assert result_id == "idea-1"  # Returned existing card
        assert len(mem.memory_cards) == 1  # No new card added
        stats = mem.get_card_write_stats()
        assert stats["rejected"] == 1

    def test_many_cards_persistence(self, tmp_path):
        """Save 50 cards, reload, verify all present and searchable."""
        mem = _make_memory(tmp_path)
        for i in range(50):
            mem.save_card(_make_idea_card(
                f"idea-{i}",
                f"Optimization technique number {i} using method_{i}",
                keywords=[f"method_{i}"],
            ))

        mem2 = _make_memory(tmp_path)
        assert len(mem2.memory_cards) == 50

        # Search for specific card
        result = mem2.search("method_42")
        assert "idea-42" in result

    def test_program_and_idea_cards_search_separately(self, tmp_path):
        """Program cards and idea cards are both stored and searchable."""
        mem = _make_memory(tmp_path)
        mem.save_card(_make_idea_card("idea-1", "annealing optimization"))
        mem.save_card(_make_program_card("prog-1", 95.0, "def solve(): return 42"))

        # Search for idea
        result = mem.search("annealing")
        assert "idea-1" in result

        # Search for program (by description keyword)
        result = mem.search("program fitness")
        assert "program-prog-1" in result


# ===========================================================================
# Search fallback paths and error handling
# ===========================================================================


class TestSearchFallbackPaths:
    """Test _search_with_ids GAM vs plain-search fallback, parse guarantees,
    and API client error handling."""

    def test_search_with_ids_fallback_to_plain_search(self, tmp_path):
        """When research_agent is None, _search_with_ids falls to memory.search()."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)
        mem.save_card(_make_idea_card("idea-1", "simulated annealing optimization",
                                       keywords=["annealing"]))
        # Ensure no research_agent
        assert mem.research_agent is None

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        result_text, card_ids = selector._search_with_ids("annealing optimization")
        assert "idea-1" in result_text
        assert isinstance(card_ids, list)

    def test_search_with_ids_gam_path(self, tmp_path):
        """When research_agent exists, _search_with_ids uses it."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)
        mem.save_card(_make_idea_card("idea-1", "annealing"))

        # Mock research_agent
        mock_result = MagicMock()
        mock_result.integrated_memory = "1. idea-1 [general] Use annealing"
        mock_result.raw_memory = {
            "final_decision": {"top_ideas": [{"card_id": "idea-1"}]},
        }
        mock_agent = MagicMock()
        mock_agent.research.return_value = mock_result
        mem.research_agent = mock_agent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        result_text, card_ids = selector._search_with_ids("annealing")
        assert "idea-1" in result_text
        assert "idea-1" in card_ids
        mock_agent.research.assert_called_once()

    def test_search_with_ids_gam_failure_falls_back(self, tmp_path):
        """When research_agent.research() raises, falls back to plain search."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)
        mem.save_card(_make_idea_card("idea-1", "annealing optimization",
                                       keywords=["annealing"]))

        mock_agent = MagicMock()
        mock_agent.research.side_effect = RuntimeError("GAM failed")
        mem.research_agent = mock_agent

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        result_text, card_ids = selector._search_with_ids("annealing")
        assert "idea-1" in result_text  # Fell back to plain search

    def test_merge_updated_card_does_not_mutate_original_explanations(self, tmp_path):
        """merge_updated_card: _safe_string_list creates new list, so
        existing card's explanation.explanations is NOT mutated.
        However, the top-level dict IS shallow-copied, so other nested
        dicts (usage, evolution_statistics) could be mutated."""
        from gigaevo.memory.shared_memory.card_update_dedup import merge_updated_card
        import copy

        existing = {
            "id": "c1",
            "description": "original",
            "programs": ["p1"],
            "explanation": {"explanations": ["old"], "summary": "old sum"},
            "last_generation": 5,
        }
        existing_snapshot = copy.deepcopy(existing)

        incoming = {"programs": ["p2"], "last_generation": 10}
        update = {"update_explanation": True, "explanation_append": "extra"}

        merge_updated_card(existing, incoming, update)

        # Explanations list is NOT mutated (safe: _safe_string_list creates new list)
        assert existing["explanation"]["explanations"] == existing_snapshot["explanation"]["explanations"]
        # But programs list IS shared — merged dict references same list objects
        # from the shallow copy. The merged result has new programs list from
        # dedupe_keep_order, so existing is safe here too.

    def test_parse_llm_card_decision_always_returns_dict(self):
        """Proves retry loop in _decide_card_action is dead code:
        parse_llm_card_decision ALWAYS returns a dict, even for garbage input.
        """
        from gigaevo.memory.shared_memory.card_update_dedup import parse_llm_card_decision

        garbage_inputs = [
            "",
            "I don't know",
            "Sorry, I cannot process this",
            "<html>502 Bad Gateway</html>",
            "null",
            "[]",
            "42",
        ]
        for text in garbage_inputs:
            result = parse_llm_card_decision(text, candidate_ids={"c1"})
            assert isinstance(result, dict), f"Expected dict for input: {text!r}"
            assert "action" in result

    def test_http_200_non_json_raises(self):
        """_ConceptApiClient._request crashes on 200 with non-JSON body."""
        import httpx
        from gigaevo.memory.shared_memory.memory import _ConceptApiClient

        def handler(request):
            return httpx.Response(200, text="<html>502 Bad Gateway</html>")

        transport = httpx.MockTransport(handler)
        client = _ConceptApiClient.__new__(_ConceptApiClient)
        client._http = httpx.Client(base_url="http://test:8000", transport=transport)

        # BUG: raises json.JSONDecodeError, not RuntimeError
        with pytest.raises(Exception):  # JSONDecodeError or RuntimeError
            client.get_concept("eid-1")
