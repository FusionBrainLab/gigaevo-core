"""Contract tests for the memory system.

These tests pin the exact shape, types, and invariants that downstream code
depends on. They are designed to BREAK loudly if a refactor changes any
behavioral contract. Each test documents WHY the contract matters.
"""

import json

from gigaevo.memory.shared_memory.memory import AmemGamMemory, normalize_memory_card


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
# Contract 1: normalize_memory_card output shape
#
# WHY: Every consumer (save_card, _load_index, _concept_to_card, _persist_index,
# comparison with stored cards) depends on the exact key set. Missing a key or
# adding one silently breaks downstream code.
# ===========================================================================


_GENERAL_CARD_KEYS = frozenset(
    {
        "id",
        "category",
        "description",
        "task_description",
        "task_description_summary",
        "strategy",
        "last_generation",
        "programs",
        "aliases",
        "keywords",
        "evolution_statistics",
        "explanation",
        "works_with",
        "links",
        "usage",
    }
)

_PROGRAM_CARD_KEYS = frozenset(
    {
        "id",
        "category",
        "program_id",
        "task_description",
        "task_description_summary",
        "description",
        "fitness",
        "code",
        "connected_ideas",
    }
)


class TestNormalizeCardContract:
    """Pin the exact output key set of normalize_memory_card."""

    def test_general_card_exact_keys(self):
        card = normalize_memory_card({"description": "test"})
        assert set(card.keys()) == _GENERAL_CARD_KEYS

    def test_program_card_exact_keys(self):
        card = normalize_memory_card({"category": "program", "program_id": "p1"})
        assert set(card.keys()) == _PROGRAM_CARD_KEYS

    def test_general_card_field_types(self):
        card = normalize_memory_card({"id": "c1", "description": "d"})
        assert isinstance(card["id"], str)
        assert isinstance(card["category"], str)
        assert isinstance(card["description"], str)
        assert isinstance(card["task_description"], str)
        assert isinstance(card["task_description_summary"], str)
        assert isinstance(card["strategy"], str)
        assert isinstance(card["last_generation"], int)
        assert isinstance(card["programs"], list)
        assert isinstance(card["aliases"], list)
        assert isinstance(card["keywords"], list)
        assert isinstance(card["evolution_statistics"], dict)
        assert isinstance(card["explanation"], dict)
        assert isinstance(card["explanation"]["explanations"], list)
        assert isinstance(card["explanation"]["summary"], str)
        assert isinstance(card["works_with"], list)
        assert isinstance(card["links"], list)
        assert isinstance(card["usage"], dict)

    def test_program_card_field_types(self):
        card = normalize_memory_card(
            {
                "category": "program",
                "program_id": "p1",
                "fitness": 90.0,
            }
        )
        assert isinstance(card["id"], str)
        assert card["category"] == "program"
        assert isinstance(card["program_id"], str)
        assert isinstance(card["description"], str)
        assert isinstance(card["task_description"], str)
        assert isinstance(card["task_description_summary"], str)
        assert isinstance(card["code"], str)
        assert isinstance(card["connected_ideas"], list)
        # fitness can be float or None
        assert card["fitness"] is None or isinstance(card["fitness"], float)


# ===========================================================================
# Contract 2: save → get roundtrip preserves data
#
# WHY: If save_card normalizes differently than get_card returns, or if
# _persist_index serializes differently than _load_index deserializes,
# data is silently corrupted.
# ===========================================================================


class TestSaveGetRoundtrip:
    """save_card → get_card must preserve all meaningful fields."""

    def test_general_card_roundtrip(self, tmp_path):
        mem = _make_memory(tmp_path)
        original = {
            "id": "c1",
            "description": "Use simulated annealing for local search",
            "task_description": "Solve TSP efficiently",
            "task_description_summary": "TSP solver",
            "strategy": "exploitation",
            "last_generation": 15,
            "programs": ["prog-1", "prog-2"],
            "keywords": ["SA", "local-search"],
            "explanation": {"explanations": ["tried SA"], "summary": "SA works"},
        }
        mem.save_card(original)
        stored = mem.get_card("c1")

        assert stored["id"] == "c1"
        assert stored["description"] == original["description"]
        assert stored["task_description"] == original["task_description"]
        assert (
            stored["task_description_summary"] == original["task_description_summary"]
        )
        assert stored["strategy"] == original["strategy"]
        assert stored["last_generation"] == original["last_generation"]
        assert stored["programs"] == original["programs"]
        assert stored["keywords"] == original["keywords"]
        assert stored["explanation"]["summary"] == "SA works"
        assert stored["explanation"]["explanations"] == ["tried SA"]

    def test_program_card_roundtrip(self, tmp_path):
        mem = _make_memory(tmp_path)
        original = {
            "id": "prog-1",
            "category": "program",
            "program_id": "prog-1",
            "description": "Top evolved program",
            "fitness": 95.5,
            "code": "def solve(x):\n    return sorted(x)\n",
            "connected_ideas": [{"idea_id": "i1", "description": "SA"}],
            "task_description": "Solve TSP",
            "task_description_summary": "TSP",
        }
        mem.save_card(original)
        stored = mem.get_card("prog-1")

        assert stored["category"] == "program"
        assert stored["program_id"] == "prog-1"
        assert stored["fitness"] == 95.5
        assert stored["code"] == original["code"]
        assert stored["connected_ideas"] == original["connected_ideas"]

    def test_persist_reload_roundtrip(self, tmp_path):
        """Save → persist → reload from disk → get must return same data."""
        mem1 = _make_memory(tmp_path)
        mem1.save_card(
            {
                "id": "c1",
                "description": "test idea",
                "keywords": ["k1", "k2"],
                "explanation": {"explanations": ["e1"], "summary": "s"},
                "last_generation": 7,
            }
        )

        mem2 = _make_memory(tmp_path)
        stored = mem2.get_card("c1")
        assert stored["description"] == "test idea"
        assert stored["keywords"] == ["k1", "k2"]
        assert stored["explanation"]["explanations"] == ["e1"]
        assert stored["last_generation"] == 7


# ===========================================================================
# Contract 3: search() output format
#
# WHY: MemorySelectorAgent._extract_card_ids_from_text parses the search
# output with regexes. If the format changes, card ID extraction breaks
# and mutations get no memory context.
# ===========================================================================


class TestSearchOutputContract:
    """Pin the format of search() output that MemorySelectorAgent depends on."""

    def test_no_results_format(self, tmp_path):
        mem = _make_memory(tmp_path)
        result = mem.search("anything")
        assert "No relevant memories found" in result

    def test_results_format_has_query_line(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {"id": "c1", "description": "annealing idea", "keywords": ["annealing"]}
        )
        result = mem.search("annealing")
        assert result.startswith("Query: annealing")

    def test_results_format_has_numbered_cards(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "annealing idea"})
        mem.save_card({"id": "c2", "description": "crossover idea"})
        result = mem.search("annealing crossover")
        # Format: "1. <card_id> [<category>] <description>"
        import re

        numbered = re.findall(r"(?m)^\d+\.\s+\S+\s+\[[^\]]+\]\s+", result)
        assert len(numbered) >= 1

    def test_results_contain_card_ids(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "idea-abc-123", "description": "unique approach"})
        result = mem.search("unique approach")
        assert "idea-abc-123" in result

    def test_results_contain_category_in_brackets(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test", "category": "insight"})
        # normalize_memory_card preserves category
        result = mem.search("test")
        assert "[insight]" in result or "[general]" in result


# ===========================================================================
# Contract 4: api_index.json persistence format
#
# WHY: Changing the index format breaks backward compatibility — old
# checkpoints can't be loaded by new code.
# ===========================================================================


class TestIndexPersistenceContract:
    """Pin the exact JSON structure of api_index.json."""

    def test_index_has_required_top_level_keys(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})

        data = json.loads(mem.index_file.read_text())
        assert "memory_cards" in data
        assert "entity_by_card_id" in data
        assert "entity_version_by_entity" in data

    def test_memory_cards_indexed_by_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})

        data = json.loads(mem.index_file.read_text())
        assert "c1" in data["memory_cards"]
        assert data["memory_cards"]["c1"]["description"] == "test"

    def test_index_card_has_normalized_shape(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})

        data = json.loads(mem.index_file.read_text())
        card = data["memory_cards"]["c1"]
        assert set(card.keys()) == _GENERAL_CARD_KEYS

    def test_index_backward_compatible_load(self, tmp_path):
        """Write a minimal valid index by hand, verify it loads."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir(parents=True)
        index = {
            "memory_cards": {
                "c1": {
                    "id": "c1",
                    "description": "old format card",
                    "category": "general",
                },
            },
            "entity_by_card_id": {},
            "entity_version_by_entity": {},
        }
        (mem_dir / "api_index.json").write_text(json.dumps(index))

        mem = _make_memory(tmp_path)
        card = mem.get_card("c1")
        assert card is not None
        # normalize_memory_card fills in missing fields
        assert card["description"] == "old format card"
        assert card["programs"] == []  # Default filled in


# ===========================================================================
# Contract 5: card_write_stats shape
#
# WHY: The write pipeline (memory_write_example.main()) reads these stats
# to report write outcomes. Missing or renamed keys break reporting.
# ===========================================================================


class TestWriteStatsContract:
    def test_stats_keys(self, tmp_path):
        mem = _make_memory(tmp_path)
        stats = mem.get_card_write_stats()
        assert set(stats.keys()) == {
            "processed",
            "added",
            "rejected",
            "updated",
            "updated_target_cards",
        }

    def test_stats_all_int(self, tmp_path):
        mem = _make_memory(tmp_path)
        stats = mem.get_card_write_stats()
        for key, val in stats.items():
            assert isinstance(val, int), f"{key} should be int, got {type(val)}"

    def test_stats_increment_correctly(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "first"})
        mem.save_card({"id": "c1", "description": "update"})
        mem.save_card({"description": "new"})

        stats = mem.get_card_write_stats()
        assert stats["processed"] == 3
        assert stats["added"] == 2  # c1 + new
        assert stats["updated"] == 1  # c1 second write


# ===========================================================================
# Contract 6: dedup decision shape
#
# WHY: save_card reads action, duplicate_of, updates from the decision dict.
# If parse_llm_card_decision changes its output shape, dedup silently breaks.
# ===========================================================================


class TestDedupDecisionContract:
    def test_parse_decision_shape(self):
        from gigaevo.memory.shared_memory.card_update_dedup import (
            parse_llm_card_decision,
        )

        result = parse_llm_card_decision(
            json.dumps({"action": "add"}),
            candidate_ids={"c1"},
        )
        assert set(result.keys()) == {"action", "reason", "duplicate_of", "updates"}
        assert isinstance(result["action"], str)
        assert isinstance(result["reason"], str)
        assert isinstance(result["duplicate_of"], str)
        assert isinstance(result["updates"], list)

    def test_parse_decision_actions(self):
        from gigaevo.memory.shared_memory.card_update_dedup import (
            parse_llm_card_decision,
        )

        for action in ("add", "discard", "update"):
            if action == "discard":
                text = json.dumps({"action": action, "duplicate_of": "c1"})
            elif action == "update":
                text = json.dumps(
                    {
                        "action": action,
                        "updates": [
                            {
                                "card_id": "c1",
                                "update_explanation": True,
                                "explanation_append": "x",
                            }
                        ],
                    }
                )
            else:
                text = json.dumps({"action": action})
            result = parse_llm_card_decision(text, candidate_ids={"c1"})
            assert result["action"] == action


# ===========================================================================
# Contract 7: MemorySelection shape
#
# WHY: LLMMutationOperator reads .cards and .card_ids from MemorySelection.
# If the dataclass changes, mutation breaks silently.
# ===========================================================================


class TestMemorySelectionContract:
    def test_memory_selection_shape(self):
        from gigaevo.llm.agents.memory_selector import MemorySelection

        sel = MemorySelection(cards=["1. idea"], card_ids=["id-1"])
        assert hasattr(sel, "cards")
        assert hasattr(sel, "card_ids")
        assert isinstance(sel.cards, list)
        assert isinstance(sel.card_ids, list)

    def test_memory_selection_empty(self):
        from gigaevo.llm.agents.memory_selector import MemorySelection

        sel = MemorySelection(cards=[], card_ids=[])
        assert sel.cards == []
        assert sel.card_ids == []


# ===========================================================================
# Contract 8: mutation metadata keys
#
# WHY: The mutation pipeline reads these exact keys from parent.metadata.
# Renaming or removing them silently drops memory from mutation prompts.
# ===========================================================================


class TestMutationMetadataKeysContract:
    def test_metadata_key_values(self):
        from gigaevo.evolution.mutation.context import (
            MUTATION_MEMORY_METADATA_KEY,
            MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
        )

        # Pin exact values — renaming these breaks the pipeline
        assert MUTATION_MEMORY_METADATA_KEY == "mutation_memory"
        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY == "memory_selected_idea_ids"
