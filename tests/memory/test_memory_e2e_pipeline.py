"""End-to-end pipeline test: ideas_tracker → normalize_memory_card → load_memory_cards.

This test covers the integration boundary that was broken by Bug #2 (PR #161):
- ideas_tracker produces RecordCardExtended with aliases: list[dict[...]]
- normalize_memory_card must pass through dict aliases without crashing
- load_memory_cards must preserve the full card metadata

Without this E2E test, bugs in the type conversions at the boundary crash
silently during memory write, wasting API credits.
"""

import json
from pathlib import Path

from gigaevo.memory.memory_write_example import load_memory_cards
from gigaevo.memory.shared_memory.card_conversion import (
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard


def _write_json(path: Path, payload: dict | list) -> None:
    """Write JSON with ensure_ascii=True (matching ideas_tracker output)."""
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


# ===========================================================================
# Mock ideas_tracker output (RecordCardExtended-like dicts)
# ===========================================================================


def make_ideas_tracker_card(
    idea_id: str,
    description: str,
    has_version_history: bool = False,
) -> dict:
    """Create a mock ideas_tracker output card with aliases as dict version history.

    Mirrors RecordCardExtended structure:
    - aliases: list[dict[str, dict[str, str | list[str]]]] (version history)
    - explanation: dict[str, list[str] | str] (explanations list + summary)
    """
    aliases = []
    if has_version_history:
        # Version 1: initial description
        aliases.append(
            {
                "exp1-prog1": {
                    "description": f"{description} (initial)",
                    "programs": ["p1"],
                    "explanations": ["Found this pattern in early runs"],
                }
            }
        )
        # Version 2: updated description (simulating update_idea call)
        aliases.append(
            {
                "exp1-prog2": {
                    "description": f"{description} (refined)",
                    "programs": ["p1", "p2"],
                    "explanations": [
                        "Found this pattern in early runs",
                        "Refined after testing",
                    ],
                }
            }
        )

    return {
        "id": idea_id,
        "category": "general",
        "description": description,
        "task_description": "Solve multi-hop retrieval",
        "task_description_summary": "Retrieval optimization",
        "strategy": "exploration",
        "last_generation": 15,
        "programs": ["p1", "p2"],
        "aliases": aliases,  # dict version history, NOT string list
        "keywords": ["retrieval", "chunking"],
        "evolution_statistics": {"improved_count": 3, "generations": [5, 10, 15]},
        "explanation": {
            "explanations": ["Found effective chunking strategy"],
            "summary": "Improved retrieval via adaptive chunking",
        },
        "works_with": ["idea-2", "idea-3"],
        "links": ["related-concept-1"],
        "usage": {"times_used": 7, "success_rate": 0.85},
    }


# ===========================================================================
# normalize_memory_card — ideas_tracker input
# ===========================================================================


class TestNormalizeWithIdeasTrackerOutput:
    """Test normalize_memory_card with realistic ideas_tracker output shapes."""

    def test_normalize_card_with_dict_aliases(self):
        """Card with dict aliases (version history) should not crash."""
        card = make_ideas_tracker_card(
            "idea-1", "Retrieval chunking", has_version_history=True
        )
        result = normalize_memory_card(card)

        assert isinstance(result, MemoryCard)
        assert result.id == "idea-1"
        assert result.description == "Retrieval chunking"
        # Aliases preserved exactly as-is (list of dicts)
        assert len(result.aliases) == 2
        assert isinstance(result.aliases[0], dict)
        assert "exp1-prog1" in result.aliases[0]

    def test_preserve_evolution_statistics(self):
        """Nested dict in evolution_statistics (nested list/dict) preserved."""
        card = make_ideas_tracker_card(
            "idea-2", "Pooling strategy", has_version_history=False
        )
        result = normalize_memory_card(card)

        assert result.evolution_statistics["improved_count"] == 3
        assert result.evolution_statistics["generations"] == [5, 10, 15]

    def test_preserve_explanation_structure(self):
        """explanation dict with explanations list and summary preserved."""
        card = make_ideas_tracker_card(
            "idea-3", "Token strategy", has_version_history=False
        )
        result = normalize_memory_card(card)

        assert result.explanation.explanations == ["Found effective chunking strategy"]
        assert result.explanation.summary == "Improved retrieval via adaptive chunking"

    def test_preserve_usage_dict(self):
        """usage dict with arbitrary values preserved."""
        card = make_ideas_tracker_card("idea-4", "Weighting", has_version_history=False)
        result = normalize_memory_card(card)

        assert result.usage["times_used"] == 7
        assert result.usage["success_rate"] == 0.85

    def test_full_roundtrip_preserves_all_fields(self):
        """Complete card with all complex nested structures."""
        card = make_ideas_tracker_card(
            "idea-full", "Complete idea", has_version_history=True
        )
        result = normalize_memory_card(card)

        assert result.id == "idea-full"
        assert result.category == "general"
        assert len(result.aliases) == 2
        assert isinstance(result.aliases[0], dict)
        assert result.keywords == ["retrieval", "chunking"]
        assert result.works_with == ["idea-2", "idea-3"]
        assert result.links == ["related-concept-1"]
        assert result.strategy == "exploration"
        assert result.last_generation == 15


# ===========================================================================
# load_memory_cards — end-to-end pipeline
# ===========================================================================


class TestLoadMemoryCardsWithIdeasTrackerOutput:
    """Test the full load_memory_cards pipeline with ideas_tracker shapes."""

    def test_e2e_load_mixed_ideas_and_programs(self, tmp_path):
        """Full pipeline: ideas_tracker output → load_memory_cards → validate types."""
        # Create banks with ideas_tracker-format cards
        banks_path = tmp_path / "banks.json"
        _write_json(
            banks_path,
            [
                {
                    "active_bank": [
                        make_ideas_tracker_card(
                            "idea-1", "Chunking", has_version_history=True
                        ),
                        make_ideas_tracker_card(
                            "idea-2", "Pooling", has_version_history=False
                        ),
                    ]
                }
            ],
        )

        best_ideas_path = tmp_path / "best_ideas.json"
        _write_json(
            best_ideas_path,
            [{"best_ideas": [{"idea_id": "idea-1"}, {"idea_id": "idea-2"}]}],
        )

        # Load and validate
        cards = load_memory_cards(banks_path, best_ideas_path)
        assert len(cards) == 2

        card1 = cards[0]
        assert isinstance(card1, MemoryCard)
        assert card1.id == "idea-1"
        assert len(card1.aliases) == 2
        assert isinstance(card1.aliases[0], dict)
        assert card1.description == "Chunking"
        assert card1.evolution_statistics["improved_count"] == 3

        card2 = cards[1]
        assert isinstance(card2, MemoryCard)
        assert card2.id == "idea-2"
        assert len(card2.aliases) == 0

    def test_e2e_program_cards_excluded_from_ideas_tracker_output(self, tmp_path):
        """Program cards should be filtered correctly in ideas_tracker scenario."""
        banks_path = tmp_path / "banks.json"
        _write_json(
            banks_path,
            [
                {
                    "active_bank": [
                        make_ideas_tracker_card(
                            "idea-1", "Good idea", has_version_history=True
                        )
                    ]
                }
            ],
        )

        best_ideas_path = tmp_path / "best_ideas.json"
        _write_json(best_ideas_path, [{"best_ideas": [{"idea_id": "idea-1"}]}])

        programs_path = tmp_path / "programs.json"
        _write_json(
            programs_path,
            [
                {
                    "programs": [
                        {
                            "id": "prog-1",
                            "fitness": 85.5,
                            "code": "def f(): pass",
                            "task_description_summary": "Task",
                        }
                    ]
                }
            ],
        )

        # Load with programs
        cards = load_memory_cards(
            banks_path,
            best_ideas_path,
            programs_path=programs_path,
            best_programs_percent=100.0,
        )

        # Should have both idea and program
        idea_cards = [c for c in cards if c.category == "general"]
        prog_cards = [c for c in cards if c.category == "program"]

        assert len(idea_cards) == 1
        assert len(prog_cards) == 1

        prog = prog_cards[0]
        assert isinstance(prog, ProgramCard)
        assert prog.program_id == "prog-1"
        assert prog.fitness == 85.5

    def test_e2e_missing_idea_creates_minimal_card(self, tmp_path):
        """If best_ideas references missing idea, create minimal card."""
        banks_path = tmp_path / "banks.json"
        _write_json(banks_path, [{"active_bank": []}])

        best_ideas_path = tmp_path / "best_ideas.json"
        _write_json(
            best_ideas_path,
            [
                {
                    "best_ideas": [
                        {
                            "idea_id": "missing-idea",
                            "description": "Reconstructed from best_ideas",
                        }
                    ]
                }
            ],
        )

        cards = load_memory_cards(banks_path, best_ideas_path)
        assert len(cards) == 1
        assert cards[0].id == "missing-idea"
        # Aliases should be empty (no active_bank entry)
        assert cards[0].aliases == []

    def test_e2e_complex_nested_structures_survive(self, tmp_path):
        """Test that deeply nested structures (common in ideas_tracker) survive."""
        complex_card = make_ideas_tracker_card(
            "idea-nested", "Complex", has_version_history=True
        )
        # Add more nesting
        complex_card["evolution_statistics"] = {
            "by_generation": {
                "5": {"fitness": 50.0, "count": 3},
                "10": {"fitness": 75.0, "count": 5},
            },
            "total_improvements": [
                {"gen": 5, "delta": 5.0},
                {"gen": 10, "delta": 25.0},
            ],
        }
        complex_card["usage"] = {
            "by_run": {
                "run-1": {"count": 3, "success": True},
                "run-2": {"count": 4, "success": False},
            },
            "aggregate": {"total": 7, "success_rate": 0.5},
        }

        banks_path = tmp_path / "banks.json"
        _write_json(banks_path, [{"active_bank": [complex_card]}])

        best_ideas_path = tmp_path / "best_ideas.json"
        _write_json(best_ideas_path, [{"best_ideas": [{"idea_id": "idea-nested"}]}])

        cards = load_memory_cards(banks_path, best_ideas_path)
        card = cards[0]

        # Verify deep nesting survived
        assert card.evolution_statistics["by_generation"]["5"]["fitness"] == 50.0
        assert card.evolution_statistics["total_improvements"][1]["delta"] == 25.0
        assert card.usage["by_run"]["run-1"]["count"] == 3
        assert card.usage["aggregate"]["total"] == 7


# ===========================================================================
# Regression: the exact bug from PR #161
# ===========================================================================


class TestRegression_BugPR161:
    """Regression test for Bug #2: Pydantic aliases type mismatch.

    Before fix: normalize_memory_card would crash when ideas_tracker output
    had aliases: list[dict[...]] because MemoryCard.aliases was typed list[str].

    This test verifies the fix (aliases: list[Any]) handles this correctly.
    """

    def test_bug_pr161_aliases_type_mismatch(self):
        """Original bug: aliases list[dict] crashed when expected list[str]."""
        # This exact shape crashed before the fix
        card_input = {
            "id": "idea-1",
            "description": "Idea",
            "aliases": [
                {
                    "exp1-prog1": {
                        "description": "Old version",
                        "programs": ["p1"],
                        "explanations": ["reason"],
                    }
                }
            ],
        }

        # This used to crash with Pydantic validation error
        result = normalize_memory_card(card_input)

        assert isinstance(result, MemoryCard)
        assert result.aliases == card_input["aliases"]
        assert isinstance(result.aliases[0], dict)
        assert "exp1-prog1" in result.aliases[0]
