"""End-to-end pipeline test: ideas_tracker → normalize_memory_card → load_memory_cards.

Covers the typed integration boundary between the tracker's serialized cards
and the write pipeline: alias version history validates into CardAlias, and
load_memory_cards preserves the full card metadata.
"""

import json
from pathlib import Path

from gigaevo.memory.shared_memory.card_conversion import (
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.models import CardAlias, MemoryCard, ProgramCard
from gigaevo.memory.write_pipeline import classify_card_type, load_memory_cards


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
    """Create a mock ideas_tracker output card with CardAlias version history."""
    aliases = []
    if has_version_history:
        aliases.append(
            {
                "key": "exp1-prog1",
                "description": f"{description} (initial)",
                "programs": ["p1"],
                "explanations": ["Found this pattern in early runs"],
            }
        )
        aliases.append(
            {
                "key": "exp1-prog2",
                "description": f"{description} (refined)",
                "programs": ["p1", "p2"],
                "explanations": [
                    "Found this pattern in early runs",
                    "Refined after testing",
                ],
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
        "explanation": {
            "explanations": ["Found effective chunking strategy"],
            "summary": "Improved retrieval via adaptive chunking",
        },
        "works_with": ["idea-2", "idea-3"],
        "links": ["related-concept-1"],
    }


# ===========================================================================
# normalize_memory_card — ideas_tracker input
# ===========================================================================


class TestNormalizeWithIdeasTrackerOutput:
    """Test normalize_memory_card with realistic ideas_tracker output shapes."""

    def test_normalize_card_with_alias_history(self):
        """Alias version history validates into typed CardAlias entries."""
        card = make_ideas_tracker_card(
            "idea-1", "Retrieval chunking", has_version_history=True
        )
        result = normalize_memory_card(card)

        assert isinstance(result, MemoryCard)
        assert result.id == "idea-1"
        assert result.description == "Retrieval chunking"
        assert len(result.aliases) == 2
        assert isinstance(result.aliases[0], CardAlias)
        assert result.aliases[0].key == "exp1-prog1"
        assert result.aliases[1].programs == ["p1", "p2"]

    def test_preserve_explanation_structure(self):
        """explanation dict with explanations list and summary preserved."""
        card = make_ideas_tracker_card(
            "idea-3", "Token strategy", has_version_history=False
        )
        result = normalize_memory_card(card)

        assert result.explanation.explanations == ["Found effective chunking strategy"]
        assert result.explanation.summary == "Improved retrieval via adaptive chunking"

    def test_full_roundtrip_preserves_all_fields(self):
        """Complete card with all complex nested structures."""
        card = make_ideas_tracker_card(
            "idea-full", "Complete idea", has_version_history=True
        )
        result = normalize_memory_card(card)

        assert result.id == "idea-full"
        assert result.category == "general"
        assert len(result.aliases) == 2
        assert isinstance(result.aliases[0], CardAlias)
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

        # Load and validate
        cards = load_memory_cards(banks_path)
        assert len(cards) == 2

        card1 = cards[0]
        assert isinstance(card1, MemoryCard)
        assert card1.id == "idea-1"
        assert len(card1.aliases) == 2
        assert isinstance(card1.aliases[0], CardAlias)
        assert card1.description == "Chunking"

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

        programs_path = tmp_path / "programs.json"
        _write_json(
            programs_path,
            [
                {
                    "programs": [
                        {
                            "id": "prog-1",
                            "fitness": 85.5,
                            "is_valid": 1.0,
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

    def test_e2e_empty_active_bank_yields_no_cards(self, tmp_path):
        """An empty active bank seeds nothing — no ghost cards."""
        banks_path = tmp_path / "banks.json"
        _write_json(banks_path, [{"active_bank": []}])

        cards = load_memory_cards(banks_path)
        assert cards == []


# ===========================================================================
# Full main() loop simulation
# ===========================================================================


class TestMainLoopSimulation:
    """Simulate the main() loop: load_memory_cards → classify_card_type for each card."""

    def test_full_loop_ideas_only(self, tmp_path):
        """Simulate main() loop with idea cards from ideas_tracker."""
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
        cards = load_memory_cards(banks_path)
        for card in cards:
            card_type = classify_card_type(card)
            assert card_type == "ideas"
            assert isinstance(card, MemoryCard)

    def test_full_loop_mixed_ideas_and_programs(self, tmp_path):
        """Simulate main() loop with both idea and program cards."""
        banks_path = tmp_path / "banks.json"
        _write_json(
            banks_path,
            [
                {
                    "active_bank": [
                        make_ideas_tracker_card(
                            "idea-1", "Good idea", has_version_history=True
                        ),
                    ]
                }
            ],
        )
        programs_path = tmp_path / "programs.json"
        _write_json(
            programs_path,
            [
                {
                    "programs": [
                        {
                            "id": "prog-1",
                            "fitness": 85.5,
                            "is_valid": 1.0,
                            "code": "def f(): pass",
                            "task_description_summary": "Task",
                        }
                    ]
                }
            ],
        )

        cards = load_memory_cards(
            banks_path,
            programs_path=programs_path,
            best_programs_percent=100.0,
        )

        # Simulate main() loop
        type_counts = {"ideas": 0, "programs": 0}
        for card in cards:
            card_type = classify_card_type(card)
            type_counts[card_type] += 1

        assert type_counts["ideas"] == 1
        assert type_counts["programs"] == 1


# ---------------------------------------------------------------------------
# Task 7: write_pipeline.main() full loop with file I/O
# ---------------------------------------------------------------------------


def test_write_pipeline_main_full_loop(tmp_path):
    """Full-loop: write banks.json → call main() → memory gets cards."""
    import json
    from unittest.mock import MagicMock

    from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
    from gigaevo.memory.write_pipeline import main

    # Use the same active_bank format that load_memory_cards expects
    banks_data = [
        {
            "active_bank": [
                {
                    "id": "idea-1",
                    "description": "Use beam search for decoding",
                    "category": "general",
                    "keywords": ["beam", "search"],
                    "last_generation": 3,
                }
            ]
        }
    ]
    banks_file = tmp_path / "banks.json"
    memory_dir = tmp_path / "memory"

    banks_file.write_text(json.dumps(banks_data))
    memory_dir.mkdir()

    saved_cards = []

    class FakeMemory:
        def __init__(self, **kwargs):
            pass

        def save_card(self, card):
            saved_cards.append(card)
            return getattr(card, "id", "fake-id")

        def get_card(self, mid):
            return None

        def search(self, query, **kw):
            return ""

        def delete(self, mid):
            return True

        def rebuild(self):
            pass

        def sweep_harmful(self):
            return []

        def close(self):
            pass

        def get_card_write_stats(self):
            return {
                "processed": len(saved_cards),
                "added": len(saved_cards),
                "updated": 0,
                "rejected": 0,
                "updated_target_cards": 0,
            }

    factory = MagicMock(spec=LocalMemoryBackendFactory)
    factory.build.return_value = FakeMemory()

    result = main(
        banks_path=banks_file,
        backend=factory,
        checkpoint_dir=memory_dir,
    )

    factory.build.assert_called_once_with(
        checkpoint_dir=memory_dir, evictor=None, deduplicator=None
    )

    assert len(saved_cards) >= 1, f"Expected at least 1 card written, got {saved_cards}"
    card_ids = [getattr(c, "id", None) for c in saved_cards]
    assert "idea-1" in card_ids, f"Expected idea-1 in saved cards, got {card_ids}"
    assert result is not None


def test_write_pipeline_main_swallows_llm_error(tmp_path):
    """A dedup-time LLM failure during save_card must not crash the write loop.

    The dedup deduplicator calls an LLM; its transient failures raise LLMError
    subclasses. main() catches them like any other write failure and returns
    None rather than propagating and taking down the whole write stage.
    """
    import json
    from unittest.mock import MagicMock

    from gigaevo.exceptions import LLMAPIError
    from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
    from gigaevo.memory.write_pipeline import main

    banks_data = [
        {
            "active_bank": [
                {
                    "id": "idea-1",
                    "description": "Use beam search for decoding",
                    "category": "general",
                    "keywords": ["beam", "search"],
                    "last_generation": 3,
                }
            ]
        }
    ]
    banks_file = tmp_path / "banks.json"
    memory_dir = tmp_path / "memory"
    banks_file.write_text(json.dumps(banks_data))
    memory_dir.mkdir()

    closed = []

    class FakeMemory:
        def save_card(self, card):
            raise LLMAPIError("dedup LLM call timed out")

        def get_card(self, mid):
            return None

        def get_card_write_stats(self):
            return {
                "processed": 0,
                "added": 0,
                "updated": 0,
                "rejected": 0,
                "updated_target_cards": 0,
            }

        def sweep_harmful(self):
            return []

        def rebuild(self):
            pass

        def close(self):
            closed.append(True)

    factory = MagicMock(spec=LocalMemoryBackendFactory)
    factory.build.return_value = FakeMemory()

    result = main(
        banks_path=banks_file,
        backend=factory,
        checkpoint_dir=memory_dir,
    )

    assert result is None
    assert closed == [True], "memory.close() must still run via finally"
