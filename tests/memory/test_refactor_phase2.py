"""Tests for memory system refactoring Phase 2: CardLoader utility.

Task 1: Extract card loading utilities into CardLoader class.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card


class TestCardLoader:
    """Tests for CardLoader utility class."""

    def test_load_from_export_file(self, tmp_path):
        """CardLoader loads cards from JSONL export file."""
        # Import here to trigger fail if CardLoader doesn't exist yet
        from gigaevo.memory.shared_memory.card_loader import CardLoader

        export_file = tmp_path / "export.jsonl"
        # Write 2 cards to JSONL
        card1 = {"id": "c1", "description": "idea 1", "category": "general"}
        card2 = {"id": "c2", "description": "idea 2", "category": "general"}
        export_file.write_text(f"{json.dumps(card1)}\n{json.dumps(card2)}\n")

        loader = CardLoader(export_file=export_file)
        cards = loader.load()

        assert len(cards) == 2
        assert cards[0]["id"] == "c1"
        assert cards[1]["id"] == "c2"

    def test_load_from_card_store(self, tmp_path):
        """CardLoader falls back to card_store.cards when export missing."""
        from gigaevo.memory.shared_memory.card_loader import CardLoader

        loader = CardLoader(
            export_file=tmp_path / "missing.jsonl",
            card_store=MagicMock(
                cards={
                    "c1": normalize_memory_card(
                        {"id": "c1", "description": "idea", "category": "general"}
                    ),
                    "c2": normalize_memory_card(
                        {"id": "c2", "description": "idea2", "category": "general"}
                    ),
                }
            ),
        )

        cards = loader.load()
        assert len(cards) == 2

    def test_filter_program_cards_excluded(self, tmp_path):
        """Load excludes program category cards."""
        from gigaevo.memory.shared_memory.card_loader import CardLoader

        export_file = tmp_path / "export.jsonl"
        idea = {"id": "idea1", "description": "general idea", "category": "general"}
        program = {"id": "prog1", "description": "program", "category": "program"}
        export_file.write_text(f"{json.dumps(idea)}\n{json.dumps(program)}\n")

        loader = CardLoader(export_file=export_file, include_programs=False)
        cards = loader.load()

        assert len(cards) == 1
        assert cards[0]["category"] == "general"

    def test_load_handles_malformed_json(self, tmp_path):
        """Load recovers from malformed lines in export file."""
        from gigaevo.memory.shared_memory.card_loader import CardLoader

        export_file = tmp_path / "export.jsonl"
        export_file.write_text("not json\n{valid}\n")

        loader = CardLoader(export_file=export_file)
        cards = loader.load()  # Should not raise

        assert isinstance(cards, list)
