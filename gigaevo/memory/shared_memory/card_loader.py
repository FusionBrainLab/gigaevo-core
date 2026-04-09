"""CardLoader utility for centralizing card I/O logic.

Extracted from write_pipeline.py and card_dedup.py to eliminate duplication.
Handles loading from JSONL export files, card store fallback, filtering, and error recovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.utils import _str_or_empty


class CardLoader:
    """Load and filter memory cards from export file or card store.

    Centralizes all card I/O logic. Handles:
    - Loading from JSONL export file
    - Fallback to card_store when export missing
    - Filtering (exclude programs, category filters)
    - Error recovery (malformed JSON)
    """

    def __init__(
        self,
        *,
        export_file: Path,
        card_store: CardStore | None = None,
        include_programs: bool = False,
        exclude_categories: set[str] | None = None,
    ):
        """Initialize CardLoader.

        Args:
            export_file: Path to JSONL export file.
            card_store: CardStore to use as fallback if export_file missing.
            include_programs: Whether to include 'program' category cards.
            exclude_categories: Additional categories to exclude.
        """
        self.export_file = export_file
        self.card_store = card_store
        self.include_programs = include_programs
        self.exclude_categories = exclude_categories or set()
        if not include_programs:
            self.exclude_categories.add("program")

    def load(self) -> list[dict[str, Any]]:
        """Load cards from export file or card store.

        Returns:
            List of card dicts, filtered and deduplicated.
        """
        if self.export_file.exists():
            try:
                cards = self._load_from_export()
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[CardLoader] Export file load failed: {}, using card_store",
                    exc,
                )
                cards = self._load_from_store()
        else:
            cards = self._load_from_store()

        # Apply filters
        filtered = self._apply_filters(cards)
        return filtered

    def _load_from_export(self) -> list[dict[str, Any]]:
        """Load cards from JSONL export file.

        Returns:
            List of card dicts parsed from JSONL.

        Raises:
            OSError: If file cannot be read.
            json.JSONDecodeError: If any line contains invalid JSON.
        """
        cards: list[dict[str, Any]] = []
        for line in self.export_file.read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
                if isinstance(card, dict):
                    cards.append(card)
            except json.JSONDecodeError:
                logger.debug("[CardLoader] Skipping malformed line in export: {}", line)
                continue
        return cards

    def _load_from_store(self) -> list[dict[str, Any]]:
        """Load cards from card_store.

        Returns:
            List of card dicts serialized from store, or empty list if no store.
        """
        if self.card_store is None:
            return []
        return [c.model_dump() for c in self.card_store.cards.values()]

    def _apply_filters(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply category filters and deduplication.

        Args:
            cards: Raw list of card dicts.

        Returns:
            Filtered list with duplicates removed and excluded categories removed.
        """
        seen = set()
        filtered = []
        for card in cards:
            card_id = _str_or_empty(card.get("id")).strip()
            if not card_id or card_id in seen:
                continue
            category = _str_or_empty(card.get("category")).strip().lower()
            if category in self.exclude_categories:
                continue
            seen.add(card_id)
            filtered.append(card)
        return filtered
