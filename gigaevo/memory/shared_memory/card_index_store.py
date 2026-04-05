"""Local card index management — owns dicts, persistence, and card ID generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import uuid

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import AnyCard, normalize_memory_card


class CardIndexStore:
    """Manages the 6 coupled card-index dicts and their persistence to disk.

    Owns:
    - memory_cards: dict[str, AnyCard]
    - entity_by_card_id: dict[str, str]
    - card_id_by_entity: dict[str, str]
    - entity_version_by_entity: dict[str, str]
    - memory_ids: set[str]
    - card_write_stats: dict[str, int]

    Methods:
    - _load_index: deserialize from JSON file
    - _serialize_cards: convert cards to dicts
    - _persist_index: write to JSON file
    - _ensure_card_id: generate UUID if missing
    """

    def __init__(self, index_file: Path):
        """Initialize empty index."""
        self.index_file = index_file

        self.memory_cards: dict[str, AnyCard] = {}
        self.entity_by_card_id: dict[str, str] = {}
        self.card_id_by_entity: dict[str, str] = {}
        self.entity_version_by_entity: dict[str, str] = {}
        self.memory_ids: set[str] = set()
        self.card_write_stats: dict[str, int] = {
            "processed": 0,
            "added": 0,
            "rejected": 0,
            "updated": 0,
            "updated_target_cards": 0,
        }

        self._load_index()

    def _load_index(self) -> None:
        """Deserialize cards and mappings from JSON file."""
        if not self.index_file.exists():
            return

        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[Memory] Could not parse index file {}: {}", self.index_file, exc
            )
            return

        raw_cards = payload.get("memory_cards", {})
        raw_map = payload.get("entity_by_card_id", {})
        raw_versions = payload.get("entity_version_by_entity", {})

        if isinstance(raw_cards, dict):
            for card_id, card in raw_cards.items():
                cid = str(card_id)
                self.memory_cards[cid] = normalize_memory_card(card, fallback_id=cid)
                self.memory_ids.add(cid)

        if isinstance(raw_map, dict):
            for card_id, entity_id in raw_map.items():
                cid = str(card_id)
                eid = str(entity_id)
                if not cid or not eid:
                    continue
                self.entity_by_card_id[cid] = eid
                self.card_id_by_entity[eid] = cid

        if isinstance(raw_versions, dict):
            for entity_id, version_id in raw_versions.items():
                eid = str(entity_id)
                vid = str(version_id or "")
                if eid:
                    self.entity_version_by_entity[eid] = vid

    def _serialize_cards(self) -> dict[str, dict[str, Any]]:
        """Convert all in-memory cards to dicts for persistence."""
        return {cid: c.model_dump() for cid, c in self.memory_cards.items()}

    def _persist_index(
        self, serialized_cards: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Write index to disk with optional pre-serialized cards (to avoid double serialization)."""
        if serialized_cards is None:
            serialized_cards = self._serialize_cards()

        payload = {
            "entity_by_card_id": self.entity_by_card_id,
            "entity_version_by_entity": self.entity_version_by_entity,
            "memory_cards": serialized_cards,
        }
        tmp_file = self.index_file.with_suffix(f".{os.getpid()}.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_file), str(self.index_file))

    def _ensure_card_id(self, card: AnyCard) -> str:
        """Ensure card has a valid ID, generate one if missing."""
        card_id = str(card.id or "").strip()
        if not card_id:
            card_id = f"mem-{uuid.uuid4().hex[:12]}"
            card.id = card_id
        return card_id
