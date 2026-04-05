"""Card index storage — owns card dicts, entity mappings, and disk persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import uuid

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import AnyCard, normalize_memory_card


class CardStore:
    """Owns memory cards, entity mappings, and JSON persistence.

    All card data lives here. Other components access cards via this store.
    """

    def __init__(self, *, index_file: Path):
        self._index_file = index_file

        self.cards: dict[str, AnyCard] = {}
        self.entity_by_card_id: dict[str, str] = {}
        self.card_id_by_entity: dict[str, str] = {}
        self.entity_version: dict[str, str] = {}
        self.note_ids: set[str] = set()
        self.write_stats: dict[str, int] = {
            "processed": 0,
            "added": 0,
            "rejected": 0,
            "updated": 0,
            "updated_target_cards": 0,
        }
        self._load()

    def get(self, card_id: str) -> AnyCard | None:
        return self.cards.get(card_id)

    def put(self, card_id: str, card: AnyCard) -> None:
        self.cards[card_id] = card

    def remove(self, card_id: str) -> AnyCard | None:
        return self.cards.pop(card_id, None)

    def ensure_id(self, card: AnyCard) -> str:
        card_id = str(card.id or "").strip()
        if not card_id:
            card_id = f"mem-{uuid.uuid4().hex[:12]}"
            card.id = card_id
        return card_id

    def serialize_all(self) -> dict[str, dict[str, Any]]:
        return {cid: c.model_dump() for cid, c in self.cards.items()}

    def persist(self, serialized: dict[str, dict[str, Any]] | None = None) -> None:
        if serialized is None:
            serialized = self.serialize_all()
        payload = {
            "entity_by_card_id": self.entity_by_card_id,
            "entity_version_by_entity": self.entity_version,
            "memory_cards": serialized,
        }
        tmp_file = self._index_file.with_suffix(f".{os.getpid()}.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_file), str(self._index_file))

    def link_entity(self, card_id: str, entity_id: str, version: str = "") -> None:
        self.entity_by_card_id[card_id] = entity_id
        self.card_id_by_entity[entity_id] = card_id
        self.entity_version[entity_id] = version

    def unlink_entity(self, entity_id: str) -> str | None:
        card_id = self.card_id_by_entity.pop(entity_id, None)
        self.entity_version.pop(entity_id, None)
        if card_id:
            self.entity_by_card_id.pop(card_id, None)
        return card_id

    def get_entity_for_card(self, card_id: str) -> str | None:
        return self.entity_by_card_id.get(card_id)

    def get_card_for_entity(self, entity_id: str) -> str | None:
        return self.card_id_by_entity.get(entity_id)

    def save_entity(self, card_id: str, entity_id: str, version: str = "") -> None:
        """Link card to entity, cleaning up stale mappings if entity_id changed."""
        old = self.entity_by_card_id.get(card_id)
        if old and old != entity_id:
            self.card_id_by_entity.pop(old, None)
            self.entity_version.pop(old, None)
        # Clean up old card that previously owned this entity
        prev_card = self.card_id_by_entity.get(entity_id)
        if prev_card and prev_card != card_id:
            self.entity_by_card_id.pop(prev_card, None)
        self.link_entity(card_id, entity_id, version)

    def clear_entity(self, card_id: str) -> str | None:
        """Remove entity mapping for a card. Returns the old entity_id."""
        entity_id = self.entity_by_card_id.pop(card_id, None)
        if entity_id:
            self.card_id_by_entity.pop(entity_id, None)
            self.entity_version.pop(entity_id, None)
        return entity_id

    def resolve_card_id(self, key: str) -> str | None:
        """Resolve a key (card_id or entity_id) to a card_id in the store."""
        if key in self.cards:
            return key
        mapped = self.card_id_by_entity.get(key)
        if mapped and mapped in self.cards:
            return mapped
        return None

    def _load(self) -> None:
        if not self._index_file.exists():
            return

        try:
            payload = json.loads(self._index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[Memory] Could not parse index file {}: {}", self._index_file, exc
            )
            return

        raw_cards = payload.get("memory_cards", {})
        raw_map = payload.get("entity_by_card_id", {})
        raw_versions = payload.get("entity_version_by_entity", {})

        if isinstance(raw_cards, dict):
            for card_id, card in raw_cards.items():
                cid = str(card_id)
                self.cards[cid] = normalize_memory_card(card, fallback_id=cid)
                self.note_ids.add(cid)

        if isinstance(raw_map, dict):
            for card_id, entity_id in raw_map.items():
                cid = str(card_id)
                eid = str(entity_id)
                if not cid or not eid:
                    continue
                if cid not in self.cards:
                    logger.debug(
                        "[Memory] Skipping dangling entity mapping: "
                        "card_id={!r} not in cards",
                        cid,
                    )
                    continue
                self.entity_by_card_id[cid] = eid
                self.card_id_by_entity[eid] = cid

        if isinstance(raw_versions, dict):
            for entity_id, version_id in raw_versions.items():
                eid = str(entity_id)
                vid = str(version_id or "")
                if eid:
                    self.entity_version[eid] = vid
