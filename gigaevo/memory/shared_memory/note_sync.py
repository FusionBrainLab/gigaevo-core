"""Synchronizes memory cards with local A-MEM vector store (Chroma)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import (
    AnyCard,
    MemoryNoteProtocol,
    export_memories_jsonl,
    note_metadata,
)
from gigaevo.memory.shared_memory.card_store import CardStore


class NoteSync:
    """Bridges memory cards and the local A-MEM vector store.

    Receives ``memory_system`` and ``note_cls`` at construction (injected).
    Reads card data from the shared ``CardStore`` reference.
    """

    def __init__(
        self,
        *,
        memory_system: Any,
        note_cls: type[Any],
        card_store: CardStore,
    ):
        self.memory_system = memory_system
        self._note_cls = note_cls
        self._card_store = card_store

    def build_note(
        self,
        card: AnyCard,
        existing: MemoryNoteProtocol | None = None,
    ) -> MemoryNoteProtocol:
        card_id = str(card.id or "")
        description = str(card.description or "")
        context = str(
            card.task_description or card.task_description_summary or "General"
        )
        category = str(card.category or "general")
        strategy = str(card.strategy or "")
        keywords = list(card.keywords or [])
        links = list(card.links or [])
        if existing is None:
            existing = self.memory_system.read(card_id)

        return self._note_cls(
            content=description,
            id=card_id,
            keywords=keywords,
            links=links,
            retrieval_count=(existing.retrieval_count if existing is not None else 0),
            timestamp=(existing.timestamp if existing is not None else None),
            last_accessed=(existing.last_accessed if existing is not None else None),
            context=context or "General",
            evolution_history=(
                existing.evolution_history if existing is not None else None
            ),
            category=category,
            tags=(existing.tags if existing is not None else []),
            strategy=strategy,
        )

    @staticmethod
    def fields_changed(
        existing: MemoryNoteProtocol,
        content: Any,
        category: Any,
        context: Any,
        strategy: Any,
        keywords: Any,
        links: Any,
    ) -> bool:
        return (
            existing.content != content
            or existing.category != category
            or existing.context != context
            or existing.strategy != strategy
            or existing.keywords != keywords
            or existing.links != links
        )

    def upsert_fast(self, card: AnyCard) -> bool:
        """Synchronize card into local A-MEM/Chroma without LLM evolution."""
        card_id = str(card.id or "")
        existing = self.memory_system.read(card_id)
        note = self.build_note(card, existing=existing)
        changed = existing is None or self.fields_changed(
            existing,
            note.content,
            note.category,
            note.context,
            note.strategy,
            note.keywords,
            note.links,
        )
        if not changed:
            self._card_store.note_ids.add(note.id)
            return False

        self.memory_system.memories[note.id] = note
        try:
            self.memory_system.retriever.delete_document(note.id)
        except Exception as exc:
            logger.warning(
                "[Memory] Failed to delete document {!r} before re-add: {}",
                note.id,
                exc,
            )
        self.memory_system.retriever.add_document(
            self.memory_system._document_for_note(note),
            note_metadata(note),
            note.id,
        )
        self._card_store.note_ids.add(note.id)
        return True

    def upsert_agentic(self, card: AnyCard) -> bool:
        """Add/update card in local A-MEM using the agentic add/update path."""
        card_id = str(card.id or "").strip()
        if not card_id:
            return False

        description = str(card.description or "")
        kwargs = {
            "category": str(card.category or "general"),
            "keywords": list(card.keywords or []),
            "context": str(
                card.task_description or card.task_description_summary or "General"
            ),
            "strategy": str(card.strategy or ""),
            "links": list(card.links or []),
            "tags": [],
        }

        existing = self.memory_system.read(card_id)
        if existing is None:
            self.memory_system.add_note(id=card_id, content=description, **kwargs)
        else:
            changed = self.fields_changed(
                existing,
                description,
                kwargs["category"],
                kwargs["context"],
                kwargs["strategy"],
                kwargs["keywords"],
                kwargs["links"],
            )
            if not changed:
                self._card_store.note_ids.add(card_id)
                return False
            self.memory_system.update(card_id, content=description, **kwargs)

        self._card_store.note_ids.add(card_id)
        return True

    def remove(self, card_id: str) -> bool:
        deleted = self.memory_system.delete(card_id)
        self._card_store.note_ids.discard(card_id)
        return deleted

    def export_jsonl(
        self,
        out_path: Path,
        serialized_cards: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        all_ids = sorted(
            set(self._card_store.note_ids) | set(self._card_store.cards.keys())
        )
        if serialized_cards is None:
            serialized_cards = self._card_store.serialize_all()
        export_memories_jsonl(
            self.memory_system,
            all_ids,
            out_path,
            card_overrides=serialized_cards,
        )
