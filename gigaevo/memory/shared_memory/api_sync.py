"""Synchronizes cards between local CardStore and remote Memory API."""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import (
    AnyCard,
    build_entity_meta,
    card_to_concept_content,
    concept_to_card,
)
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.note_sync import NoteSync
from gigaevo.memory.shared_memory.utils import looks_like_uuid


class ApiSync:
    """Synchronizes cards between local CardStore and remote Memory API.

    Owns paginated fetch, full sync, and search-via-API operations.
    Does NOT own rebuild decisions — returns change flags so the
    orchestrator can decide.
    """

    def __init__(
        self,
        *,
        client: _ConceptApiClient,
        card_store: CardStore,
        note_sync: NoteSync | None,
        namespace: str,
        channel: str,
        author: str | None = None,
        sync_batch_size: int,
        search_limit: int,
    ):
        self.client = client
        self._card_store = card_store
        self._note_sync = note_sync
        self.namespace = namespace
        self.channel = channel
        self.author = author
        self.sync_batch_size = sync_batch_size
        self.search_limit = search_limit

    def fetch_all_hits(self) -> tuple[list[dict[str, Any]], bool]:
        """Paginated fetch of all concept hits, filtered by namespace.

        Returns (hits, pagination_complete). ``pagination_complete`` is True
        only when the final page was smaller than ``sync_batch_size`` (natural
        end-of-list).  False means we may have partial results.
        """
        hits: list[dict[str, Any]] = []
        offset = 0
        pagination_complete = False
        while True:
            try:
                rows = self.client.list_memory_cards(
                    limit=self.sync_batch_size,
                    offset=offset,
                    channel=self.channel,
                )
            except Exception as exc:
                logger.warning(
                    "[Memory] API pagination interrupted at offset {}: {}",
                    offset,
                    exc,
                )
                break
            if not rows:
                pagination_complete = True
                break

            page_hits: list[dict[str, Any]] = []
            for row in rows:
                meta = row.get("meta") if isinstance(row, dict) else None
                row_namespace = None
                if isinstance(meta, dict):
                    row_namespace = meta.get("namespace")
                if self.namespace and row_namespace not in (None, "", self.namespace):
                    continue
                page_hits.append(row)

            offset += len(rows)
            hits.extend(page_hits)

            if len(rows) < self.sync_batch_size:
                pagination_complete = True
                break
        return hits, pagination_complete

    def sync(self, force_full: bool = False) -> bool:
        """Sync from API. Returns True if local state changed.

        The caller is responsible for calling rebuild() or persist()
        based on the return value and its own state.
        """
        remote_hits, pagination_complete = self.fetch_all_hits()
        remote_entity_ids: set[str] = set()
        changed = False

        store = self._card_store

        for hit in remote_hits:
            entity_id = str(hit.get("entity_id") or "").strip()
            if not entity_id:
                continue
            remote_entity_ids.add(entity_id)
            remote_version = str(hit.get("version_id") or "").strip()

            known_card_id = store.card_id_by_entity.get(entity_id)
            known_version = store.entity_version.get(entity_id, "")
            if (
                not force_full
                and known_card_id
                and remote_version
                and known_version == remote_version
            ):
                # Version unchanged — but ensure note exists in A-MEM
                if (
                    self._note_sync is not None
                    and self._note_sync.memory_system.read(known_card_id) is None
                    and known_card_id in store.cards
                ):
                    if self._note_sync.upsert_fast(store.cards[known_card_id]):
                        changed = True
                store.note_ids.add(known_card_id)
                continue

            concept = self.client.get_concept(entity_id, channel=self.channel)
            content = concept.get("content") or {}
            fallback_id = store.card_id_by_entity.get(entity_id) or str(
                content.get("id") or entity_id
            )
            card = concept_to_card(content, fallback_id=fallback_id)
            card_id = store.ensure_id(card)

            previous_card_id = store.card_id_by_entity.get(entity_id)
            if previous_card_id and previous_card_id != card_id:
                store.cards.pop(previous_card_id, None)
                self._remove_note(previous_card_id)
                changed = True

            version = str(concept.get("version_id") or remote_version or "")
            store.save_entity(card_id, entity_id, version)

            old_card = store.cards.get(card_id)
            if old_card != card:
                changed = True
            store.cards[card_id] = card

            if self._upsert_note(card):
                changed = True

        # Remove entities no longer present on remote — only safe when
        # pagination completed fully (partial results would incorrectly
        # mark healthy entities as stale).
        if pagination_complete:
            stale_entities = set(store.card_id_by_entity) - remote_entity_ids
            for entity_id in stale_entities:
                stale_card_id = store.unlink_entity(entity_id)
                if stale_card_id and store.cards.pop(stale_card_id, None) is not None:
                    self._remove_note(stale_card_id)
                    changed = True

        return changed

    def search(
        self,
        query: str,
        memory_state: str | None = None,
    ) -> tuple[list[AnyCard], bool]:
        """Search API and update local state.

        Returns (cards, local_changed). The caller handles
        rebuild/persist and result formatting.
        """
        effective_query = query.strip()
        if memory_state:
            effective_query = f"{effective_query}\n{memory_state.strip()}"

        payload = self.client.search_concepts(
            query=effective_query,
            limit=self.search_limit,
            namespace=self.namespace,
            offset=0,
        )
        hits = payload.get("hits", [])
        if not hits:
            return [], False

        store = self._card_store
        cards: list[AnyCard] = []
        local_changed = False

        for hit in hits:
            entity_id = str(hit.get("entity_id") or "").strip()
            if not entity_id:
                continue

            concept = self.client.get_concept(entity_id, channel=self.channel)
            content = concept.get("content") or {}

            card_id = str(
                content.get("id") or store.card_id_by_entity.get(entity_id) or entity_id
            )
            card = concept_to_card(content, fallback_id=card_id)
            card_id = store.ensure_id(card)

            version = str(concept.get("version_id") or hit.get("version_id") or "")
            store.save_entity(card_id, entity_id, version)
            store.cards[card_id] = card
            cards.append(card)

            if self._upsert_note(card):
                local_changed = True

        return cards, local_changed

    # --- Card save / delete via API ---

    def save_card_to_api(self, card: AnyCard, card_id: str) -> None:
        """Save card to remote API and update entity mappings in card_store."""
        content = card_to_concept_content(card)
        name, tags, when_to_use = build_entity_meta(card)
        store = self._card_store
        response = self.client.save_concept(
            content=content,
            name=name,
            tags=tags,
            when_to_use=when_to_use,
            channel=self.channel,
            namespace=self.namespace,
            author=self.author,
            entity_id=store.entity_by_card_id.get(card_id),
        )
        store.save_entity(
            card_id,
            str(response["entity_id"]),
            str(response.get("version_id") or ""),
        )

    def delete_from_api(self, key: str) -> str | None:
        """Delete card from API by card_id or entity_id.

        Returns the resolved card_id if found and deleted, None otherwise.
        Falls back to local-only delete when card has no API entity mapping.
        """
        store = self._card_store
        entity_id = store.entity_by_card_id.get(key)
        if not entity_id and looks_like_uuid(key):
            entity_id = key
        if not entity_id:
            # No API entity — fall back to local resolution so local-only
            # cards (mem-XXXX) can still be deleted.
            return store.resolve_card_id(key)
        self.client.delete_concept(entity_id)
        return store.unlink_entity(entity_id) or key

    # --- Internal helpers ---

    def _upsert_note(self, card: AnyCard) -> bool:
        if self._note_sync is None:
            return False
        return self._note_sync.upsert_fast(card)

    def _remove_note(self, card_id: str) -> bool:
        if self._note_sync is not None:
            return self._note_sync.remove(card_id)
        self._card_store.note_ids.discard(card_id)
        return False
