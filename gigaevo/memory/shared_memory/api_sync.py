"""Synchronizes cards between local CardStore and remote Memory API."""

from __future__ import annotations

from typing import Any

from gigaevo.memory.shared_memory.card_conversion import AnyCard, concept_to_card
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.note_sync import NoteSync


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
        sync_batch_size: int,
        search_limit: int,
    ):
        self.client = client
        self._card_store = card_store
        self._note_sync = note_sync
        self.namespace = namespace
        self.channel = channel
        self.sync_batch_size = sync_batch_size
        self.search_limit = search_limit

    def fetch_all_hits(self) -> list[dict[str, Any]]:
        """Paginated fetch of all concept hits, filtered by namespace."""
        hits: list[dict[str, Any]] = []
        offset = 0
        while True:
            rows = self.client.list_memory_cards(
                limit=self.sync_batch_size,
                offset=offset,
                channel=self.channel,
            )
            if not rows:
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
                break
        return hits

    def sync(self, force_full: bool = False) -> bool:
        """Sync from API. Returns True if local state changed.

        The caller is responsible for calling rebuild() or persist()
        based on the return value and its own state.
        """
        remote_hits = self.fetch_all_hits()
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

        # Remove entities no longer present on remote
        stale_entities = set(store.card_id_by_entity) - remote_entity_ids
        for entity_id in stale_entities:
            card_id = store.unlink_entity(entity_id)
            if card_id:
                store.cards.pop(card_id, None)
                self._remove_note(card_id)
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

            version = str(
                concept.get("version_id") or hit.get("version_id") or ""
            )
            store.save_entity(card_id, entity_id, version)
            store.cards[card_id] = card
            cards.append(card)

            if self._upsert_note(card):
                local_changed = True

        return cards, local_changed

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
