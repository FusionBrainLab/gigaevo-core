"""ChromaNeighborSource: the librarian's nearest-card primitive.

The write path needs "closest existing cards to this text" for the pre-gate
near-duplicate check and to ground the reconcile LLM. The reader already indexes
every card in the shared A-MEM Chroma collection — cards sync into it on every
admit/merge (``note_sync`` -> ``retriever.add_document``) — so we reuse that one
populated store instead of re-embedding the bank and hand-rolling cosine.

The collection's ``retriever.search(query, k)`` returns ids ordered by ascending
distance; its embeddings are unit-normalized, so the returned distance is exactly
``1 - cosine_similarity`` (verified: identical to the prior hand-rolled cosine).
We query it and map the returned ids back to live cards, treating the card bank
as the source of truth — an id Chroma still indexes but the bank has dropped
(deleted, not yet re-indexed) is skipped, never fabricated.
"""

from __future__ import annotations

from typing import Any

from gigaevo.memory.shared_memory.card_conversion import AnyCard


class ChromaNeighborSource:
    def __init__(self, store: Any) -> None:
        self._store = store

    def nearest(self, note: str, k: int) -> list[tuple[AnyCard, float]]:
        text = (note or "").strip()
        if not text or k <= 0:
            return []
        memory_system = self._store.memory_system
        if memory_system is None:
            return []
        results = memory_system.retriever.search(text, k)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        cards = self._store.card_store.cards
        out: list[tuple[AnyCard, float]] = []
        for card_id, distance in zip(ids, distances):
            card = cards.get(card_id)
            if card is not None:
                out.append((card, float(distance)))
        return out
