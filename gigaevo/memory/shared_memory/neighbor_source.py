"""ChromaNeighborSource: the A-MEM backend's nearest-card primitive.

The write path needs "closest existing cards to this text" for the pre-gate
near-duplicate check and to ground the reconcile LLM. The reader already indexes
every card in the shared A-MEM Chroma collection — cards sync into it on every
admit/merge (``note_sync`` -> ``retriever.add_document``) — so we reuse that one
populated store instead of re-embedding the bank and hand-rolling cosine.

This lives with the backend (not the write path): it reaches into the backend's
A-MEM ``memory_system.retriever``, so it is an ``AmemGamMemory`` implementation
detail. The backend exposes it through the ``nearest`` contract method, and the
write path depends only on that (the librarian's ``NeighborSource``).

The collection's ``retriever.search(query, k)`` returns ids ordered by ascending
distance. We query it and map the returned ids back to live cards, treating the
card bank as the source of truth — an id Chroma still indexes but the bank has
dropped (deleted, not yet re-indexed) is skipped, never fabricated.
"""

from __future__ import annotations

from typing import Any

from gigaevo.memory.shared_memory.models import CardT


class ChromaNeighborSource:
    def __init__(self, store: Any) -> None:
        self._store = store

    def nearest(
        self, note: str, k: int, card_type: type[CardT]
    ) -> list[tuple[CardT, float]]:
        """The k nearest cards of ``card_type`` (idea dedup and the consolidation
        sweep want ``MemoryCard``; exemplar twin dedup wants ``ProgramCard``).

        Idea and program cards share one A-MEM index, but each consumer wants
        exactly one kind. A top-k crowded with the other kind would starve the
        wanted neighbors, so over-fetch (doubling) until k of ``card_type``
        surface or Chroma is exhausted, then return the k closest.
        """
        text = (note or "").strip()
        if not text or k <= 0:
            return []
        memory_system = self._store.memory_system
        if memory_system is None:
            return []
        fetch = k
        while True:
            results = memory_system.retriever.search(text, fetch)
            ids = (results.get("ids") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            out: list[tuple[CardT, float]] = []
            for card_id, distance in zip(ids, distances):
                card = self._store.get_card(card_id)
                if isinstance(card, card_type):
                    out.append((card, float(distance)))
            if len(out) >= k or len(ids) < fetch:
                return out[:k]
            fetch *= 2
