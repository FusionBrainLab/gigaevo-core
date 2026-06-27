"""Tests for ChromaNeighborSource: nearest cards via the reader's A-MEM Chroma.

The write path reuses the *same* populated Chroma collection the reader queries.
We fake the collection's ``retriever.search(query, k)`` (chromadb's result shape:
parallel ``ids``/``distances`` lists, ascending distance) and assert the source
maps ids back to live cards — never re-embedding, never hand-rolling cosine.
"""

from __future__ import annotations

from typing import Any

from gigaevo.memory.ideas_tracker.librarian_retriever import ChromaNeighborSource
from gigaevo.memory.shared_memory.models import MemoryCard


class _FakeRetriever:
    """Twin of A-MEM ChromaRetriever.search over a fixed id->distance ranking."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self._ranking = ranking
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> dict[str, Any]:
        self.calls.append((query, k))
        top = self._ranking[:k]
        return {
            "ids": [[cid for cid, _ in top]],
            "distances": [[dist for _, dist in top]],
        }


class _FakeMemorySystem:
    def __init__(self, retriever: _FakeRetriever) -> None:
        self.retriever = retriever


class _FakeCardStore:
    def __init__(self, cards: dict) -> None:
        self.cards = cards


class _FakeStore:
    def __init__(self, cards: dict, retriever: _FakeRetriever | None) -> None:
        self.card_store = _FakeCardStore(cards)
        self.memory_system = (
            _FakeMemorySystem(retriever) if retriever is not None else None
        )


def _card(cid: str) -> MemoryCard:
    return MemoryCard(id=cid, description=f"lever {cid}", keywords=[])


def _store_with(ranking: list[tuple[str, float]], cards) -> _FakeStore:  # noqa: ANN001
    return _FakeStore({c.id: c for c in cards}, _FakeRetriever(ranking))


def test_nearest_maps_search_ids_to_live_cards_in_order() -> None:
    near, far = _card("mem-near"), _card("mem-far")
    store = _store_with([("mem-near", 0.02), ("mem-far", 0.81)], [far, near])
    src = ChromaNeighborSource(store)
    hits = src.nearest("query note", 5)
    assert [c.id for c, _ in hits] == ["mem-near", "mem-far"]
    assert [round(d, 2) for _, d in hits] == [0.02, 0.81]
    assert hits[0][1] < hits[1][1]


def test_nearest_passes_k_to_the_retriever() -> None:
    near, far = _card("mem-near"), _card("mem-far")
    store = _store_with([("mem-near", 0.02), ("mem-far", 0.81)], [near, far])
    src = ChromaNeighborSource(store)
    hits = src.nearest("query note", 1)
    assert [c.id for c, _ in hits] == ["mem-near"]
    assert store.memory_system.retriever.calls == [("query note", 1)]


def test_stale_ids_absent_from_bank_are_skipped() -> None:
    near = _card("mem-near")
    # Chroma still indexes 'mem-gone' (deleted but not yet re-indexed); the bank
    # is the source of truth, so an unmappable id is dropped, not faked.
    store = _store_with([("mem-gone", 0.01), ("mem-near", 0.30)], [near])
    src = ChromaNeighborSource(store)
    hits = src.nearest("query note", 5)
    assert [c.id for c, _ in hits] == ["mem-near"]


def test_blank_note_returns_empty_without_querying() -> None:
    store = _store_with([("mem-near", 0.02)], [_card("mem-near")])
    src = ChromaNeighborSource(store)
    assert src.nearest("   ", 5) == []
    assert store.memory_system.retriever.calls == []


def test_missing_agentic_memory_system_returns_empty() -> None:
    store = _FakeStore({"mem-near": _card("mem-near")}, retriever=None)
    src = ChromaNeighborSource(store)
    assert src.nearest("query note", 5) == []
