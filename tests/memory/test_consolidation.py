"""Behavior tests for the periodic consolidation pass.

``consolidate`` runs the same NeighborSource nearest-card primitive over the
whole bank and folds each near-dup pair into a single canonical card via the
gate. We fake every collaborator and assert on the bank state and the merge
count, never on internal calls. The fake NeighborSource is faithful: it ranks
the *live* store (deleted cards disappear) and encodes proximity in a group
prefix on the description, so two cards collapse iff they share a group.
"""

from __future__ import annotations

import pytest

from gigaevo.llm.agents.reconcile import LibrarianCard
from gigaevo.memory.ideas_tracker.consolidation import consolidate
from gigaevo.memory.shared_memory.models import MemoryCard


class _FakeCardStore:
    def __init__(self) -> None:
        self.cards: dict[str, MemoryCard] = {}


class _FakeStore:
    def __init__(self) -> None:
        self.card_store = _FakeCardStore()

    def delete(self, card_id: str) -> bool:
        return self.card_store.cards.pop(card_id, None) is not None


class _FakeGate:
    """Twin of CardAdmissionGate.merge: overwrites the target card in place."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store
        self.merged: list[tuple[str, MemoryCard]] = []

    def merge(self, target_id: str, card: MemoryCard) -> str:
        if target_id not in self._store.card_store.cards:
            return ""
        self._store.card_store.cards[target_id] = card
        self.merged.append((target_id, card))
        return target_id


class _FakeNeighbors:
    """Ranks the live store by a group prefix on the description.

    Same group -> near (0.01); different group -> far (0.9). Mirrors the real
    ChromaNeighborSource: ranks every card incl. the query itself, ascending.
    """

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    def nearest(self, note: str, k: int) -> list[tuple[MemoryCard, float]]:
        group = note.split(":", 1)[0]
        scored = [
            (c, 0.01 if (c.description or "").split(":", 1)[0] == group else 0.9)
            for c in self._store.card_store.cards.values()
            if (c.description or "").strip()
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]


class _FakeMergeAgent:
    """Synthesizes union prose, keeping the survivor's group prefix."""

    def __init__(self) -> None:
        self.calls = 0

    async def arun(self, *, card_a, card_b):  # noqa: ANN001
        self.calls += 1
        group = (card_a.description or "").split(":", 1)[0]
        return LibrarianCard(description=f"{group}: union prose", keywords=["union"])


def _card(cid: str, group: str, programs: list[str] | None = None) -> MemoryCard:
    return MemoryCard(
        id=cid,
        description=f"{group}: lever {cid}",
        keywords=[group],
        programs=programs or [cid],
    )


def _stack(*cards: MemoryCard):
    store = _FakeStore()
    for c in cards:
        store.card_store.cards[c.id] = c
    return store, _FakeGate(store), _FakeNeighbors(store), _FakeMergeAgent()


@pytest.mark.asyncio
async def test_two_near_dups_collapse_into_one_canonical_card() -> None:
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])
    )
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 1
    assert agent.calls == 1
    assert "mem-b" not in store.card_store.cards
    survivor = store.card_store.cards["mem-a"]
    assert survivor.description == "g1: union prose"
    assert set(survivor.programs) == {"p1", "p2"}


@pytest.mark.asyncio
async def test_single_card_bank_is_a_no_op() -> None:
    store, gate, neighbors, agent = _stack(_card("mem-a", "g1"))
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 0
    assert agent.calls == 0
    assert "mem-a" in store.card_store.cards


@pytest.mark.asyncio
async def test_distinct_cards_within_eps_guard_are_not_merged() -> None:
    store, gate, neighbors, agent = _stack(_card("mem-a", "g1"), _card("mem-b", "g2"))
    merges = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert merges == 0
    assert agent.calls == 0
    assert set(store.card_store.cards) == {"mem-a", "mem-b"}


@pytest.mark.asyncio
async def test_second_run_on_deduped_bank_returns_zero() -> None:
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]), _card("mem-b", "g1", ["p2"])
    )
    first = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    second = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    assert first == 1
    assert second == 0
    assert set(store.card_store.cards) == {"mem-a"}


@pytest.mark.asyncio
async def test_three_mutual_near_dups_collapse_to_one() -> None:
    store, gate, neighbors, agent = _stack(
        _card("mem-a", "g1", ["p1"]),
        _card("mem-b", "g1", ["p2"]),
        _card("mem-c", "g1", ["p3"]),
    )
    first = await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent)
    # a single pass folds pairwise; idempotent re-runs converge to one card.
    while await consolidate(store=store, gate=gate, neighbors=neighbors, agent=agent):
        pass
    assert first >= 1
    assert len(store.card_store.cards) == 1
