"""Periodic consolidation: a batch near-duplicate merge over the card bank.

The online librarian pre-gate is greedy and order-dependent — two same-lever
cards can both enter as NEW if neither pulls the other into top-k at birth. This
sweep is the standard drift fix: run the *same* ``NeighborSource`` nearest-card
primitive over the whole bank instead of one note, and fold each near-dup pair
(distance <= ``eps``) into one canonical card via ``CardAdmissionGate.merge``.
The absorbed card's provenance is preserved on the survivor; the absorbed id is
then removed. Idempotent — a second run over a deduped bank finds no pair within
``eps`` and merges nothing. No DBSCAN, no clustering utility.
"""

from __future__ import annotations

from typing import Any

from gigaevo.memory.ideas_tracker.librarian import NeighborSource
from gigaevo.memory.shared_memory.models import AnyCard, MemoryCard


async def consolidate(
    *,
    store: Any,
    gate: Any,
    neighbors: NeighborSource,
    agent: Any,
    eps: float = 0.05,
    k: int = 5,
) -> int:
    """Fold near-duplicate idea cards into canonical cards. Returns merge count.

    Deletion of absorbed cards is deferred to the end of the pass so the bank is
    stable while neighbors are ranked, and so the ``consumed`` set is the sole
    guard against re-merging a pair in both directions.
    """
    cards = list(store.card_store.cards.values())
    consumed: set[str] = set()
    absorbed: list[str] = []
    merges = 0
    for card in cards:
        # Only idea cards drift into duplicates; program exemplar cards are
        # identity-keyed and re-authored each sweep, so never merge them.
        if not isinstance(card, MemoryCard) or card.id in consumed:
            continue
        desc = (card.description or "").strip()
        if not desc:
            continue
        partner = _nearest_drift(card, neighbors.nearest(desc, k), eps, consumed)
        if partner is None:
            continue
        union = await agent.arun(card_a=card, card_b=partner)
        gate.merge(
            card.id,
            MemoryCard(
                id=card.id,
                description=union.description,
                keywords=list(union.keywords),
                programs=_union_programs(card, partner),
                task_description=card.task_description,
                task_description_summary=card.task_description_summary,
            ),
        )
        consumed.add(card.id)
        consumed.add(partner.id)
        absorbed.append(partner.id)
        merges += 1
    for cid in absorbed:
        store.delete(cid)
    return merges


def _nearest_drift(
    card: MemoryCard,
    hits: list[tuple[AnyCard, float]],
    eps: float,
    consumed: set[str],
) -> MemoryCard | None:
    for neighbor, distance in hits:
        if (
            neighbor.id == card.id
            or neighbor.id in consumed
            or not isinstance(neighbor, MemoryCard)
        ):
            continue
        # Hits are ascending: the first eligible neighbor is the closest, so if
        # it is beyond eps no later eligible neighbor can be a drift pair.
        return neighbor if distance <= eps else None
    return None


def _union_programs(a: MemoryCard, b: MemoryCard) -> list[str]:
    out: list[str] = []
    for prog in [*a.programs, *b.programs]:
        if prog not in out:
            out.append(prog)
    return out
