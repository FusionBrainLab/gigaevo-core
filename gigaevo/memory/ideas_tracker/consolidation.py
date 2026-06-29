"""Periodic consolidation: a batch near-duplicate merge over the card bank.

The online librarian pre-gate is greedy and order-dependent — two same-lever
cards can both enter as NEW if neither pulls the other into top-k at birth. This
sweep is the standard drift fix: run the *same* ``NeighborSource`` nearest-card
primitive over the whole bank instead of one note, surface every near neighbor
(distance <= ``eps``) as a merge *candidate*, and let the consolidate agent be
the precision arbiter — it folds a candidate pair into one canonical card via
``CardAdmissionGate.merge`` only when it rules they name the same lever, and
abstains otherwise so a generous candidate gate can never force-merge distinct
cards. On abstain the pass tries the next-nearest candidate. The absorbed card's
provenance is preserved on the survivor; the absorbed id is then removed.
Idempotent — a second run over a deduped bank finds no foldable pair and merges
nothing. No DBSCAN, no clustering utility.
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
    eps: float = 0.2,
    k: int = 5,
) -> int:
    """Fold near-duplicate idea cards into canonical cards. Returns merge count.

    Deletion of absorbed cards is deferred to the end of the pass so the bank is
    stable while neighbors are ranked, and so the ``consumed`` set is the sole
    guard against re-merging a pair in both directions.
    """
    cards = list(store.all_cards_snapshot().values())
    consumed: set[str] = set()
    absorbed: list[str] = []
    # The merge ruling is symmetric, so once the agent declines an unordered pair
    # we must not pay a second LLM call to re-review it as (partner, card) when
    # the loop reaches the partner. Records every declined pair.
    reviewed: set[frozenset[str]] = set()
    merges = 0
    # Absorbed partners are deleted in a finally so a mid-pass agent failure
    # cannot leave an already-merged partner in the bank (its evidence is now
    # on the survivor — an undeleted partner would be double-counted).
    try:
        for card in cards:
            # Only idea cards drift into duplicates; program exemplar cards are
            # identity-keyed and re-authored each sweep, so never merge them.
            if not isinstance(card, MemoryCard) or card.id in consumed:
                continue
            desc = (card.description or "").strip()
            if not desc:
                continue
            # nearest() truncates at its k; the query card's own description and
            # this-pass consumed cards still occupy index slots (deletion is
            # deferred to the finally), so they crowd the fixed top-k and can
            # hide a valid partner past the cutoff. Over-fetch past self + every
            # consumed id so k genuine candidates always reach the arbiter.
            fetch = k + 1 + len(consumed)
            candidates = _drift_candidates(
                card, neighbors.nearest(desc, fetch, MemoryCard), eps, consumed
            )
            for partner in candidates:
                pair = frozenset({card.id, partner.id})
                if pair in reviewed:
                    continue
                # The eps gate only recalls candidates; the agent is the precision
                # arbiter. On abstain (the two are not the same lever) move on to
                # the next-nearest candidate rather than force-folding them.
                decision = await agent.arun(card_a=card, card_b=partner)
                if not decision.merge or decision.card is None:
                    reviewed.add(pair)
                    continue
                union = decision.card
                fid = gate.merge(
                    card.id,
                    MemoryCard(
                        # The survivor is ``card`` (target_id); the partner is
                        # folded away and deleted. The gate reads the submitted
                        # card's id as the ledger's incoming_id, so it must be the
                        # absorbed partner's — else the deleted card has no
                        # ledger/replay trace.
                        id=partner.id,
                        description=union.description,
                        explanation_summary=union.explanation_summary,
                        # Trust the agent's curated union keyword set; the gate's
                        # replace-on-merge fold takes it verbatim. Re-unioning the
                        # partner's raw list here would re-bloat the survivor.
                        keywords=list(union.keywords),
                        programs=list(partner.programs),
                        # Carry the partner's own absorbed-id chain forward so a
                        # multi-hop absorption keeps re-aliasing the earliest id
                        # onto this survivor (merge_cards adds partner.id itself).
                        absorbed_ids=list(partner.absorbed_ids),
                        gain_events=partner.gain_events,
                        task_description=card.task_description
                        or partner.task_description,
                        task_description_summary=(
                            card.task_description_summary
                            or partner.task_description_summary
                        ),
                    ),
                )
                # Queue the partner for deletion ONLY after a committed fold. The
                # gate returns a truthy id iff it folded the partner's evidence
                # onto the survivor; it returns "" without folding on a
                # harmful-union eviction or a backend miss, and ``apply_merges``
                # swallows per-target persist failures rather than raising — so
                # the only way merge raises is the harm-path ``delete`` (before
                # any fold). Queuing before the merge would orphan the partner on
                # that raise.
                if not fid:
                    consumed.add(card.id)
                    break
                absorbed.append(partner.id)
                consumed.add(card.id)
                consumed.add(partner.id)
                merges += 1
                break
    finally:
        for cid in absorbed:
            store.delete(cid)
    return merges


def _drift_candidates(
    card: MemoryCard,
    hits: list[tuple[AnyCard, float]],
    eps: float,
    consumed: set[str],
) -> list[MemoryCard]:
    out: list[MemoryCard] = []
    for neighbor, distance in hits:
        if (
            neighbor.id == card.id
            or neighbor.id in consumed
            or not isinstance(neighbor, MemoryCard)
        ):
            continue
        # Hits are ascending: once a neighbor is beyond eps, no later one can be a
        # candidate, so stop rather than keep scanning far cards.
        if distance > eps:
            break
        out.append(neighbor)
    return out
