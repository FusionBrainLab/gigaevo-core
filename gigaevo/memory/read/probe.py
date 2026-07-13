"""Explicit cold-card exploration policy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.read.auction import AuctionBid


class NoColdProbePolicy(BaseModel):
    """No-op probe policy for legacy/ablation configs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def apply(
        self,
        *,
        budgeted_ids: list[str],
        slate: list[AuctionBid],
        max_cards: int,
        rng: Any,
    ) -> tuple[list[str], list[AuctionBid]]:
        del max_cards, rng
        return list(budgeted_ids), list(slate)


class ColdProbePolicy(BaseModel):
    """Spend a small explicit exploration lane on under-evidenced candidates.

    This policy runs after the exploitation auction and hard budget.  If the
    auction selected nothing, it can fill one slot with a probe card.  If warm
    winners exist, it can rarely add a probe, displacing the weakest budgeted
    card only when the budget is already full.  Eligibility is
    evidential — any bid that reports a support kind and whose effective
    (staleness-scaled) support is below ``probe_until_effective_events`` may
    probe — so a card whose only history is diluted unused/invalid exposure
    keeps a path back into circulation instead of dying on one ignored prompt.
    The threshold defaults to the eviction evidence floor, partitioning
    card-space: below the floor a card may probe, at or above it the card is
    adjudicable by auction merit or eviction.  Bids with an empty support kind
    (auctioneers that do not report support) are never probe-eligible, so the
    lane fails safe instead of reading the field default as cold.  All
    probabilities are config fields; there are no hidden constants in the
    selection rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=True)
    probe_until_effective_events: float = Field(
        default=3.0,
        ge=0.0,
        description="Cards with effective support_n below this remain "
        "probe-eligible; keep equal to the eviction evidence floor "
        "(memory.evidence.min_effective_events) so no card falls between "
        "the probe and eviction lanes.",
    )
    max_probe_cards_per_decision: int = Field(
        default=1,
        ge=0,
        description="Maximum cold cards this policy may add/override per read.",
    )
    empty_selection_probe_rate: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Probability of filling an otherwise empty selection with a cold probe.",
    )
    warm_override_probe_rate: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Probability of replacing one warm budgeted card with a cold probe.",
    )

    def apply(
        self,
        *,
        budgeted_ids: list[str],
        slate: list[AuctionBid],
        max_cards: int,
        rng: Any,
    ) -> tuple[list[str], list[AuctionBid]]:
        marked_slate = [
            bid.model_copy(
                update={
                    "probe_eligible": bool(bid.support_kind)
                    and bid.support_n < self.probe_until_effective_events
                }
            )
            for bid in slate
        ]
        if not self.enabled or max_cards <= 0 or self.max_probe_cards_per_decision <= 0:
            return list(budgeted_ids), marked_slate
        budgeted = list(budgeted_ids)
        selected = set(budgeted)
        candidates = [
            bid
            for bid in marked_slate
            if bid.card_id not in selected and bid.probe_eligible
        ]
        if not candidates:
            return budgeted, marked_slate
        candidates.sort(
            key=lambda bid: (
                -(bid.bid if bid.bid is not None else 0.0),
                -bid.theta,
                bid.card_id,
            )
        )
        if not budgeted:
            if float(rng.random()) >= self.empty_selection_probe_rate:
                return budgeted, marked_slate
            return self._select_probe(
                budgeted,
                marked_slate,
                candidates[: self.max_probe_cards_per_decision],
                reason="cold_probe_empty",
                max_cards=max_cards,
            )
        if float(rng.random()) >= self.warm_override_probe_rate:
            return budgeted, marked_slate
        return self._select_probe(
            budgeted,
            marked_slate,
            candidates[: self.max_probe_cards_per_decision],
            reason="cold_probe_override",
            max_cards=max_cards,
            replace=True,
        )

    def _select_probe(
        self,
        budgeted: list[str],
        slate: list[AuctionBid],
        probes: list[AuctionBid],
        *,
        reason: str,
        max_cards: int,
        replace: bool = False,
    ) -> tuple[list[str], list[AuctionBid]]:
        kept = list(budgeted)
        if replace and kept:
            overflow = len(kept) + len(probes) - max_cards
            if overflow > 0:
                if overflow >= len(kept):
                    kept = []
                else:
                    bids = {bid.card_id: bid for bid in slate}

                    def _rank(cid: str) -> tuple[float, float, str]:
                        bid = bids.get(cid)
                        if bid is None:
                            return (0.0, 0.0, cid)
                        return (
                            -(bid.bid if bid.bid is not None else 0.0),
                            -bid.theta,
                            bid.card_id,
                        )

                    drop = set(sorted(kept, key=_rank)[len(kept) - overflow :])
                    kept = [cid for cid in kept if cid not in drop]
        for probe in probes:
            if len(kept) >= max_cards:
                break
            kept.append(probe.card_id)
        probe_ids = {probe.card_id for probe in probes[: max_cards or 0]}
        final_selected = set(kept)
        return kept, [
            bid.model_copy(
                update={
                    "selected": bid.card_id in final_selected,
                    "probe_eligible": bid.probe_eligible,
                    "probe_selected": bid.card_id in probe_ids,
                    "selection_reason": reason
                    if bid.card_id in probe_ids
                    else bid.selection_reason,
                    "rejected_by_ev_floor": False
                    if bid.card_id in probe_ids
                    else bid.rejected_by_ev_floor,
                    "rejected_by_no_card_gate": False
                    if bid.card_id in probe_ids
                    else bid.rejected_by_no_card_gate,
                }
            )
            for bid in slate
        ]
