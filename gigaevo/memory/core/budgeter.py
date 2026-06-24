from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.memory.core.auctioneer import AuctionBid
from gigaevo.memory.core.events import emit_memory_event


class TopThetaBudgeter(BaseModel):
    """Hard ceiling on what reaches the mutator. The auction is an emergent 0..N
    filter; when it keeps more than ``max_cards``, retain the strongest winners
    by sampled theta (the kept list is reordered theta-descending). Within
    budget, auction order is preserved."""

    model_config = ConfigDict(frozen=True)

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]:
        if len(card_ids) <= max_cards:
            return list(card_ids)
        theta = {bid.card_id: bid.theta for bid in slate}
        kept = sorted(card_ids, key=lambda c: theta.get(c, 0.0), reverse=True)[
            :max_cards
        ]
        dropped = [c for c in card_ids if c not in kept]
        emit_memory_event(
            component="Budgeter",
            event_type="budget.cap",
            payload={
                "winner_count": len(card_ids),
                "max_cards": max_cards,
                "kept_ids": kept,
                "dropped_ids": dropped,
                "theta_by_card_id": theta,
            },
            level="INFO",
        )
        logger.debug(
            "[Memory][Budgeter] Capped {} auction winner(s) to max_cards={}: kept={} dropped={}",
            len(card_ids),
            max_cards,
            kept,
            dropped,
        )
        return kept


class TopBidBudgeter(BaseModel):
    """Hard ceiling that ranks by the EV bid (``theta_bid x magnitude``) rather
    than the gate's theta — the budgeter half of the ``thompson_ev`` arm. When
    the auction keeps more than ``max_cards``, retain the strongest winners by
    realized bid (kept list reordered bid-descending). Within budget, auction
    order is preserved. A winner whose slate row carries no bid sorts as 0.0."""

    model_config = ConfigDict(frozen=True)

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]:
        if len(card_ids) <= max_cards:
            return list(card_ids)
        bid = {b.card_id: (b.bid if b.bid is not None else 0.0) for b in slate}
        kept = sorted(card_ids, key=lambda c: bid.get(c, 0.0), reverse=True)[:max_cards]
        dropped = [c for c in card_ids if c not in kept]
        emit_memory_event(
            component="Budgeter",
            event_type="budget.cap",
            payload={
                "winner_count": len(card_ids),
                "max_cards": max_cards,
                "kept_ids": kept,
                "dropped_ids": dropped,
                "bid_by_card_id": bid,
            },
            level="INFO",
        )
        logger.debug(
            "[Memory][Budgeter] Capped {} winner(s) to max_cards={} by EV bid: kept={} dropped={}",
            len(card_ids),
            max_cards,
            kept,
            dropped,
        )
        return kept
