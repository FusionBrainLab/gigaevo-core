"""Thompson auction over retrieved candidates, plus the injection budgeters.

The auction is an emergent 0..N filter — every candidate's posterior draw
competes against a fresh no-card baseline draw. The budgeter is the hard
ceiling that caps the winner set to the mutator-facing ``max_cards``.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.events import MemoryAuctionRun, MemoryBudgetCap, emit_memory_event
from gigaevo.programs.metrics.context import MetricsContext

# Last-resort cold magnitude for a degenerate round: all cards cold AND the task
# declares no significant_change. Its value is inert — a common positive factor
# cancels from the bid ranking and clears the default zero floor — so cold cards
# stay explorable without a tunable scale-blind knob.
_UNSCALED_COLD_MAGNITUDE = 1.0


class AuctionCandidate(BaseModel):
    """One card entering the Thompson auction with its help-probability posterior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str = Field(description="Bank id of the candidate card.")
    posterior_a: float = Field(
        description="Alpha of the card's Beta help-probability posterior."
    )
    posterior_b: float = Field(
        description="Beta of the card's Beta help-probability posterior."
    )
    magnitude: float | None = Field(
        default=None,
        description="Expected-gain magnitude (IntroGain_best_median); None when "
        "cold. Only the EV auction consumes it; the safety auction ignores it.",
    )


class AuctionBid(BaseModel):
    """Audit record of one candidate's auction round: its sampled help
    probability against a fresh no-card baseline draw."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str = Field(description="Bank id of the candidate card.")
    posterior_a: float = Field(
        description="Alpha of the card's Beta help-probability posterior."
    )
    posterior_b: float = Field(
        description="Beta of the card's Beta help-probability posterior."
    )
    theta: float = Field(
        description="Help probability sampled from the card posterior."
    )
    baseline_a: float = Field(description="Alpha of the no-card baseline prior.")
    baseline_b: float = Field(description="Beta of the no-card baseline prior.")
    baseline_theta: float = Field(
        description="Help probability sampled from the no-card baseline prior."
    )
    selected: bool = Field(description="True iff the card draw beat the baseline draw.")
    magnitude: float | None = Field(
        default=None,
        description="Magnitude used for the EV bid (card's own, or the borrowed cold "
        "magnitude); None for the safety auction.",
    )
    bid: float | None = Field(
        default=None,
        description="Realized EV bid (theta_bid x magnitude); None for the safety "
        "auction, where the budgeter ranks by theta instead.",
    )


def _bid_dicts(slate: list[AuctionBid]) -> tuple[dict[str, Any], ...]:
    return tuple(bid.model_dump(mode="json") for bid in slate)


class ThompsonAuctioneer(BaseModel):
    """Thompson auction: each card's posterior draw competes against a no-card arm.

    For each candidate draw ``theta ~ Beta(posterior_a, posterior_b)`` and a
    fresh no-card ``base ~ Beta(*baseline_prior)``; select the card iff
    ``theta > base``. Winners are the emergent 0..N subset; the returned
    :class:`AuctionBid` slate keeps per-candidate draws for audit. Draw order
    (theta then base, per candidate) is part of the contract — it makes runs
    seed-exact reproducible.
    """

    model_config = ConfigDict(frozen=True)

    baseline_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="(alpha, beta) of the no-card baseline arm each candidate must beat.",
    )

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]:
        base_a, base_b = self.baseline_prior
        winners: list[str] = []
        slate: list[AuctionBid] = []
        for candidate in candidates:
            theta = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            base_theta = float(rng.beta(base_a, base_b))
            selected = theta > base_theta
            if selected:
                winners.append(candidate.card_id)
            slate.append(
                AuctionBid(
                    card_id=candidate.card_id,
                    posterior_a=float(candidate.posterior_a),
                    posterior_b=float(candidate.posterior_b),
                    theta=theta,
                    baseline_a=float(base_a),
                    baseline_b=float(base_b),
                    baseline_theta=base_theta,
                    selected=selected,
                )
            )
        if slate:
            emit_memory_event(
                MemoryAuctionRun(
                    auction="thompson",
                    candidate_count=len(slate),
                    winner_count=len(winners),
                    winner_ids=tuple(winners),
                    baseline_prior=(float(base_a), float(base_b)),
                    bids=_bid_dicts(slate),
                )
            )
            logger.debug(
                "[Memory][Auction] {}/{} candidate(s) beat baseline Beta{}: {}",
                len(winners),
                len(slate),
                self.baseline_prior,
                "; ".join(
                    "{} a/b={:.3g}/{:.3g} theta={:.3f} base={:.3f} {}".format(
                        bid.card_id,
                        bid.posterior_a,
                        bid.posterior_b,
                        bid.theta,
                        bid.baseline_theta,
                        "WIN" if bid.selected else "lose",
                    )
                    for bid in slate
                ),
            )
        return winners, slate


class EVThompsonAuctioneer(BaseModel):
    """EV-bid variant of the Thompson auction (the ``thompson_ev`` arm).

    Same no-card abstain gate as :class:`ThompsonAuctioneer` — a card is a
    winner iff a fresh ``theta ~ Beta(posterior_a, posterior_b)`` beats a fresh
    ``base ~ Beta(*baseline_prior)`` — but each candidate additionally bids
    ``theta_bid x magnitude`` so the downstream :class:`TopBidBudgeter` ranks
    winners by expected gain, not raw help probability. A cold card
    (``magnitude is None``) has no gain scale of its own, so it borrows one, in
    order of preference: (1) the median of the positive magnitudes of the warm
    cards in the same round — the task's own realized helpful-gain scale;
    (2) failing that, the primary metric's ``significant_change`` from
    ``metrics_context`` — a declared task-scaled threshold in the same units. So
    the cold bid tracks the task's gain scale rather than a fixed constant. Only
    a degenerate round — all cold, with no ``significant_change`` declared —
    falls back to an inert unit magnitude (see ``_UNSCALED_COLD_MAGNITUDE``).

    A winner must pass the safety gate AND clear the EV reserve ``ev_floor`` —
    its bid must be strictly positive (by default). Magnitude is signed, so a
    proven-harmful card (negative ``IntroGain_best_median``) can clear the
    safety gate yet bids negative; the floor abstains on it. Cold cards bid at
    that positive borrowed magnitude, so the floor never strands them. If every
    retrieved card is expected to hurt, the auction injects nothing.

    Draw order is pinned so a run is seed-exact reproducible: one bid draw per
    candidate first, then per candidate the gate draw followed by the baseline
    draw.
    """

    model_config = ConfigDict(frozen=True)

    baseline_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="(alpha, beta) of the no-card baseline arm each candidate must beat.",
    )
    metrics_context: MetricsContext | None = Field(
        default=None,
        description="Task metrics (shared ${ref:problem_context::metrics_context}). Its "
        "primary metric's significant_change — a task-scaled gain threshold in the same "
        "units as magnitude — is the cold fallback when a round has no warm magnitude to "
        "borrow.",
    )
    ev_floor: float = Field(
        default=0.0,
        description="EV reserve: a winner's bid (theta x magnitude) must exceed this. "
        "Default 0 abstains on non-positive expected gain (never inject a card you "
        "expect to hurt).",
    )

    def _cold_fallback(self) -> float:
        """Cold magnitude for an all-cold round: the primary metric's
        significant_change (task-scaled, same units as magnitude) if declared,
        else the inert unit placeholder."""
        if self.metrics_context is not None:
            sig = self.metrics_context.get_primary_spec().significant_change
            if sig is not None and math.isfinite(sig) and sig > 0.0:
                return float(sig)
        return _UNSCALED_COLD_MAGNITUDE

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]:
        warm = [
            c.magnitude
            for c in candidates
            if c.magnitude is not None
            and math.isfinite(c.magnitude)
            and c.magnitude > 0.0
        ]
        cold_magnitude = statistics.median(warm) if warm else self._cold_fallback()
        bid_draws: list[tuple[float, float]] = []
        for candidate in candidates:
            theta_bid = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            mag = (
                candidate.magnitude
                if candidate.magnitude is not None
                else cold_magnitude
            )
            bid_draws.append((mag, theta_bid * mag))
        base_a, base_b = self.baseline_prior
        winners: list[str] = []
        slate: list[AuctionBid] = []
        for candidate, (mag, bid_value) in zip(candidates, bid_draws):
            theta = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            base_theta = float(rng.beta(base_a, base_b))
            selected = theta > base_theta and bid_value > self.ev_floor
            if selected:
                winners.append(candidate.card_id)
            slate.append(
                AuctionBid(
                    card_id=candidate.card_id,
                    posterior_a=float(candidate.posterior_a),
                    posterior_b=float(candidate.posterior_b),
                    theta=theta,
                    baseline_a=float(base_a),
                    baseline_b=float(base_b),
                    baseline_theta=base_theta,
                    selected=selected,
                    magnitude=float(mag),
                    bid=float(bid_value),
                )
            )
        if slate:
            emit_memory_event(
                MemoryAuctionRun(
                    auction="thompson_ev",
                    candidate_count=len(slate),
                    winner_count=len(winners),
                    winner_ids=tuple(winners),
                    baseline_prior=(float(base_a), float(base_b)),
                    cold_magnitude=float(cold_magnitude),
                    ev_floor=float(self.ev_floor),
                    bids=_bid_dicts(slate),
                )
            )
            logger.debug(
                "[Memory][EVAuction] {}/{} candidate(s) beat baseline Beta{}: {}",
                len(winners),
                len(slate),
                self.baseline_prior,
                "; ".join(
                    "{} theta={:.3f} base={:.3f} mag={:.4g} bid={:.4g} {}".format(
                        bid.card_id,
                        bid.theta,
                        bid.baseline_theta,
                        bid.magnitude if bid.magnitude is not None else float("nan"),
                        bid.bid if bid.bid is not None else float("nan"),
                        "WIN" if bid.selected else "lose",
                    )
                    for bid in slate
                ),
            )
        return winners, slate


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
            MemoryBudgetCap(
                rank_key="theta",
                winner_count=len(card_ids),
                max_cards=max_cards,
                kept_ids=tuple(kept),
                dropped_ids=tuple(dropped),
                rank_by_card_id=theta,
            )
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
            MemoryBudgetCap(
                rank_key="bid",
                winner_count=len(card_ids),
                max_cards=max_cards,
                kept_ids=tuple(kept),
                dropped_ids=tuple(dropped),
                rank_by_card_id=bid,
            )
        )
        logger.debug(
            "[Memory][Budgeter] Capped {} winner(s) to max_cards={} by EV bid: kept={} dropped={}",
            len(card_ids),
            max_cards,
            kept,
            dropped,
        )
        return kept
