from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.events import emit_memory_event


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
        description="Magnitude used for the EV bid (card's own, or the cold prior); "
        "None for the safety auction.",
    )
    bid: float | None = Field(
        default=None,
        description="Realized EV bid (theta_bid x magnitude); None for the safety "
        "auction, where the budgeter ranks by theta instead.",
    )


class ThompsonAuctioneer(BaseModel):
    """Thompson auction: each card's posterior draw competes against a no-card arm.

    For each candidate draw ``theta ~ Beta(posterior_a, posterior_b)`` and a
    fresh no-card ``base ~ Beta(*baseline_prior)``; select the card iff
    ``theta > base``. Winners are the emergent 0..N subset; the returned
    :class:`AuctionBid` slate keeps per-candidate draws for audit. Draw order
    (theta then base, per candidate) is part of the contract — it makes runs
    seed-exact reproducible (pinned in tests/memory/test_core_efficacy.py).
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
                component="Auction",
                event_type="auction.run",
                payload={
                    "candidate_count": len(slate),
                    "winner_count": len(winners),
                    "winner_ids": winners,
                    "baseline_prior": [float(base_a), float(base_b)],
                    "bids": [bid.model_dump(mode="json") for bid in slate],
                },
                level="INFO",
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
    ``theta_bid x magnitude`` so the downstream :class:`~gigaevo.memory.core.
    budgeter.TopBidBudgeter` ranks winners by expected gain, not raw help
    probability. Cold cards (``magnitude is None``) bid against
    ``prior_magnitude`` so exploration does not starve them at zero.

    A winner must pass the safety gate AND clear the EV reserve ``ev_floor`` —
    its bid must be strictly positive (by default). Magnitude is signed, so a
    proven-harmful card (negative ``IntroGain_best_median``) can clear the
    safety gate yet bids negative; the floor abstains on it. Cold cards bid
    against the positive ``prior_magnitude`` so the floor never strands them. If
    every retrieved card is expected to hurt, the auction injects nothing.

    Draw order is pinned for replay parity with the offline reference
    (``rerank_arm.py``): one bid draw per candidate first, then per candidate
    the gate draw followed by the baseline draw.
    """

    model_config = ConfigDict(frozen=True)

    baseline_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="(alpha, beta) of the no-card baseline arm each candidate must beat.",
    )
    prior_magnitude: float = Field(
        default=0.1,
        description="Optimistic expected-gain prior bid for cold cards (no stamped "
        "magnitude); keeps exploration from starving fresh cards at zero.",
    )
    ev_floor: float = Field(
        default=0.0,
        description="EV reserve: a winner's bid (theta x magnitude) must exceed this. "
        "Default 0 abstains on non-positive expected gain (never inject a card you "
        "expect to hurt).",
    )

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]:
        bid_draws: list[tuple[float, float]] = []
        for candidate in candidates:
            theta_bid = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            mag = (
                candidate.magnitude
                if candidate.magnitude is not None
                else self.prior_magnitude
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
                component="Auction",
                event_type="auction.run",
                payload={
                    "candidate_count": len(slate),
                    "winner_count": len(winners),
                    "winner_ids": winners,
                    "baseline_prior": [float(base_a), float(base_b)],
                    "prior_magnitude": float(self.prior_magnitude),
                    "ev_floor": float(self.ev_floor),
                    "bids": [bid.model_dump(mode="json") for bid in slate],
                },
                level="INFO",
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
