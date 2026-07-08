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
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta as scipy_beta

from gigaevo.memory.events import MemoryAuctionRun, MemoryBudgetCap, emit_memory_event
from gigaevo.memory.read.bootstrap import bootstrap_ev_samples
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
        description="Expected-gain magnitude; None when cold. Bootstrap reputation "
        "projects bootstrap EV here, non-bootstrap EV uses IntroGain_best_median. "
        "Only the EV auction consumes it; the safety auction ignores it.",
    )
    deltas: tuple[float, ...] | None = Field(
        default=None,
        description="The card's EV support values. None/empty with magnitude=None "
        "means true cold; zero values from unused/invalid exposure mean known "
        "zero support. Only the bootstrap auction consumes it.",
    )
    delta_weights: tuple[float, ...] | None = Field(
        default=None,
        description="Causal weights aligned with deltas for weighted bootstrap EV.",
    )
    staleness_weight: float = Field(
        default=1.0,
        description="Bank-cycle evidence discount w = 2**(-s/H), the per-event "
        "bootstrap resample weight; 1.0 (no ageing) under an un-decayed reputation.",
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
    baseline_quantile: float | None = Field(
        default=None,
        description="CDF_Beta(baseline_prior)(baseline_theta) for CDF-space "
        "slate gates; None when the auction compares raw theta directly.",
    )
    theta_quantile: float | None = Field(
        default=None,
        description="CDF-space audit value for theta under the gate's reference "
        "distribution; None when the auction compares raw theta directly.",
    )
    gate_quantile: float | None = Field(
        default=None,
        description="Baseline-CDF quantile threshold used by a CDF-space slate "
        "gate; None when the auction compares raw theta directly.",
    )
    selected: bool = Field(
        description="True iff the card draw cleared this auction's no-card gate."
    )
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
    support_kind: str = Field(
        default="",
        description="EV support source: cold_prior, ev_rewards, or zero_support.",
    )
    support_n: float = Field(
        default=0.0, description="Effective EV support weight behind this bid."
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

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    A winner must pass the safety gate AND clear the EV reserve — its bid must
    strictly exceed ``max(ev_floor, Beta(*baseline_prior).ppf(ev_floor_quantile)
    * cold_magnitude)``. The quantile leg is self-normalizing: it asks the bid
    to beat what the no-card baseline arm itself would bid at that quantile,
    on the round's own gain scale (the borrowed cold magnitude above) — so the
    same config transfers across tasks whose fitness deltas differ by orders
    of magnitude. The absolute ``ev_floor`` stays available for
    experiment-calibrated pins only. Magnitude is signed, so a proven-harmful
    card (negative magnitude) can clear the safety gate yet bids negative; the
    floor abstains on it. If every retrieved card is
    expected to hurt, the auction injects nothing.

    Draw order is pinned so a run is seed-exact reproducible: one bid draw per
    candidate first, then per candidate the gate draw followed by the baseline
    draw.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
        description="Absolute EV reserve: a winner's bid (theta x magnitude) must exceed "
        "max(ev_floor, quantile leg). Default 0 abstains on non-positive expected gain "
        "(never inject a card you expect to hurt). Task-scale-dependent — pin only in "
        "calibrated experiments, never in shared config.",
    )
    ev_floor_quantile: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Self-normalizing EV reserve: the bid must beat the q-quantile of "
        "the no-card baseline's own EV, Beta(*baseline_prior).ppf(q) x the round's "
        "cold magnitude (warm-median gain, else the primary metric's "
        "significant_change). Quantile of the run's own distributions — transfers "
        "across tasks. In a degenerate all-cold round with no declared "
        "significant_change it gates on theta_bid > ppf(q) directly (unit magnitude).",
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
        base_a, base_b = self.baseline_prior
        effective_floor = self.ev_floor
        if self.ev_floor_quantile is not None:
            baseline_theta_q = float(
                scipy_beta.ppf(self.ev_floor_quantile, base_a, base_b)
            )
            effective_floor = max(effective_floor, baseline_theta_q * cold_magnitude)
        bid_draws: list[tuple[float, float]] = []
        for candidate in candidates:
            theta_bid = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            mag = (
                candidate.magnitude
                if candidate.magnitude is not None
                else cold_magnitude
            )
            bid_draws.append((mag, theta_bid * mag))
        winners: list[str] = []
        slate: list[AuctionBid] = []
        for candidate, (mag, bid_value) in zip(candidates, bid_draws):
            theta = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            base_theta = float(rng.beta(base_a, base_b))
            selected = theta > base_theta and bid_value > effective_floor
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
                    ev_floor=float(effective_floor),
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


class BootstrapThompsonAuctioneer(BaseModel):
    """Bootstrap-EV variant of the Thompson auction (the ``thompson_bootstrap``
    arm).

    Same no-card safety gate as :class:`EVThompsonAuctioneer` — a slate injects
    only if a card draw beats one multiple-comparison-adjusted no-card baseline
    for the decision — but the EV bid is priced on the card's RAW oriented
    deltas rather than a binarized ``p_help x median``. Each candidate bids the
    mean of ONE weighted bootstrap resample of known-card EV support plus a
    neutral zero pseudo-event. Genuinely cold cards instead bid a posterior draw
    times the round's borrowed gain scale (in-round warm-magnitude median, else
    the primary metric's ``significant_change``, else the inert unit — the same
    cold ladder ``EVThompsonAuctioneer`` climbs). Pricing on the mean makes a
    fat left tail visible: a card whose wins outnumber its losses but loses far
    more per loss bids negative, the failure mode the median bid was blind to.
    Staleness rides
    the per-event resample weight
    (``staleness_weight`` — the one bank-cycle mechanism shared with
    ``DecayingReputation``): a stale known card's own deltas fade toward neutral
    zero, while genuinely cold cards use the round's cold scale for their first
    probe.

    The EV reserve is two self-normalizing gates, neither a Beta assumption nor
    an absolute scale. (1) A sign gate ``bid > 0``: never inject a card you
    expect to hurt — this alone yields the abstain-on-all-harmful guarantee (an
    all-negative round wins nothing). (2) A spread reserve ``bid >=`` the
    ``ev_floor_quantile`` quantile of the round's OWN bids: the bid must sit in
    the upper part of the round's realized distribution. The quantile is
    inclusive so ties at the floor do not self-annihilate; true cold bids are
    posterior-sampled, so an all-cold slate no longer becomes a fixed point mass.

    Draw order is pinned for seed-exact replay: one bid resample per candidate
    first, then one slate baseline draw, then per-candidate gate draws.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="(alpha, beta) of the no-card baseline arm each candidate must beat.",
    )
    metrics_context: MetricsContext | None = Field(
        default=None,
        description="Task metrics (shared ${ref:problem_context::metrics_context}). Its "
        "primary metric's significant_change is the cold gain scale when a round has no "
        "warm magnitude to borrow.",
    )
    ev_floor_quantile: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Self-normalizing EV reserve: the winning bid must exceed the "
        "q-quantile of the round's own bid distribution, clamped at zero. Quantile of "
        "the run's own bids — no Beta assumption, no absolute scale; the clamp keeps "
        "the abstain-on-all-harmful guarantee.",
    )

    def _cold_fallback(self) -> float:
        """Cold gain scale for an all-cold round: the primary metric's
        significant_change if declared, else the inert unit placeholder."""
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
        base_a, base_b = self.baseline_prior
        bids: list[float] = []
        support_scales: list[float] = []
        support_kinds: list[str] = []
        support_counts: list[float] = []
        for candidate in candidates:
            deltas = candidate.deltas or ()
            weights = candidate.delta_weights
            support_count = (
                float(sum(weights)) if weights is not None else float(len(deltas))
            )
            true_cold = not deltas and candidate.magnitude is None
            support_scale = cold_magnitude if true_cold else 0.0
            support_scales.append(support_scale)
            support_counts.append(support_count)
            if true_cold:
                support_kinds.append("cold_prior")
                theta_bid = float(
                    rng.beta(candidate.posterior_a, candidate.posterior_b)
                )
                bids.append(theta_bid * cold_magnitude)
            else:
                support_kinds.append("ev_rewards" if deltas else "zero_support")
                sample = bootstrap_ev_samples(
                    deltas,
                    support_scale,
                    candidate.staleness_weight,
                    1,
                    rng,
                    delta_weights=weights,
                )
                bids.append(float(sample[0]))
        quantile_floor = (
            float(np.quantile(bids, self.ev_floor_quantile)) if bids else 0.0
        )
        # Telemetry: the binding reserve a winner actually clears is the sign gate
        # OR-ed with the spread quantile, whichever is higher.
        effective_floor = max(0.0, quantile_floor)
        eligible = [
            bid_value > 0.0 and bid_value >= quantile_floor for bid_value in bids
        ]
        eligible_count = sum(1 for ok in eligible if ok)
        base_theta = float(rng.beta(base_a, base_b)) if candidates else 0.0
        # Family-wise no-card gate in baseline-CDF space: convert the sampled
        # no-card draw to its baseline quantile, Sidak-adjust that quantile over
        # the eligible slate, and map it back through the SAME no-card baseline
        # distribution. Candidate quality stays in the raw theta draw; using
        # each card's own posterior CDF would make every posterior look uniform
        # and erase whether the card is actually strong or weak.
        baseline_quantile = (
            float(scipy_beta.cdf(base_theta, base_a, base_b)) if candidates else 0.0
        )
        gate_quantile = (
            baseline_quantile ** (1.0 / eligible_count)
            if eligible_count > 1
            else baseline_quantile
        )
        gate_theta = float(scipy_beta.ppf(gate_quantile, base_a, base_b))
        winners: list[str] = []
        slate: list[AuctionBid] = []
        for (
            candidate,
            bid_value,
            support_scale,
            support_kind,
            support_n,
            can_bid,
        ) in zip(
            candidates, bids, support_scales, support_kinds, support_counts, eligible
        ):
            theta = float(rng.beta(candidate.posterior_a, candidate.posterior_b))
            theta_quantile = float(scipy_beta.cdf(theta, base_a, base_b))
            selected = can_bid and theta > gate_theta
            if selected:
                winners.append(candidate.card_id)
            mag = (
                candidate.magnitude
                if candidate.magnitude is not None
                else support_scale
            )
            slate.append(
                AuctionBid(
                    card_id=candidate.card_id,
                    posterior_a=float(candidate.posterior_a),
                    posterior_b=float(candidate.posterior_b),
                    theta=theta,
                    baseline_a=float(base_a),
                    baseline_b=float(base_b),
                    baseline_theta=base_theta,
                    baseline_quantile=baseline_quantile,
                    theta_quantile=theta_quantile,
                    gate_quantile=gate_quantile,
                    selected=selected,
                    magnitude=float(mag),
                    bid=float(bid_value),
                    support_kind=support_kind,
                    support_n=support_n,
                )
            )
        if slate:
            emit_memory_event(
                MemoryAuctionRun(
                    auction="thompson_bootstrap",
                    candidate_count=len(slate),
                    winner_count=len(winners),
                    winner_ids=tuple(winners),
                    baseline_prior=(float(base_a), float(base_b)),
                    cold_magnitude=float(cold_magnitude),
                    ev_floor=float(effective_floor),
                    bids=_bid_dicts(slate),
                )
            )
            logger.debug(
                "[Memory][BootstrapAuction] {}/{} candidate(s) beat baseline Beta{} "
                "at EV floor {:.4g}: {}",
                len(winners),
                len(slate),
                self.baseline_prior,
                effective_floor,
                "; ".join(
                    "{} theta={:.3f} q={:.3f} gate_q={:.3f} bid={:.4g} {}".format(
                        bid.card_id,
                        bid.theta,
                        bid.theta_quantile if bid.theta_quantile is not None else 0.0,
                        bid.gate_quantile if bid.gate_quantile is not None else 0.0,
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]:
        if len(card_ids) <= max_cards:
            return list(card_ids)
        theta = {bid.card_id: bid.theta for bid in slate}
        kept = sorted(card_ids, key=lambda c: (-theta.get(c, 0.0), c))[:max_cards]
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    def cap(
        self, card_ids: list[str], slate: list[AuctionBid], max_cards: int
    ) -> list[str]:
        if len(card_ids) <= max_cards:
            return list(card_ids)
        bid = {b.card_id: (b.bid if b.bid is not None else 0.0) for b in slate}
        theta = {b.card_id: b.theta for b in slate}
        kept = sorted(
            card_ids, key=lambda c: (-bid.get(c, 0.0), -theta.get(c, 0.0), c)
        )[:max_cards]
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
