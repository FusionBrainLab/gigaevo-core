"""Thompson auction over retrieved candidates, plus the injection budgeters.

The auction is an emergent 0..N filter — every candidate's posterior draw
competes against a fresh no-card baseline draw. The budgeter is the hard
ceiling that caps the winner set to the mutator-facing ``max_cards``.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Literal

from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import beta as scipy_beta

from gigaevo.memory.context.no_card import NoCardGateSummary
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
        description="Causal credit times per-event staleness, aligned with deltas.",
    )
    deltas_se: tuple[float | None, ...] | None = Field(
        default=None,
        description="Per-event gain ses aligned with deltas (0 exact, None unknown). "
        "Only the bootstrap auction consumes finite positive entries, jittering "
        "each corresponding resampled atom by N(0, se).",
    )
    support_n_unstaled: float = Field(
        default=0.0,
        description="Audit-only: raw Sum max(0, per-event credit) BEFORE staleness. "
        "support_n / support_n_unstaled is the per-event aging bite. No decision "
        "consumes it.",
    )
    gain_se: float | None = Field(
        default=None,
        description="Audit-only: paired se of the card's most-recent native gain "
        "event (0 exact, None unknown/degraded). No decision consumes it.",
    )
    staleness_weight: float = Field(
        default=1.0,
        description="Compatibility scalar applied after delta_weights; projected "
        "candidates neutralize it to 1.0 because ageing is already per event.",
    )
    prior_source: str = Field(
        default="reputation",
        description="Source of posterior_a/posterior_b when the card is cold.",
    )
    context_key: str = Field(
        default="",
        description="Read-context bucket used by contextual prior/no-card evidence.",
    )
    use_count: int = Field(
        default=0,
        ge=0,
        description="Deterministic injection count (the card's non-founding gain "
        "events). Only the novelty-discounted auction consumes it.",
    )
    pending_count: int = Field(
        default=0,
        ge=0,
        description="Audit count of uncredited in-flight exposures. Only the "
        "pending-discounted auction consumes it.",
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
        description="True iff the card cleared this auction's EV reserve and no-card "
        "gate."
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
        default=0.0,
        description="Effective (staleness-scaled) EV support weight behind this "
        "bid, on the same measure as eviction's effective support.",
    )
    support_n_unstaled: float = Field(
        default=0.0,
        description="Audit-only: raw EV support weight BEFORE staleness. "
        "support_n / support_n_unstaled is the per-event aging bite. No decision "
        "consumes it.",
    )
    gain_se: float | None = Field(
        default=None,
        description="Audit-only: paired se of the card's most-recent native gain "
        "event (0 exact, None unknown/degraded). No decision consumes it.",
    )
    use_count: int = Field(
        default=0,
        description="Injection count carried from the candidate; under the "
        "novelty-discounted auction, `bid` already includes the (1+use_count) tax.",
    )
    prior_source: str = Field(
        default="",
        description="Source of posterior_a/posterior_b for cold/context-cold cards.",
    )
    context_key: str = Field(
        default="",
        description="Read-context bucket used by contextual prior/no-card evidence.",
    )
    baseline_source: str = Field(
        default="",
        description="Source of the no-card baseline prior.",
    )
    no_card_baseline: float | None = Field(
        default=None,
        description="Robust no-card child-parent delta location for audit only.",
    )
    no_card_n: float = Field(
        default=0.0,
        description="Effective no-card evidence behind the baseline prior.",
    )
    ev_reserve_mode: Literal["quantile", "risk"] | None = Field(
        default=None,
        description="EV reserve that made this bid's admission decision; None when "
        "the auction has no bootstrap-EV reserve.",
    )
    ev_positive_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of this card's bootstrap-EV samples above zero when "
        "the risk reserve is active; None for the legacy quantile reserve.",
    )
    ev_risk_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Risk alpha used for this bid; None unless ev_reserve_mode=risk.",
    )
    rejected_by_ev_floor: bool = Field(
        default=False,
        description="True when the candidate failed the configured EV reserve.",
    )
    rejected_by_no_card_gate: bool = Field(
        default=False,
        description="True when the candidate cleared EV reserve but lost to no-card.",
    )
    probe_eligible: bool = Field(
        default=False,
        description="True when the cold-probe policy may spend exploration on it.",
    )
    probe_selected: bool = Field(
        default=False,
        description="True when selected by the explicit cold-probe policy.",
    )
    selection_reason: str = Field(
        default="",
        description="auction, cold_probe_empty, cold_probe_override, or empty.",
    )


def _bid_dicts(slate: list[AuctionBid]) -> tuple[dict[str, Any], ...]:
    return tuple(bid.model_dump(mode="json") for bid in slate)


def _decision_baseline(
    summary: NoCardGateSummary | None, fallback: tuple[float, float]
) -> tuple[float, float, str, float | None, float]:
    if summary is not None:
        a = float(summary.prior.alpha)
        b = float(summary.prior.beta)
        if math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0:
            return (
                a,
                b,
                summary.source or "dynamic",
                float(summary.baseline),
                float(summary.evidence_n),
            )
    return (float(fallback[0]), float(fallback[1]), "fixed", None, 0.0)


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
        self,
        candidates: list[AuctionCandidate],
        rng: Any,
        *,
        baseline: NoCardGateSummary | None = None,
    ) -> tuple[list[str], list[AuctionBid]]:
        base_a, base_b, baseline_source, no_card_baseline, no_card_n = (
            _decision_baseline(baseline, self.baseline_prior)
        )
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
                    prior_source=candidate.prior_source,
                    context_key=candidate.context_key,
                    baseline_source=baseline_source,
                    no_card_baseline=no_card_baseline,
                    no_card_n=no_card_n,
                    rejected_by_no_card_gate=not selected,
                    selection_reason="auction" if selected else "",
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

    Same no-card abstain gate as :class:`ThompsonAuctioneer`, but each candidate
    draws one posterior world ``u ~ U(0, 1)`` and maps it to
    ``theta = Beta(posterior_a, posterior_b).ppf(u)``. That same ``theta`` both
    faces the fresh ``base ~ Beta(*baseline_prior)`` gate and bids
    ``theta x magnitude``, so the downstream :class:`TopBidBudgeter` ranks a
    gate-coherent expected-gain draw, not an independent help-probability draw.
    A cold card
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

    Draw order is pinned so a run is seed-exact reproducible: one shared
    uniform world per candidate first, then one baseline draw per candidate.
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
        self,
        candidates: list[AuctionCandidate],
        rng: Any,
        *,
        baseline: NoCardGateSummary | None = None,
    ) -> tuple[list[str], list[AuctionBid]]:
        warm = [
            c.magnitude
            for c in candidates
            if c.magnitude is not None
            and math.isfinite(c.magnitude)
            and c.magnitude > 0.0
        ]
        cold_magnitude = statistics.median(warm) if warm else self._cold_fallback()
        base_a, base_b, baseline_source, no_card_baseline, no_card_n = (
            _decision_baseline(baseline, self.baseline_prior)
        )
        effective_floor = self.ev_floor
        if self.ev_floor_quantile is not None:
            baseline_theta_q = float(
                scipy_beta.ppf(self.ev_floor_quantile, base_a, base_b)
            )
            effective_floor = max(effective_floor, baseline_theta_q * cold_magnitude)
        bid_draws: list[tuple[float, float, float]] = []
        for candidate in candidates:
            u = float(rng.uniform())
            theta = float(
                scipy_beta.ppf(u, candidate.posterior_a, candidate.posterior_b)
            )
            mag = (
                candidate.magnitude
                if candidate.magnitude is not None
                else cold_magnitude
            )
            bid_draws.append((mag, theta, theta * mag))
        winners: list[str] = []
        slate: list[AuctionBid] = []
        for candidate, (mag, theta, bid_value) in zip(candidates, bid_draws):
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
                    prior_source=candidate.prior_source,
                    context_key=candidate.context_key,
                    baseline_source=baseline_source,
                    no_card_baseline=no_card_baseline,
                    no_card_n=no_card_n,
                    rejected_by_ev_floor=not (bid_value > effective_floor),
                    rejected_by_no_card_gate=bid_value > effective_floor
                    and not (theta > base_theta),
                    selection_reason="auction" if selected else "",
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
    deltas rather than a binarized ``p_help x median``. Each candidate draws one
    posterior world ``u ~ U(0, 1)``. Its gate theta is
    ``Beta(posterior_a, posterior_b).ppf(u)`` and its known-card bid is the same
    ``u`` quantile of one batch of weighted bootstrap-EV resamples over its
    support plus a neutral zero pseudo-event. Genuinely cold cards instead bid
    that same gate theta times the round's borrowed gain scale (in-round
    warm-magnitude median, else the primary metric's ``significant_change``,
    else the inert unit — the same cold ladder ``EVThompsonAuctioneer`` climbs).
    Thus bid and gate describe one coherent posterior world. Pricing on the
    resampled mean makes a fat left tail visible: a card whose wins outnumber
    its losses but loses far more per loss bids negative, the failure mode the
    median bid was blind to.
    Projected ``delta_weights`` already contain each event's causal credit times
    its own bank-cycle staleness; the scalar compatibility seam is neutral.
    Stale known deltas therefore fade toward zero without a fresh event reviving
    older history, while genuinely cold cards use the round's cold scale.

    In legacy quantile mode, the EV reserve is two self-normalizing gates,
    neither a Beta assumption nor an absolute scale. (1) A sign gate ``bid >
    0``: never inject a card whose
    sampled EV is non-positive. With exact deltas (all ``deltas_se`` zero or
    absent) this alone yields the abstain-on-all-harmful guarantee — an
    all-negative round wins nothing. When events price evaluation noise, a
    confidently-harmful card (mean far below zero on its se scale) still
    essentially never clears it, while a marginally-harmful one keeps a bounded
    chance of a positive jittered draw — bounded exploration at the boundary,
    with the Sidak no-card gate as a partial backstop. (2) A spread reserve
    ``bid >=`` the
    ``ev_floor_quantile`` quantile of the round's OWN bids: the bid must sit in
    the upper part of the round's realized distribution. The quantile is
    inclusive so ties at the floor do not self-annihilate; true cold bids are
    posterior-sampled, so an all-cold slate no longer becomes a fixed point mass.
    Risk mode replaces both with the per-card condition ``P(EV > 0) >= 1 -
    ev_risk_alpha``, measured on the same bootstrap-EV vector as the bid.

    Draw order is pinned for seed-exact replay: one shared uniform world and,
    for known cards, one bootstrap batch per candidate first, then one slate
    baseline draw. The gate reuses each candidate's world and consumes no
    second posterior draw.
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
    n_bootstrap: int = Field(
        default=512,
        gt=0,
        description="Number of bootstrap-EV resamples whose empirical u-quantile "
        "supplies a known card's coherent Thompson bid.",
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
    ev_reserve_mode: Literal["quantile", "risk"] = Field(
        default="quantile",
        description="EV reserve seam: quantile preserves the round-relative legacy "
        "gate; risk uses only each card's bootstrap-EV sign probability.",
    )
    ev_risk_alpha: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="In risk mode, require P(EV > 0) >= 1 - alpha. Exclusive of 1.0: "
        "alpha=1.0 would zero the sign threshold and admit a provably non-positive "
        "card (risk mode has no separate bid>0 check). None keeps the legacy "
        "quantile mode byte-exact.",
    )

    @model_validator(mode="after")
    def _risk_mode_requires_alpha(self) -> BootstrapThompsonAuctioneer:
        if self.ev_reserve_mode == "risk" and self.ev_risk_alpha is None:
            raise ValueError("ev_risk_alpha is required when ev_reserve_mode='risk'")
        return self

    def _cold_fallback(self) -> float:
        """Cold gain scale for an all-cold round: the primary metric's
        significant_change if declared, else the inert unit placeholder."""
        if self.metrics_context is not None:
            sig = self.metrics_context.get_primary_spec().significant_change
            if sig is not None and math.isfinite(sig) and sig > 0.0:
                return float(sig)
        return _UNSCALED_COLD_MAGNITUDE

    def run(
        self,
        candidates: list[AuctionCandidate],
        rng: Any,
        *,
        baseline: NoCardGateSummary | None = None,
    ) -> tuple[list[str], list[AuctionBid]]:
        warm = [
            c.magnitude
            for c in candidates
            if c.magnitude is not None
            and math.isfinite(c.magnitude)
            and c.magnitude > 0.0
        ]
        cold_magnitude = statistics.median(warm) if warm else self._cold_fallback()
        base_a, base_b, baseline_source, no_card_baseline, no_card_n = (
            _decision_baseline(baseline, self.baseline_prior)
        )
        bids: list[float] = []
        world_thetas: list[float] = []
        positive_probabilities: list[float | None] = []
        support_scales: list[float] = []
        support_kinds: list[str] = []
        support_counts: list[float] = []
        for candidate in candidates:
            u = float(rng.uniform())
            theta = float(
                scipy_beta.ppf(u, candidate.posterior_a, candidate.posterior_b)
            )
            world_thetas.append(theta)
            deltas = candidate.deltas or ()
            weights = candidate.delta_weights
            # Mirror eviction's _effective_support so the probe and eviction
            # lanes partition card-space on one per-event-aged measure.
            support_count = (
                sum(
                    max(0.0, float(weight))
                    for weight in weights
                    if math.isfinite(float(weight))
                )
                if weights is not None
                else float(len(deltas))
            )
            staleness = float(candidate.staleness_weight)
            if math.isfinite(staleness) and staleness >= 0.0:
                support_count *= staleness
            true_cold = not deltas and candidate.magnitude is None
            support_scale = cold_magnitude if true_cold else 0.0
            support_scales.append(support_scale)
            support_counts.append(support_count)
            if true_cold:
                support_kinds.append("cold_prior")
                bids.append(theta * cold_magnitude)
                positive_probabilities.append(
                    1.0 if self.ev_reserve_mode == "risk" else None
                )
            else:
                support_kinds.append("ev_rewards" if deltas else "zero_support")
                # This bootstrap_ev_samples vector feeds both the u-quantile bid and risk gate.
                samples = bootstrap_ev_samples(
                    deltas,
                    support_scale,
                    candidate.staleness_weight,
                    self.n_bootstrap,
                    rng,
                    delta_weights=weights,
                    ses=candidate.deltas_se,
                )
                samples.sort()
                quantile_index = min(int(u * len(samples)), len(samples) - 1)
                bids.append(float(samples[quantile_index]))
                positive_probabilities.append(
                    float(np.mean(samples > 0.0))
                    if self.ev_reserve_mode == "risk"
                    else None
                )
        # Price-adjustment seam (consumes no rng, so draw order stays pinned);
        # applied before the legacy reserve so its floor prices adjusted bids.
        bids = self._adjust_bids(candidates, bids)
        if self.ev_reserve_mode == "quantile":
            quantile_floor = (
                float(np.quantile(bids, self.ev_floor_quantile)) if bids else 0.0
            )
            # Telemetry: the binding reserve a winner actually clears is the sign gate
            # OR-ed with the spread quantile, whichever is higher.
            effective_floor = max(0.0, quantile_floor)
            eligible = [
                bid_value > 0.0 and bid_value >= quantile_floor for bid_value in bids
            ]
        else:
            assert self.ev_risk_alpha is not None
            effective_floor = None
            risk_threshold = 1.0 - self.ev_risk_alpha
            eligible = [
                probability is not None and probability >= risk_threshold
                for probability in positive_probabilities
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
            theta,
            bid_value,
            support_scale,
            support_kind,
            support_n,
            positive_probability,
            can_bid,
        ) in zip(
            candidates,
            world_thetas,
            bids,
            support_scales,
            support_kinds,
            support_counts,
            positive_probabilities,
            eligible,
        ):
            theta_quantile = float(scipy_beta.cdf(theta, base_a, base_b))
            passes_no_card = theta > gate_theta
            selected = can_bid and passes_no_card
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
                    support_n_unstaled=candidate.support_n_unstaled,
                    gain_se=candidate.gain_se,
                    use_count=candidate.use_count,
                    prior_source=candidate.prior_source,
                    context_key=candidate.context_key,
                    baseline_source=baseline_source,
                    no_card_baseline=no_card_baseline,
                    no_card_n=no_card_n,
                    ev_reserve_mode=self.ev_reserve_mode,
                    ev_positive_probability=positive_probability,
                    ev_risk_alpha=(
                        self.ev_risk_alpha if self.ev_reserve_mode == "risk" else None
                    ),
                    rejected_by_ev_floor=not can_bid,
                    rejected_by_no_card_gate=can_bid and not passes_no_card,
                    selection_reason="auction" if selected else "",
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
                    ev_floor=(
                        float(effective_floor) if effective_floor is not None else None
                    ),
                    bids=_bid_dicts(slate),
                )
            )
            reserve = (
                f"EV floor {effective_floor:.4g}"
                if effective_floor is not None
                else f"EV risk alpha {self.ev_risk_alpha:.4g}"
            )
            logger.debug(
                "[Memory][BootstrapAuction] {}/{} candidate(s) beat baseline Beta{} "
                "at {}: {}",
                len(winners),
                len(slate),
                self.baseline_prior,
                reserve,
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

    def _adjust_bids(
        self, candidates: list[AuctionCandidate], bids: list[float]
    ) -> list[float]:
        return bids


class NoveltyDiscountedBootstrapAuctioneer(BootstrapThompsonAuctioneer):
    """Bootstrap auction whose bid pays a novelty tax: each candidate's EV bid
    is scaled by ``(1 + use_count) ** -novelty_power`` before the round's EV
    reserve is computed. In legacy quantile mode that reserve reads the round's
    OWN (taxed) bids, so taxing a dominant repeat winner drags the floor down
    with it — wins redistribute to fresher cards instead of vanishing (offline
    replay on four hover runs, 2026-07-11: injection volume preserved, top-card
    share down 20–35% at power 0.5–1.0). ``use_count`` is the candidate
    projector's deterministic injection count (non-founding gain events —
    event-based, no wall clock, so identical decisions replay identically); an
    unused card pays no tax, and ``novelty_power=0`` is bid-for-bid identical to
    the base auction. The discount consumes no rng. In risk mode its positive
    factor leaves the per-card EV sign probability unchanged."""

    novelty_power: float = Field(
        default=0.5,
        ge=0.0,
        description="Exponent of the (1 + use_count) bid tax; 0 disables the "
        "discount, 1 makes a card's k-th injection bid ~1/(1+k) of raw.",
    )

    def _adjust_bids(
        self, candidates: list[AuctionCandidate], bids: list[float]
    ) -> list[float]:
        if self.novelty_power == 0.0:
            return bids
        return [
            bid * (1.0 + candidate.use_count) ** -self.novelty_power
            for candidate, bid in zip(candidates, bids)
        ]


class PendingDiscountedBootstrapAuctioneer(BootstrapThompsonAuctioneer):
    """Bootstrap auction whose bid pays for uncredited in-flight exposure.

    Each candidate's EV bid is scaled by
    ``(1 + pending_count) ** -pending_power`` before the round's EV reserve is
    computed. ``pending_count`` excludes the current selection because the
    provider snapshots lease counts before attaching this round's winners.
    ``pending_power=0`` is bid-for-bid identical to the base auction, and the
    adjustment consumes no rng.
    """

    pending_power: float = Field(
        default=0.0,
        ge=0.0,
        description="Exponent of the (1 + pending_count) bid tax; 0 disables "
        "the discount.",
    )

    def _adjust_bids(
        self, candidates: list[AuctionCandidate], bids: list[float]
    ) -> list[float]:
        if self.pending_power == 0.0:
            return bids
        return [
            bid * (1.0 + candidate.pending_count) ** -self.pending_power
            for candidate, bid in zip(candidates, bids)
        ]


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
