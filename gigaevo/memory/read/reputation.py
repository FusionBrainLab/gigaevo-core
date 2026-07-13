"""Per-card efficacy statistics derived from injection outcomes.

Owner of the gain → downside-posterior math: ``block_from_events`` turns a
card's gain events into the reputation block (median magnitude plus a downside
posterior that counts every below-zero gain as harm) that the auction and
renderer read. ``BetaBinomialReputation`` binds that math to injectable
thresholds; ``BDProximityReputation`` partitions it by the query parent's
MAP-Elites cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.context import BDCellMemoryContext
from gigaevo.memory.context.evidence import (
    event_weight,
    harm_mass,
    median,
    sign_help_counts,
    split_events_by_task,
)
from gigaevo.memory.read.bootstrap import bootstrap_ev_samples, stable_rng
from gigaevo.memory.read.interfaces import EvictionFacingReputation
from gigaevo.memory.read.staleness import bank_cycle_event_weights
from gigaevo.memory.storage.base import MemoryStore


def _use_gains(events: Sequence[ContextualGain]) -> tuple[float, ...]:
    """Valid non-founding use deltas that define the display magnitude."""
    return tuple(
        float(e.gain)
        for e in events
        if not e.invalid
        and not e.founding
        and not e.unused
        and e.gain is not None
        and math.isfinite(float(e.gain))
    )


def _use_gain_weights(events: Sequence[ContextualGain]) -> tuple[float, ...]:
    """Causal weights aligned with :func:`_use_gains`."""
    return tuple(
        event_weight(e)
        for e in events
        if not e.invalid
        and not e.founding
        and not e.unused
        and e.gain is not None
        and math.isfinite(float(e.gain))
    )


def _ev_rewards(events: Sequence[ContextualGain]) -> tuple[float, ...]:
    """EV support for the bootstrap auction.

    Founding evidence is origin/audit evidence and is deliberately excluded.
    Real direct use contributes its finite gain. Invalid or unused prompt
    exposure contributes a zero atom, which makes the card no longer cold
    without inventing a gain magnitude.
    """
    rewards: list[float] = []
    for event in events:
        if event.founding:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            rewards.append(0.0)
        elif event.gain is not None and math.isfinite(float(event.gain)):
            rewards.append(float(event.gain))
    return tuple(rewards)


def _ev_events(events: Sequence[ContextualGain]) -> tuple[ContextualGain, ...]:
    """The concrete event rows aligned with :func:`_ev_rewards`."""
    out: list[ContextualGain] = []
    for event in events:
        if event.founding:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            out.append(event)
        elif event.gain is not None and math.isfinite(float(event.gain)):
            out.append(event)
    return tuple(out)


def _ev_reward_weights(events: Sequence[ContextualGain]) -> tuple[float, ...]:
    weights: list[float] = []
    for event in events:
        if event.founding:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            weights.append(weight)
        elif event.gain is not None and math.isfinite(float(event.gain)):
            weights.append(weight)
    return tuple(weights)


def _coerce_gain_se(raw: float | None) -> float | None:
    """Preserve unknown se; sanitize non-positive/non-finite numbers to exact."""
    if raw is None:
        return None
    se = float(raw)
    return se if math.isfinite(se) and se > 0.0 else 0.0


def _ev_ses(events: Sequence[ContextualGain]) -> tuple[float | None, ...]:
    """Per-atom gain ses aligned with :func:`_ev_rewards` (0 exact, None unknown).

    Invalid/unused zero atoms are exact binary observations; a non-finite or
    negative stored se degrades to exact rather than poisoning the resample.
    """
    ses: list[float | None] = []
    for event in events:
        if event.founding:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            ses.append(0.0)
        elif event.gain is not None and math.isfinite(float(event.gain)):
            ses.append(_coerce_gain_se(event.gain_se))
    return tuple(ses)


def _moment_matched_beta(
    a0: float, b0: float, sigma2_k: float, n_events: float
) -> tuple[float, float]:
    """Beta(a, b) matching the latent-sign mixture's first two moments.

    Mean ``a0 / N`` and variance ``(a0*b0 + N*sigma2_k) / (N**2 * (N+1))`` with
    ``N = a0 + b0`` and ``sigma2_k`` the summed per-event sign variance. Exact
    (sign-known) evidence has ``sigma2_k == 0`` and returns ``(a0, b0)``
    unchanged, so the soft-count / legacy path is byte-exact. The same
    matching folds foreign hard-sign counts into the native moments, keeping
    the read posterior and its foreign extension on one formula.
    """
    total = a0 + b0
    if sigma2_k <= 0.0 or n_events == 0 or a0 <= 0 or b0 <= 0:
        return a0, b0
    matched_total = a0 * b0 * (total + 1.0) / (a0 * b0 + total * sigma2_k) - 1.0
    mean = a0 / total
    matched_a = mean * matched_total
    matched_b = (1.0 - mean) * matched_total
    if (
        matched_a > 0
        and matched_b > 0
        and math.isfinite(matched_a)
        and math.isfinite(matched_b)
    ):
        return matched_a, matched_b
    return a0, b0


def beta_binomial_posterior(
    gains: Sequence[float],
    *,
    prior: tuple[float, float] = (1.0, 1.0),
    threshold: float = 0.0,
    weights: Sequence[float] | None = None,
    staleness_weights: Sequence[float] | None = None,
    event_ses: Sequence[float | None] | None = None,
    invalid_events: float = 0.0,
    unused_events: float = 0.0,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
    harm_model: str = "soft_count",
) -> CardStatsBlock:
    """Downside Beta-Binomial posterior on P(not harmful) from per-event gains.

    ``a = prior_a + (n - k_harm)``, ``b = prior_b + k_harm`` with ``k_harm``
    the summed per-event harm mass: the probability the event's true gain lies
    below ``threshold`` (default 0) — the exact below-threshold indicator for
    an exact event (its ``event_ses`` entry 0 or absent, the historical strict
    sign test), the Gaussian tail mass when the entry prices evaluation noise,
    and the maximally uncertain ``Phi(0)`` mass when the entry is ``None``.
    With ``harm_model="mixture"``, uncertain event signs are independent latent
    Bernoulli variables and the returned Beta matches the resulting mixture's
    exact first two moments. ``efficacy_confident`` iff the
    ``confident_quantile`` of Beta(a, b) exceeds ``confident_threshold``. The
    ``p_help_lo20`` field name is part of the serialized-card stats contract
    regardless of the configured quantile.
    ``invalid_events`` are evaluated-and-judged-invalid children;
    ``unused_events`` are prompt exposures the mutator ignored. Both are forced
    failure observations with no gain magnitude and no se.
    """
    raw_weights = tuple(weights) if weights is not None else ()
    if raw_weights and len(raw_weights) != len(gains):
        raise ValueError(
            f"weights length {len(raw_weights)} != gains length {len(gains)}"
        )
    raw_staleness = tuple(staleness_weights) if staleness_weights is not None else None
    if raw_staleness is not None and len(raw_staleness) != len(gains):
        raise ValueError(
            "staleness_weights length "
            f"{len(raw_staleness)} != gains length {len(gains)}"
        )
    raw_ses = tuple(event_ses) if event_ses is not None else ()
    if raw_ses and len(raw_ses) != len(gains):
        raise ValueError(
            f"event_ses length {len(raw_ses)} != gains length {len(gains)}"
        )
    finite: list[tuple[float, float, float | None]] = []
    for idx, gain in enumerate(gains):
        if gain is None or not math.isfinite(float(gain)):
            continue
        credit = float(raw_weights[idx]) if raw_weights else 1.0
        age = float(raw_staleness[idx]) if raw_staleness is not None else 1.0
        weight = credit * age
        se = _coerce_gain_se(raw_ses[idx]) if raw_ses else 0.0
        if math.isfinite(weight) and weight > 0.0:
            finite.append((float(gain), weight, se))
    forced_failures = invalid_events + unused_events
    n = sum(weight for _, weight, _ in finite) + forced_failures
    k_harm = (
        sum(weight * harm_mass(g, se, threshold) for g, weight, se in finite)
        + forced_failures
    )
    prior_a, prior_b = prior
    a0 = prior_a + (n - k_harm)
    b0 = prior_b + k_harm
    sigma2_k = 0.0
    if harm_model == "mixture":
        sigma2_k = sum(
            weight**2 * (p_harm := harm_mass(gain, se, threshold)) * (1.0 - p_harm)
            for gain, weight, se in finite
        )
    a, b = _moment_matched_beta(a0, b0, sigma2_k, n)
    lo = float(beta.ppf(confident_quantile, a, b)) if n else float("nan")
    return CardStatsBlock(
        posterior_a=a,
        posterior_b=b,
        intro_events=n,
        k_harm=k_harm,
        p_help_mean=a / (a + b),
        p_help_lo20=lo,
        efficacy_confident=bool(n and lo > confident_threshold),
    )


def block_from_events(
    events: Sequence[ContextualGain],
    *,
    prior: tuple[float, float] = (1.0, 1.0),
    staleness_weights: Sequence[float] | None = None,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
    harm_model: str = "soft_count",
) -> CardStatsBlock | None:
    """Global, unadjusted card block from its gain events: median magnitude plus
    the downside posterior, harm being each event's mass below zero — the exact
    sign indicator for an exact event (``gain_se == 0``, the historical strict
    sign test), the Gaussian tail mass when the event prices evaluation noise,
    and the maximally uncertain ``Phi(0)`` mass when its se is unknown.
    Each event's harm mass is its own — one event never shifts another's
    verdict — so the harm count is monotone in the events and a uniformly-losing
    card counts every loss; eval noise is absorbed by the counting posterior
    (``harm_min_events`` plus the optimistic harm quantile) and, when priced,
    by the per-event tail mass — never a per-card dead band, which cannot tell
    noise-level losses from genuine ones using one card's own gains.
    Invalid events are forced harm with no magnitude. A block is efficacy-
    confident only when the downside posterior is confident AND the median gain
    is a genuine positive (a zero/negative median is a no-op, never a confident
    win). Returns ``None`` for a card with no events (no evidence, no block).

    Magnitude is the median of the card's *use* events (real injection
    outcomes). The one-time founding delta is origin evidence measured before
    the card existed, so it does not enter posterior, confidence, or EV. A
    founding-only card is therefore still statistically cold; catastrophic
    founding failures are handled by the write-side birth-failure evictor.
    """
    ages = tuple(staleness_weights) if staleness_weights is not None else None
    if ages is not None and len(ages) != len(events):
        raise ValueError(
            "staleness_weights must align with events: "
            f"{len(ages)} weights for {len(events)} events"
        )
    indexed_proof = [
        (idx, event) for idx, event in enumerate(events) if not event.founding
    ]
    proof_events = [event for _, event in indexed_proof]
    if not proof_events:
        return None
    valid = [
        (idx, event)
        for idx, event in indexed_proof
        if not event.invalid and not event.unused
    ]

    def aged_weight(idx: int, event: ContextualGain) -> float:
        age = float(ages[idx]) if ages is not None else 1.0
        weight = event_weight(event) * age
        return weight if math.isfinite(weight) and weight > 0.0 else 0.0

    invalid_events = sum(
        aged_weight(idx, event) for idx, event in indexed_proof if event.invalid
    )
    unused_events = sum(
        aged_weight(idx, event)
        for idx, event in indexed_proof
        if event.unused and not event.invalid
    )
    valid_gains = [float(event.gain) for _, event in valid]
    valid_weights = [event_weight(event) for _, event in valid]
    valid_ages = [float(ages[idx]) for idx, _ in valid] if ages is not None else None
    valid_ses = [_coerce_gain_se(event.gain_se) for _, event in valid]
    block = beta_binomial_posterior(
        valid_gains,
        prior=prior,
        weights=valid_weights,
        staleness_weights=valid_ages,
        event_ses=valid_ses,
        invalid_events=invalid_events,
        unused_events=unused_events,
        confident_quantile=confident_quantile,
        confident_threshold=confident_threshold,
        harm_model=harm_model,
    )
    use_gains = _use_gains(events)
    magnitude: float | None
    expired = ages is not None and block.intro_events < 1.0
    if expired:
        magnitude = None
    elif use_gains:
        magnitude = median(use_gains)
    elif invalid_events or unused_events:
        magnitude = 0.0
    else:
        magnitude = None
    return block.model_copy(
        update={
            "IntroGain_best_median": magnitude,
            "efficacy_confident": block.efficacy_confident
            and magnitude is not None
            and magnitude > 0,
        }
    )


def _block_from_partition(
    native: Sequence[ContextualGain],
    foreign: Sequence[ContextualGain],
    *,
    prior: tuple[float, float] = (1.0, 1.0),
    native_staleness_weights: Sequence[float] | None = None,
    foreign_staleness_weights: Sequence[float] | None = None,
    confident_quantile: float,
    confident_threshold: float,
    harm_model: str = "soft_count",
) -> CardStatsBlock | None:
    block = block_from_events(
        native,
        prior=prior,
        staleness_weights=native_staleness_weights,
        confident_quantile=confident_quantile,
        confident_threshold=confident_threshold,
        harm_model=harm_model,
    )
    foreign_help, foreign_total = sign_help_counts(
        foreign, staleness_weights=foreign_staleness_weights
    )
    if foreign_total <= 0.0:
        return block
    if block is None:
        block = beta_binomial_posterior(
            (),
            prior=prior,
            confident_quantile=confident_quantile,
            confident_threshold=confident_threshold,
            harm_model=harm_model,
        )
    assert block.posterior_a is not None and block.posterior_b is not None
    # Fold foreign hard-sign counts into the NATIVE moment inputs, not onto the
    # variance-shrunk matched Beta. Under the mixture the matched total
    # S = a*+b* is below the true native sample total N, so adding hard counts
    # to S over-weights foreign evidence. Reconstruct (a0, b0, N) from the
    # block, recover the native sign variance by inverting the moment match
    # (S == N under soft-count => sigma2_k == 0 => byte-identical legacy fold),
    # then re-match with foreign counts added to the exact moments.
    prior_a, prior_b = prior
    n_native = float(block.intro_events)
    a0 = prior_a + (n_native - block.k_harm)
    b0 = prior_b + block.k_harm
    native_total = a0 + b0
    matched_total = float(block.posterior_a) + float(block.posterior_b)
    sigma2_k = (
        a0
        * b0
        * (native_total - matched_total)
        / (native_total * (matched_total + 1.0))
        if native_total > 0.0 and matched_total > -1.0
        else 0.0
    )
    a, b = _moment_matched_beta(
        a0 + foreign_help,
        b0 + (foreign_total - foreign_help),
        sigma2_k,
        n_native + foreign_total,
    )
    lo = float(beta.ppf(confident_quantile, a, b))
    magnitude = block.IntroGain_best_median
    return block.model_copy(
        update={
            "posterior_a": a,
            "posterior_b": b,
            "p_help_mean": a / (a + b),
            "p_help_lo20": lo,
            "efficacy_confident": bool(
                block.intro_events > 0.0
                and lo > confident_threshold
                and magnitude is not None
                and magnitude > 0.0
            ),
            "foreign_help_events": foreign_help,
            "foreign_total_events": foreign_total,
        }
    )


def _task_block(
    events: Sequence[ContextualGain],
    task_key: str,
    *,
    prior: tuple[float, float] = (1.0, 1.0),
    staleness_weights: Sequence[float] | None = None,
    confident_quantile: float,
    confident_threshold: float,
    harm_model: str = "soft_count",
) -> CardStatsBlock | None:
    native, foreign = split_events_by_task(events, task_key)
    native_staleness: tuple[float, ...] | None = None
    foreign_staleness: tuple[float, ...] | None = None
    if staleness_weights is not None:
        ages = tuple(staleness_weights)
        if len(ages) != len(events):
            raise ValueError(
                "staleness_weights must align with events: "
                f"{len(ages)} weights for {len(events)} events"
            )
        native_staleness = tuple(
            age
            for event, age in zip(events, ages)
            if event.context.task_key == task_key
        )
        foreign_staleness = tuple(
            age
            for event, age in zip(events, ages)
            if event.context.task_key != task_key
        )
    return _block_from_partition(
        native,
        foreign,
        prior=prior,
        native_staleness_weights=native_staleness,
        foreign_staleness_weights=foreign_staleness,
        confident_quantile=confident_quantile,
        confident_threshold=confident_threshold,
        harm_model=harm_model,
    )


class BetaBinomialReputation(BaseModel):
    """Downside Beta-Binomial reputation over per-card injection gains.

    Configurable façade over :func:`block_from_events` — one implementation,
    bound here to injectable thresholds. Also the single home of the
    ``is_confidently_harmful`` predicate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    harm_min_events: int = Field(
        default=3,
        description="Minimum intro events before a card can be judged harmful.",
    )
    harm_quantile: float = Field(
        default=0.80,
        description="Optimistic posterior quantile used by the harm predicate.",
    )
    harm_threshold: float = Field(
        default=0.5,
        description="Harmful iff the optimistic P(not harmful) read stays below this.",
    )
    harm_model: Literal["soft_count", "mixture"] = Field(
        default="soft_count",
        description="Uncertain signs use legacy soft counts or an exact-moment mixture.",
    )
    confident_quantile: float = Field(
        default=0.20,
        description="Pessimistic posterior quantile used for the confidence flag.",
    )
    confident_threshold: float = Field(
        default=0.5,
        description="Confident iff the pessimistic P(help) read clears this.",
    )
    cold_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="(alpha, beta) Beta prior assumed for cards with no stamped posterior.",
    )
    prior: Any | None = Field(
        default=None,
        description="Optional cold-card prior policy; when set, the warm posterior and decay shrink toward it instead of Beta(1,1) so cold and warm are one coherent world.",
    )

    @property
    def requires_decision_context(self) -> bool:
        """Whether global eviction decisions need an explicit read context."""

        return False

    @property
    def policy_min_effective_events(self) -> float:
        """Default evidence floor for active-policy cleanup."""

        return float(self.harm_min_events)

    def posterior(
        self, gains: Sequence[float], *, threshold: float = 0.0
    ) -> CardStatsBlock:
        return beta_binomial_posterior(
            gains,
            threshold=threshold,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
            harm_model=self.harm_model,
        )

    def prior_base(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]:
        if self.prior is None:
            return (1.0, 1.0)
        return self.prior.cold_card_prior(card, context).as_tuple()

    def _card_stats_with_prior(
        self,
        card: Card,
        context: DecisionContext | None,
        *,
        prior: tuple[float, float],
        staleness_weights: Sequence[float] | None = None,
    ) -> CardStatsBlock | None:
        task_key = context.task_key if context is not None else ""
        return _task_block(
            card.gain_events,
            task_key,
            prior=prior,
            staleness_weights=staleness_weights,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
            harm_model=self.harm_model,
        )

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        """Resolve native magnitudes plus sign-only foreign evidence."""
        a0, b0 = self.prior_base(card, context)
        return self._card_stats_with_prior(
            card,
            context,
            prior=(a0, b0),
        )

    def card_stats_with_staleness(
        self,
        card: Card,
        context: DecisionContext | None = None,
        *,
        staleness_weights: Sequence[float],
    ) -> CardStatsBlock | None:
        a0, b0 = self.prior_base(card, context)
        return self._card_stats_with_prior(
            card,
            context,
            prior=(a0, b0),
            staleness_weights=staleness_weights,
        )

    def posterior_of(self, block: CardStatsBlock | None) -> tuple[float, float]:
        """(alpha, beta) of a resolved block's downside posterior; ``cold_prior``
        when absent. Pure projection over an already-resolved ``card_stats``
        block, so one resolve per candidate serves the auction and the render
        alike."""
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return self.cold_prior
        a = float(block.posterior_a)
        b = float(block.posterior_b)
        # Beta(a, b) requires finite a > 0, b > 0; a corrupt stamped block
        # would otherwise raise inside the auction's rng.beta draw.
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            return self.cold_prior
        return (a, b)

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None:
        """A resolved block's expected gain (``IntroGain_best_median``) — the EV
        auction's magnitude. ``None`` when the block is absent or carries no use
        evidence (no events, or founding-only birth evidence — either sign), so
        the auction falls back to its borrowed gain scale."""
        if block is None or block.IntroGain_best_median is None:
            return None
        return float(block.IntroGain_best_median)

    def card_posterior(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]:
        """``posterior_of`` composed with ``card_stats`` — one-card convenience
        for callers outside the reader's batched resolve."""
        return self.posterior_of(self.card_stats(card, context))

    def card_magnitude(
        self, card: Card, context: DecisionContext | None = None
    ) -> float | None:
        """``magnitude_of`` composed with ``card_stats`` — one-card convenience
        for callers outside the reader's batched resolve."""
        return self.magnitude_of(self.card_stats(card, context))

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """Bootstrap EV support values.

        Empty means genuinely cold or founding-only. Unused/invalid exposures
        contribute zero support so they do not receive another cold-start bid.
        """
        task_key = context.task_key if context is not None else ""
        native, _ = split_events_by_task(card.gain_events, task_key)
        return _ev_rewards(native)

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """Causal weights aligned with :meth:`event_deltas`."""
        task_key = context.task_key if context is not None else ""
        native, _ = split_events_by_task(card.gain_events, task_key)
        return _ev_reward_weights(native)

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        """Concrete event rows behind :meth:`event_deltas`."""
        task_key = context.task_key if context is not None else ""
        native, _ = split_events_by_task(card.gain_events, task_key)
        return _ev_events(native)

    def event_ses(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float | None, ...]:
        """Per-atom gain ses aligned with deltas (0 exact, None unknown)."""
        task_key = context.task_key if context is not None else ""
        native, _ = split_events_by_task(card.gain_events, task_key)
        return _ev_ses(native)

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        """Contexts the write-side active-policy cleanup can evaluate.

        Global reputation has one value surface, so one contextless evaluation
        is the complete active-policy view.
        """
        del card
        return (None,)

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """Unit ages aligned with EV evidence; the base owns no store."""
        return (1.0,) * len(self.evidence_events(card, context))

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool:
        """True iff the resolved stats block excludes the card as harmful: at
        least ``harm_min_events`` intro events and even the optimistic
        ``harm_quantile`` read of P(not harmful) stays below ``harm_threshold``.
        A missing block or one without posterior parameters is never harmful."""
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return False
        a = float(block.posterior_a)
        b = float(block.posterior_b)
        # Beta(a, b) requires finite a > 0, b > 0; beta.ppf on a degenerate
        # (0/negative) posterior returns nan, and ``nan < threshold`` is False —
        # which would silently read a corrupt block as "never harmful".
        if block.intro_events < self.harm_min_events or not (
            math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0
        ):
            return False
        return float(beta.ppf(self.harm_quantile, a, b)) < self.harm_threshold


class BDProximityReputation(BetaBinomialReputation):
    """Read-time BD-cell partitioned reputation (the contextual bandit's value
    channel).

    Re-buckets each card's stored ``gain_events`` into the query parent's
    *current* MAP-Elites cell via the run's own ``behavior_space.get_cell`` and
    bids over the in-cell subset only — a card that helped near cell A and hurt
    near cell B bids high in A and abstains in B from the same stored list. A
    parent cell with no in-cell event delegates byte-for-byte to ``fallback``.

    The cell is recomputed every read from the immutable ``parent_metrics``
    under a per-read snapshot of the ``behavior_space``'s bounds — the bandit
    reads one frozen tessellation, never stores a cell id (``DynamicBehaviorSpace``
    moves cells on every reindex).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    behavior_space: BehaviorSpace = Field(
        description="The run's tessellation; bucketing reads its CURRENT bounds.",
    )
    fallback: BetaBinomialReputation = Field(
        default_factory=BetaBinomialReputation,
        description="Cold-cell delegate: the global event-derived reputation.",
    )

    @property
    def requires_decision_context(self) -> bool:
        """BD-local values are only meaningful for a concrete parent context."""

        return True

    @property
    def policy_min_effective_events(self) -> float:
        return float(self.harm_min_events)

    @cached_property
    def _cell_context(self) -> BDCellMemoryContext:
        return BDCellMemoryContext(behavior_space=self.behavior_space)

    def _in_cell(
        self, card: Card, context: DecisionContext | None
    ) -> list[ContextualGain] | None:
        # The pre-guard matters: without a context or events the context model
        # would fall back to the card's full global event list, but this
        # reputation must delegate those reads byte-for-byte to ``fallback``.
        if context is None or not card.gain_events:
            return None
        native, _ = split_events_by_task(card.gain_events, context.task_key)
        if not native:
            return None
        scoped = card.model_copy(update={"gain_events": native})
        local = self._cell_context.local_evidence_events(scoped, context)
        return list(local) if local else None

    def _card_stats(
        self,
        card: Card,
        context: DecisionContext | None,
        staleness_weights: Sequence[float] | None,
    ) -> CardStatsBlock | None:
        a0, b0 = self.prior_base(card, context)
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback._card_stats_with_prior(
                card,
                context,
                prior=(a0, b0),
                staleness_weights=staleness_weights,
            )
        # Same global block math as the base reputation, but over the in-cell
        # subset only: the cell partition already controls for context, so the
        # harm count and median magnitude are measured BD-locally rather than
        # against a parent-fitness counterfactual. Cold cells delegated above.
        assert context is not None
        native, foreign = split_events_by_task(card.gain_events, context.task_key)
        in_cell_staleness: tuple[float, ...] | None = None
        foreign_staleness: tuple[float, ...] | None = None
        if staleness_weights is not None:
            ages = tuple(staleness_weights)
            if len(ages) != len(card.gain_events):
                raise ValueError(
                    "staleness_weights must align with events: "
                    f"{len(ages)} weights for {len(card.gain_events)} events"
                )
            native_ages = tuple(
                age
                for event, age in zip(card.gain_events, ages)
                if event.context.task_key == context.task_key
            )
            selected = {id(event) for event in in_cell}
            in_cell_staleness = tuple(
                age for event, age in zip(native, native_ages) if id(event) in selected
            )
            foreign_staleness = tuple(
                age
                for event, age in zip(card.gain_events, ages)
                if event.context.task_key != context.task_key
            )
            if len(in_cell_staleness) != len(in_cell):
                raise ValueError("in-cell evidence must preserve event identity")
        return _block_from_partition(
            in_cell,
            foreign,
            prior=(a0, b0),
            native_staleness_weights=in_cell_staleness,
            foreign_staleness_weights=foreign_staleness,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
            harm_model=self.harm_model,
        )

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        return self._card_stats(card, context, None)

    def card_stats_with_staleness(
        self,
        card: Card,
        context: DecisionContext | None = None,
        *,
        staleness_weights: Sequence[float],
    ) -> CardStatsBlock | None:
        return self._card_stats(card, context, staleness_weights)

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """The card's in-cell EV support values — the bootstrap resample support
        partitioned by the query parent's MAP-Elites cell, exactly the subset
        ``card_stats`` prices here. Falls back to the global deltas when the
        parent cell holds no in-cell event (the cold-cell delegation above)."""
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.event_deltas(card, context)
        return _ev_rewards(in_cell)

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.event_weights(card, context)
        return _ev_reward_weights(in_cell)

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.evidence_events(card, context)
        return _ev_events(in_cell)

    def event_ses(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float | None, ...]:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.event_ses(card, context)
        return _ev_ses(in_cell)

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        """One observed context per BD evidence cell.

        Policy-nonviable eviction is allowed to retire a BD-local card only
        from contexts where the card already has concrete non-founding EV
        evidence. It never asks the fallback global view to delete a card that
        might still be useful in an unobserved cell.
        """
        return tuple(
            event.context
            for event in self._cell_context.evidence_cells(_ev_events(card.gain_events))
        )


class BootstrapReputation:
    """Bootstrap-EV reputation: re-prices the gain summary a card's block carries
    on the mean and low quantile of its weighted-bootstrap expected-gain
    distribution, so every reader that already trusts the block — the shortlist
    bench, the renderer endorsement, the auction's cold-magnitude borrow — sees
    the same tail-aware EV the bootstrap auction bids on, without re-deriving it.

    Decorates any :class:`EvictionFacingReputation` (global, BD-local, or
    decayed). The
    inner reputation still owns the downside Beta posterior (the harm gate reads
    it unchanged via ``is_confidently_harmful``) and the raw event deltas; this
    layer adds explicit bootstrap-EV fields: ``IntroGain_bootstrap_ev_mean`` is
    the central EV the auction bids, ``IntroGain_bootstrap_ev_lo20`` is the
    pessimistic EV the fused ranking floors on, and
    ``IntroGain_bootstrap_ev_hi80`` is the optimistic EV the bench requires to
    be non-positive before retiring a card. The inner ``p_help_*`` probability
    fields remain probabilities; EV is never written into them. The observed
    median stays in ``IntroGain_best_median`` so rendering does not call an EV a
    median.

    Staleness reuses ``bank_cycle_event_weights`` (shared with
    :class:`~gigaevo.memory.read.decay.DecayingReputation`): each discount
    ``w_i = 2**(-s_i/H)`` multiplies only its aligned event's causal credit, so
    old deltas fade without a fresh event reviving them. A card with no
    non-founding evidence keeps its inner block untouched; the write-side
    birth-failure evictor handles catastrophic founding losses.
    """

    def __init__(
        self,
        inner: EvictionFacingReputation,
        store: MemoryStore,
        *,
        half_life_cycles: float = 1.0,
        ev_lo_quantile: float = 0.20,
        n_bootstrap: int = 512,
        confident_min_events: int = 3,
    ) -> None:
        if half_life_cycles <= 0:
            raise ValueError(
                f"half_life_cycles must be positive, got {half_life_cycles}"
            )
        if not 0.0 <= ev_lo_quantile < 1.0:
            raise ValueError(f"ev_lo_quantile must be in [0, 1), got {ev_lo_quantile}")
        if n_bootstrap <= 0:
            raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
        if confident_min_events < 1:
            raise ValueError(
                f"confident_min_events must be positive, got {confident_min_events}"
            )
        self._inner = inner
        self._store = store
        self._half_life_cycles = half_life_cycles
        self._ev_lo_quantile = ev_lo_quantile
        self._n_bootstrap = n_bootstrap
        self._confident_min_events = confident_min_events

    @property
    def requires_decision_context(self) -> bool:
        return self._inner.requires_decision_context

    @property
    def policy_min_effective_events(self) -> float:
        return max(
            float(self._confident_min_events),
            float(self._inner.policy_min_effective_events),
        )

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        block = self._inner.card_stats(card, context)
        if block is None:
            return None
        deltas = self._inner.event_deltas(card, context)
        if not deltas:
            # No non-founding EV support to bootstrap. Leave the inner block as
            # is; true cold exploration is owned by the auction, while severe
            # founding losses are deleted on the write side.
            return block
        weights = self._inner.event_weights(card, context)
        staleness = self.staleness_weights(card, context)
        if len(weights) != len(deltas) or len(staleness) != len(deltas):
            raise ValueError(
                "event_weights and staleness_weights must align with event_deltas"
            )
        combined = tuple(
            float(credit) * float(age) for credit, age in zip(weights, staleness)
        )
        # Deterministic per-card seed: the block's pessimistic EV must be
        # reproducible read-to-read and must never consume the live auction
        # round's RNG stream.
        rng = stable_rng(card.id, len(deltas), self._n_bootstrap)
        samples = bootstrap_ev_samples(
            deltas,
            0.0,
            1.0,
            self._n_bootstrap,
            rng,
            delta_weights=combined,
            ses=self._inner.event_ses(card, context),
        )
        ev_mean = float(samples.mean())
        ev_lo = float(np.quantile(samples, self._ev_lo_quantile))
        ev_hi = float(np.quantile(samples, 1.0 - self._ev_lo_quantile))
        effective_events = sum(combined)
        return block.model_copy(
            update={
                "IntroGain_bootstrap_ev_mean": ev_mean,
                "IntroGain_bootstrap_ev_lo20": ev_lo,
                "IntroGain_bootstrap_ev_hi80": ev_hi,
                "efficacy_confident": bool(
                    effective_events >= self._confident_min_events
                    and block.efficacy_confident
                    and ev_lo > 0.0
                ),
            }
        )

    def posterior_of(self, block: CardStatsBlock | None) -> tuple[float, float]:
        return self._inner.posterior_of(block)

    def prior_base(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]:
        return self._inner.prior_base(card, context)

    def magnitude_of(self, block: CardStatsBlock | None) -> float | None:
        if block is not None and block.IntroGain_bootstrap_ev_mean is not None:
            return float(block.IntroGain_bootstrap_ev_mean)
        return self._inner.magnitude_of(block)

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool:
        return self._inner.is_confidently_harmful(block)

    def event_deltas(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        return self._inner.event_deltas(card, context)

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        return self._inner.event_weights(card, context)

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        return self._inner.evidence_events(card, context)

    def event_ses(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float | None, ...]:
        return self._inner.event_ses(card, context)

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        return self._inner.eviction_contexts(card)

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        task_key = context.task_key if context is not None else ""
        events = self.evidence_events(card, context)
        return bank_cycle_event_weights(
            events,
            self._store.snapshot(),
            self._half_life_cycles,
            task_key=task_key,
        )
