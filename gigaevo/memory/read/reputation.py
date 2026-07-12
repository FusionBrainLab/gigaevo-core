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

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta, norm

from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.context import BDCellMemoryContext
from gigaevo.memory.context.evidence import event_weight, median
from gigaevo.memory.read.bootstrap import bootstrap_ev_samples, stable_rng
from gigaevo.memory.read.interfaces import EvictionFacingReputation
from gigaevo.memory.read.staleness import bank_cycle_weight
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


def _ev_ses(events: Sequence[ContextualGain]) -> tuple[float, ...]:
    """Per-atom gain ses aligned with :func:`_ev_rewards` (0 = exact).

    Invalid/unused zero atoms are exact binary observations; a non-finite or
    negative stored se degrades to exact rather than poisoning the resample.
    """
    ses: list[float] = []
    for event in events:
        if event.founding:
            continue
        weight = event_weight(event)
        if weight <= 0.0:
            continue
        if event.invalid or event.unused:
            ses.append(0.0)
        elif event.gain is not None and math.isfinite(float(event.gain)):
            se = float(event.gain_se)
            ses.append(se if math.isfinite(se) and se > 0.0 else 0.0)
    return tuple(ses)


def _harm_mass(gain: float, se: float, threshold: float) -> float:
    """P(true gain < threshold) for one event: the exact below-threshold
    indicator when the gain is exact (``se == 0``), else the Gaussian tail
    mass. The explicit exact branch keeps ``gain == threshold`` from
    evaluating 0/0."""
    if se <= 0.0:
        return 1.0 if gain < threshold else 0.0
    return float(norm.cdf((threshold - gain) / se))


def beta_binomial_posterior(
    gains: Sequence[float],
    *,
    threshold: float = 0.0,
    weights: Sequence[float] | None = None,
    event_ses: Sequence[float] | None = None,
    invalid_events: float = 0.0,
    unused_events: float = 0.0,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
) -> CardStatsBlock:
    """Downside Beta-Binomial posterior on P(not harmful) from per-event gains.

    ``a = 1 + (n - k_harm)``, ``b = 1 + k_harm`` with ``k_harm`` the summed
    per-event harm mass: the probability the event's true gain lies below
    ``threshold`` (default 0) — the exact below-threshold indicator for an
    exact event (its ``event_ses`` entry 0 or absent, the historical strict
    sign test), the Gaussian tail mass when the entry prices evaluation noise.
    ``efficacy_confident`` iff the ``confident_quantile`` of Beta(a, b) exceeds
    ``confident_threshold``. The ``p_help_lo20`` field name is part of the
    serialized-card stats contract regardless of the configured quantile.
    ``invalid_events`` are evaluated-and-judged-invalid children;
    ``unused_events`` are prompt exposures the mutator ignored. Both are forced
    failure observations with no gain magnitude and no se.
    """
    raw_weights = tuple(weights) if weights is not None else ()
    if raw_weights and len(raw_weights) != len(gains):
        raise ValueError(
            f"weights length {len(raw_weights)} != gains length {len(gains)}"
        )
    raw_ses = tuple(event_ses) if event_ses is not None else ()
    if raw_ses and len(raw_ses) != len(gains):
        raise ValueError(
            f"event_ses length {len(raw_ses)} != gains length {len(gains)}"
        )
    finite: list[tuple[float, float, float]] = []
    for idx, gain in enumerate(gains):
        if gain is None or not math.isfinite(float(gain)):
            continue
        weight = float(raw_weights[idx]) if raw_weights else 1.0
        se = float(raw_ses[idx]) if raw_ses else 0.0
        if not (math.isfinite(se) and se > 0.0):
            se = 0.0
        if math.isfinite(weight) and weight > 0.0:
            finite.append((float(gain), weight, se))
    forced_failures = invalid_events + unused_events
    n = sum(weight for _, weight, _ in finite) + forced_failures
    k_harm = (
        sum(weight * _harm_mass(g, se, threshold) for g, weight, se in finite)
        + forced_failures
    )
    a = 1.0 + (n - k_harm)
    b = 1.0 + k_harm
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
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
) -> CardStatsBlock | None:
    """Global, unadjusted card block from its gain events: median magnitude plus
    the downside posterior, harm being each event's mass below zero — the exact
    sign indicator for an exact event (``gain_se == 0``, the historical strict
    sign test), the Gaussian tail mass when the event prices evaluation noise.
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
    proof_events = [event for event in events if not event.founding]
    if not proof_events:
        return None
    valid = [e for e in proof_events if not e.invalid and not e.unused]
    invalid_events = sum(event_weight(e) for e in proof_events if e.invalid)
    unused_events = sum(
        event_weight(e) for e in proof_events if e.unused and not e.invalid
    )
    valid_gains = [float(e.gain) for e in valid]
    valid_weights = [event_weight(e) for e in valid]
    valid_ses = [float(e.gain_se) for e in valid]
    block = beta_binomial_posterior(
        valid_gains,
        weights=valid_weights,
        event_ses=valid_ses,
        invalid_events=invalid_events,
        unused_events=unused_events,
        confident_quantile=confident_quantile,
        confident_threshold=confident_threshold,
    )
    use_gains = _use_gains(events)
    magnitude: float | None
    if use_gains:
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
        )

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        """The single statistics block every per-card efficacy view resolves
        through: the global, unadjusted block computed from the card's gain
        events (``None`` when the card has none). ``context`` is the additive
        read-seam hook contextual reputations condition on; ignored here."""
        return block_from_events(
            card.gain_events,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
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
        return _ev_rewards(card.gain_events)

    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """Causal weights aligned with :meth:`event_deltas`."""
        return _ev_reward_weights(card.gain_events)

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        """Concrete event rows behind :meth:`event_deltas`."""
        return _ev_events(card.gain_events)

    def event_ses(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]:
        """Per-atom gain ses aligned with :meth:`event_deltas` (0 = exact)."""
        return _ev_ses(card.gain_events)

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        """Contexts the write-side active-policy cleanup can evaluate.

        Global reputation has one value surface, so one contextless evaluation
        is the complete active-policy view.
        """
        del card
        return (None,)

    def staleness_weight(
        self, card: Card, context: DecisionContext | None = None
    ) -> float:
        """Bank-cycle evidence discount ``w = 2**(-s/H)`` used as the bootstrap
        resample weight. The un-decayed base never ages evidence (``1.0``); the
        staleness stack overrides this and the decay reputation reuses the same
        mechanism."""
        return 1.0

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
        local = self._cell_context.local_evidence_events(card, context)
        return list(local) if local else None

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.card_stats(card, context)
        # Same global block math as the base reputation, but over the in-cell
        # subset only: the cell partition already controls for context, so the
        # harm count and median magnitude are measured BD-locally rather than
        # against a parent-fitness counterfactual. Cold cells delegated above.
        return block_from_events(
            in_cell,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )

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
    ) -> tuple[float, ...]:
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

    Staleness reuses the one bank-cycle mechanism ``bank_cycle_weight`` (shared
    with :class:`~gigaevo.memory.read.decay.DecayingReputation`): the discount
    ``w = 2**(-s/H)`` enters as the per-event resample weight, so a stale known
    card's own deltas fade toward neutral zero rather than borrowing unrelated
    winners' positive scale. A card with no non-founding evidence keeps its
    inner block untouched; the write-side birth-failure evictor is responsible
    for deleting catastrophic founding losses before they reach this read path.
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
        weight = self.staleness_weight(card, context)
        # Deterministic per-card seed: the block's pessimistic EV must be
        # reproducible read-to-read and must never consume the live auction
        # round's RNG stream.
        rng = stable_rng(card.id, len(deltas), self._n_bootstrap)
        samples = bootstrap_ev_samples(
            deltas,
            0.0,
            weight,
            self._n_bootstrap,
            rng,
            delta_weights=weights,
            ses=self._inner.event_ses(card, context),
        )
        ev_mean = float(samples.mean())
        ev_lo = float(np.quantile(samples, self._ev_lo_quantile))
        ev_hi = float(np.quantile(samples, 1.0 - self._ev_lo_quantile))
        effective_events = weight * sum(weights or (1.0 for _ in deltas))
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
    ) -> tuple[float, ...]:
        return self._inner.event_ses(card, context)

    def eviction_contexts(self, card: Card) -> tuple[DecisionContext | None, ...]:
        return self._inner.eviction_contexts(card)

    def staleness_weight(
        self, card: Card, context: DecisionContext | None = None
    ) -> float:
        return bank_cycle_weight(
            card,
            self._store.snapshot(),
            self._half_life_cycles,
            reference_events=self.evidence_events(card, context),
        )
