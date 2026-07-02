"""Per-card efficacy statistics derived from injection outcomes.

Owner of the gain → downside-posterior math: ``block_from_events`` turns a
card's gain events into the reputation block (median magnitude, MAD harm band)
that the auction and renderer read. ``BetaBinomialReputation`` binds that math
to injectable thresholds; ``BDProximityReputation`` partitions it by the query
parent's MAP-Elites cell.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import statistics

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.memory.cards import Card, CardStatsBlock, ContextualGain, DecisionContext

_MAD_TO_SIGMA = 1.4826


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def robust_noise_band(values: Sequence[float]) -> float:
    """Robust noise scale of the values (MAD -> sigma), centred on their median.
    Collapses to 0 for a degenerate/flat set so genuine discrete steps still
    register. The single source of the harm-predicate noise band for both the
    global counterfactual path and the BD-cell partition."""
    if not values:
        return 0.0
    med = _median(values)
    return _MAD_TO_SIGMA * _median([abs(x - med) for x in values])


def beta_binomial_posterior(
    gains: Sequence[float],
    *,
    threshold: float = 0.0,
    invalid_events: int = 0,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
) -> CardStatsBlock:
    """Downside Beta-Binomial posterior on P(not harmful) from per-event gains.

    ``a = 1 + (n - k_harm)``, ``b = 1 + k_harm`` with ``k_harm`` the count of
    events whose gain is below ``threshold`` (default 0); ``efficacy_confident``
    iff the ``confident_quantile`` of Beta(a, b) exceeds ``confident_threshold``.
    The ``p_help_lo20`` field name is part of the serialized-card stats contract
    regardless of the configured quantile. ``invalid_events`` are evaluated-and-
    judged-invalid children: each is one forced harm event with no gain magnitude.
    """
    finite = [float(g) for g in gains if g is not None and math.isfinite(float(g))]
    n = len(finite) + invalid_events
    k_harm = sum(1 for g in finite if g < threshold) + invalid_events
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
    noise_band_k: float = 1.0,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
) -> CardStatsBlock | None:
    """Global, unadjusted card block from its gain events: median magnitude plus
    the downside posterior, harm being a gain below the robust noise band
    ``-noise_band_k * MAD`` of the finite valid gains. Invalid events are forced
    harm with no magnitude. Returns ``None`` for a card with no events (no
    evidence, no block). The single owner of the events -> card block math,
    shared by the global reputation and the BD in-cell partition.
    """
    if not events:
        return None
    valid = [e for e in events if not e.invalid]
    invalid_events = len(events) - len(valid)
    valid_gains = [float(e.gain) for e in valid]
    finite_gains = [g for g in valid_gains if math.isfinite(g)]
    epsilon = noise_band_k * robust_noise_band(finite_gains)
    block = beta_binomial_posterior(
        valid_gains,
        threshold=-epsilon,
        invalid_events=invalid_events,
        confident_quantile=confident_quantile,
        confident_threshold=confident_threshold,
    )
    magnitude = _median(finite_gains) if finite_gains else 0.0
    return block.model_copy(update={"IntroGain_best_median": magnitude})


class BetaBinomialReputation(BaseModel):
    """Downside Beta-Binomial reputation over per-card injection gains.

    Configurable façade over :func:`block_from_events` — one implementation,
    bound here to injectable thresholds. Also the single home of the
    ``is_confidently_harmful`` predicate.
    """

    model_config = ConfigDict(frozen=True)

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
    noise_band_k: float = Field(
        default=1.0,
        description="Robust noise-scale multiplier; gains within the band are not harm.",
    )
    cold_prior: tuple[float, float] = Field(
        default=(1.0, 1.0),
        description="(alpha, beta) Beta prior assumed for cards with no stamped posterior.",
    )

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
            noise_band_k=self.noise_band_k,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )

    def card_posterior(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, float]:
        """(alpha, beta) of the card's resolved downside posterior; ``cold_prior``
        when the card carries no posterior. Resolved through ``card_stats``, so
        contextual reputations condition it on ``context``."""
        block = self.card_stats(card, context)
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return self.cold_prior
        a = float(block.posterior_a)
        b = float(block.posterior_b)
        # Beta(a, b) requires finite a > 0, b > 0; a corrupt stamped block
        # would otherwise raise inside the auction's rng.beta draw.
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            return self.cold_prior
        return (a, b)

    def card_magnitude(
        self, card: Card, context: DecisionContext | None = None
    ) -> float | None:
        """The card's resolved expected gain (``IntroGain_best_median``) — the EV
        auction's magnitude. ``None`` when the card has no events (cold), so the
        auction falls back to its optimistic prior. Resolved through
        ``card_stats``, so contextual reputations condition it on ``context``."""
        block = self.card_stats(card, context)
        if block is None or block.IntroGain_best_median is None:
            return None
        return float(block.IntroGain_best_median)

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
    under the held ``behavior_space``'s current bounds — the bandit reads the
    one tessellation, never stores a cell id (``DynamicBehaviorSpace`` moves
    cells on every reindex).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    behavior_space: BehaviorSpace = Field(
        description="The run's tessellation; bucketing reads its CURRENT bounds.",
    )
    fallback: BetaBinomialReputation = Field(
        default_factory=BetaBinomialReputation,
        description="Cold-cell delegate: the global event-derived reputation.",
    )

    def _cell(self, metrics: dict[str, float]) -> tuple[int, ...] | None:
        # A missing or non-finite behavior coord has no well-defined cell —
        # LinearBinning silently clamps NaN to bin 0, so guard here and abstain
        # to fallback rather than credit events to a spurious low-end cell.
        for key in self.behavior_space.behavior_keys:
            value = metrics.get(key)
            if value is None or not math.isfinite(value):
                return None
        return self.behavior_space.get_cell(metrics)

    def _in_cell(
        self, card: Card, context: DecisionContext | None
    ) -> list[ContextualGain] | None:
        if context is None:
            return None
        events = card.gain_events
        if not events:
            return None
        parent_cell = self._cell(context.parent_metrics)
        if parent_cell is None:
            return None
        in_cell = [
            event
            for event in events
            if self._cell(event.context.parent_metrics) == parent_cell
        ]
        return in_cell or None

    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.card_stats(card, context)
        # Same global block math as the base reputation, but over the in-cell
        # subset only: the cell partition already controls for context, so the
        # MAD harm band and median magnitude are measured BD-locally rather than
        # against a parent-fitness counterfactual. Cold cells delegated above.
        return block_from_events(
            in_cell,
            noise_band_k=self.noise_band_k,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )
