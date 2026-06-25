from __future__ import annotations

from collections.abc import Sequence
import math

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from gigaevo.memory.context import DecisionContext
from gigaevo.memory.efficacy import (
    beta_binomial_posterior,
    block_from_events,
)
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
)


class BetaBinomialReputation(BaseModel):
    """Downside Beta-Binomial reputation over per-card injection gains.

    Configurable façade over ``gigaevo.memory.efficacy.block_from_events`` —
    one implementation, bound here to injectable thresholds. Also the single
    home of the ``is_confidently_harmful`` predicate (pinned by
    tests/memory/test_harm_predicate_single_source.py).
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
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        """The single statistics block every per-card efficacy view resolves
        through: the global, unadjusted block computed from the card's gain
        events (``None`` when the card has none). ``context`` is the additive
        read-seam hook contextual reputations condition on; ignored here."""
        return block_from_events(
            card.gain_events or [],
            noise_band_k=self.noise_band_k,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )

    def card_posterior(
        self, card: AnyCard, context: DecisionContext | None = None
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
        self, card: AnyCard, context: DecisionContext | None = None
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
