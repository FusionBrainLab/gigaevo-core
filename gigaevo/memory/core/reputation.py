from __future__ import annotations

from collections.abc import Sequence
import math

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from gigaevo.memory.efficacy import EfficacyScorer, beta_binomial_posterior
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_injection_posterior,
)
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
    EvolutionStatistics,
)


class BetaBinomialReputation(BaseModel):
    """Downside Beta-Binomial reputation over per-card injection gains.

    Configurable façade over ``gigaevo.memory.efficacy.EfficacyScorer`` — one
    implementation, bound here to injectable thresholds. Also the single home
    of the ``is_confidently_harmful`` predicate (pinned by
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
    baseline_neighbors: int = Field(
        default=15,
        description="Parent-fitness neighbors forming the local counterfactual cohort.",
    )
    noise_band_k: float = Field(
        default=1.0,
        description="Robust noise-scale multiplier; gains within the band are not harm.",
    )
    noise_floor_rel: float = Field(
        default=1e-4,
        description="Minimum dead-band as a fraction of the cohort's parent-fitness "
        "scale; keeps a zero-MAD plateau from flagging float jitter as harm.",
    )
    cold_prior: tuple[float, float] = Field(
        default=(1.0, 1.0),
        description="(alpha, beta) Beta prior assumed for cards with no stamped posterior.",
    )

    def scorer(self) -> EfficacyScorer:
        """The gain-scoring policy under this reputation's thresholds — the one
        parameterization both the card-side injection posterior and the
        idea-side origin aggregation must use."""
        return EfficacyScorer(
            baseline_neighbors=self.baseline_neighbors,
            noise_band_k=self.noise_band_k,
            noise_floor_rel=self.noise_floor_rel,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
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

    def card_posterior(self, card: AnyCard) -> tuple[float, float]:
        """(alpha, beta) of the card's stamped ``ALL`` downside posterior;
        ``cold_prior`` when the card carries no posterior."""
        block = card.evolution_statistics.ALL
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return self.cold_prior
        a = float(block.posterior_a)
        b = float(block.posterior_b)
        # Beta(a, b) requires finite a > 0, b > 0; a corrupt stamped block
        # would otherwise raise inside the auction's rng.beta draw.
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            return self.cold_prior
        return (a, b)

    def is_confidently_harmful(
        self, evolution_statistics: EvolutionStatistics | None
    ) -> bool:
        """True iff the ALL-block posterior excludes the card as harmful: at least
        ``harm_min_events`` intro events and even the optimistic ``harm_quantile``
        read of P(not harmful) stays below ``harm_threshold``. Missing or thin
        statistics are never harmful."""
        if evolution_statistics is None:
            return False
        block = evolution_statistics.ALL
        if block is None or block.posterior_a is None or block.posterior_b is None:
            return False
        a = float(block.posterior_a)
        b = float(block.posterior_b)
        if block.intro_events < self.harm_min_events or not (
            math.isfinite(a) and math.isfinite(b)
        ):
            return False
        return float(beta.ppf(self.harm_quantile, a, b)) < self.harm_threshold

    def compute_injection_posteriors(
        self,
        programs: Sequence[InjectionOutcome],
        *,
        higher_is_better: bool = True,
    ) -> dict[str, CardStatsBlock]:
        """Per-card downside posteriors over injection outcomes, as typed blocks
        ready to stamp into ``evolution_statistics.ALL``."""
        return compute_injection_posterior(
            programs,
            higher_is_better=higher_is_better,
            scorer=self.scorer(),
        )
