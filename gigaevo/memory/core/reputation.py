from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from pydantic import BaseModel, ConfigDict
from scipy.stats import beta

from gigaevo.memory.shared_memory.injection_posterior import (
    beta_binomial_posterior,
    compute_injection_posterior,
)


class BetaBinomialReputation(BaseModel):
    """Downside Beta-Binomial reputation over per-card injection gains.

    Configurable façade over the numeric primitives in
    ``gigaevo.memory.shared_memory.injection_posterior`` — one implementation,
    bound here to injectable thresholds. Also the single home of the
    ``is_confidently_harmful`` predicate (pinned by
    tests/memory/test_harm_predicate_single_source.py).
    """

    model_config = ConfigDict(frozen=True)

    harm_min_events: int = 3
    harm_quantile: float = 0.80
    harm_threshold: float = 0.5
    confident_quantile: float = 0.20
    confident_threshold: float = 0.5
    baseline_neighbors: int = 15
    noise_band_k: float = 1.0
    cold_prior: tuple[float, float] = (1.0, 1.0)

    def posterior(
        self, gains: Sequence[float], *, threshold: float = 0.0
    ) -> dict[str, Any]:
        return beta_binomial_posterior(
            gains,
            threshold=threshold,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )

    def card_posterior(self, card: Any) -> tuple[float, float]:
        """(a, b) of the card's downside posterior; ``cold_prior`` if absent."""
        if isinstance(card, Mapping):
            stats = card.get("evolution_statistics")
        else:
            stats = getattr(card, "evolution_statistics", None)
        if not isinstance(stats, Mapping):
            return self.cold_prior
        all_block = stats.get("ALL") or {}
        a = all_block.get("posterior_a")
        b = all_block.get("posterior_b")
        if a is None or b is None:
            return self.cold_prior
        return (float(a), float(b))

    def is_confidently_harmful(
        self, evolution_statistics: Mapping[str, Any] | None
    ) -> bool:
        """True iff the ALL-block posterior excludes the card as harmful: at least
        ``harm_min_events`` intro events and even the optimistic ``harm_quantile``
        read of P(not harmful) stays below ``harm_threshold``. Thin, missing, or
        malformed statistics are never harmful."""
        if not isinstance(evolution_statistics, Mapping):
            return False
        block = evolution_statistics.get("ALL")
        if not isinstance(block, Mapping):
            return False
        try:
            n = int(block.get("intro_events") or 0)
            a = float(block["posterior_a"])
            b = float(block["posterior_b"])
        except (KeyError, TypeError, ValueError):
            return False
        if n < self.harm_min_events or not (math.isfinite(a) and math.isfinite(b)):
            return False
        return float(beta.ppf(self.harm_quantile, a, b)) < self.harm_threshold

    def compute_injection_posteriors(
        self,
        programs: Sequence[Mapping[str, Any]],
        *,
        higher_is_better: bool = True,
    ) -> dict[str, dict[str, Any]]:
        return compute_injection_posterior(
            programs,
            higher_is_better=higher_is_better,
            baseline_neighbors=self.baseline_neighbors,
            noise_band_k=self.noise_band_k,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )
