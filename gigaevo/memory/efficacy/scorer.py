"""Single owner of the gain -> downside-posterior math.

Both efficacy paths — the card-side injection posterior
(``gigaevo.memory.shared_memory.injection_posterior``) and the idea-side origin
aggregation (``gigaevo.memory.ideas_tracker.utils.origin_analysis``) — judge a
gain against the same counterfactual: the median improvement of children whose
best parent had similar fitness, with harm only counted beyond a robust noise
band of that centred population. ``EfficacyScorer`` is the one place that policy
lives; it is constructed from ``BetaBinomialReputation`` so the Hydra
``memory/reputation`` knobs reach both paths identically.

The cohort is deduplicated by child id when fitted: a child that introduces
several ideas (or carries several cards) still weighs exactly once in the
baseline and noise band.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import statistics

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from gigaevo.memory.shared_memory.models import CardStatsBlock

_MAD_TO_SIGMA = 1.4826


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


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
    The ``p_help_lo20`` field name is part of the banks.json contract regardless
    of the configured quantile. ``invalid_events`` are evaluated-and-judged-
    invalid children: each is one forced harm event with no gain magnitude.
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


class GainObservation(BaseModel):
    """One child's parent-relative improvement, in direction-normalized units
    (positive = improvement regardless of the metric's optimization direction)."""

    model_config = ConfigDict(frozen=True)

    child_id: str = Field(description="Program id of the child; cohort dedup key.")
    parent_fitness: float = Field(
        description="Best-parent fitness, the reference the gain was measured against."
    )
    gain: float = Field(
        description="Child fitness minus best-parent fitness, direction-normalized."
    )
    invalid: bool = Field(
        default=False,
        description="Evaluated-and-judged-invalid child: forced harm event, "
        "excluded from the baseline cohort, gain magnitude meaningless.",
    )


class EfficacyScorer(BaseModel):
    """Parent-local counterfactual baseline, MAD noise band, downside
    Beta-Binomial posterior — the only producer of
    ``posterior_a``/``posterior_b``/``k_harm``/``efficacy_confident``.
    ``fit`` a cohort of gain observations once, then score any subset of events
    against it.
    """

    model_config = ConfigDict(frozen=True)

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
    confident_quantile: float = Field(
        default=0.20,
        description="Pessimistic posterior quantile used for the confidence flag.",
    )
    confident_threshold: float = Field(
        default=0.5,
        description="Confident iff the pessimistic P(help) read clears this.",
    )

    def fit(self, observations: Sequence[GainObservation]) -> FittedEfficacyScorer:
        """Fit the counterfactual baseline and noise band on the cohort,
        deduplicated by child id (first observation per child wins)."""
        return FittedEfficacyScorer(self, observations)


class FittedEfficacyScorer:
    """An ``EfficacyScorer`` bound to a fitted cohort: holds the parent-local
    baseline and the noise-band half-width ``epsilon``."""

    def __init__(
        self, scorer: EfficacyScorer, observations: Sequence[GainObservation]
    ) -> None:
        self._scorer = scorer
        seen: set[str] = set()
        cohort: list[GainObservation] = []
        for o in observations:
            if o.invalid or o.child_id in seen:
                continue
            seen.add(o.child_id)
            cohort.append(o)
        self._ids = [o.child_id for o in cohort]
        self._refs = [o.parent_fitness for o in cohort]
        self._gains = [o.gain for o in cohort]
        centered = [self.adjusted_gain(o) for o in cohort]
        scale = max((abs(r) for r in self._refs), default=0.0)
        self.epsilon = max(
            scorer.noise_band_k * self._noise_band(centered),
            scorer.noise_floor_rel * scale,
        )

    @staticmethod
    def _noise_band(centered: Sequence[float]) -> float:
        """Robust noise scale of the centred improvements (MAD -> sigma).
        Collapses to 0 for a degenerate/flat cohort so genuine discrete steps
        still register."""
        if not centered:
            return 0.0
        med = _median(centered)
        return _MAD_TO_SIGMA * _median([abs(x - med) for x in centered])

    def baseline(
        self, parent_fitness: float, *, exclude_child_id: str | None = None
    ) -> float:
        """Counterfactual improvement for a parent of this fitness: the median
        gain of the ``baseline_neighbors`` cohort children with the nearest
        best-parent fitness, leave-one-out for the scored child itself (a lone
        regressing child must not be its own counterfactual). With a small
        cohort every point is used (a global median); an empty pool yields 0."""
        pool = [i for i in range(len(self._gains)) if self._ids[i] != exclude_child_id]
        if not pool:
            return 0.0
        k = min(self._scorer.baseline_neighbors, len(pool))
        nearest = sorted(pool, key=lambda i: abs(self._refs[i] - parent_fitness))[:k]
        return _median([self._gains[i] for i in nearest])

    def adjusted_gain(self, event: GainObservation) -> float:
        return event.gain - self.baseline(
            event.parent_fitness, exclude_child_id=event.child_id
        )

    def adjusted_gains(self, events: Sequence[GainObservation]) -> list[float]:
        return [self.adjusted_gain(e) for e in events]

    def posterior(self, events: Sequence[GainObservation]) -> CardStatsBlock:
        """Downside posterior over the events' baseline-adjusted gains; harm is
        an adjusted gain below ``-epsilon``. Invalid events skip the gain math
        entirely and enter as forced harm."""
        scored = [e for e in events if not e.invalid]
        return beta_binomial_posterior(
            self.adjusted_gains(scored),
            threshold=-self.epsilon,
            invalid_events=len(events) - len(scored),
            confident_quantile=self._scorer.confident_quantile,
            confident_threshold=self._scorer.confident_threshold,
        )
