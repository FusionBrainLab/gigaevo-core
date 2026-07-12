"""Effect estimation for injection outcomes — the pluggable crediting seam.

An ``EffectEstimator`` turns one ``InjectionOutcome`` into a ``Measurement``:
the oriented child-vs-base effect plus its standard error. The seam is
domain-general — nothing here knows what an evaluation computes, only that an
outcome carries scalar fitnesses and, optionally, per-sample score vectors.

``PointEffectEstimator`` (default) reproduces the historical behavior exactly:
the oriented delta treated as exact (``se=0``). ``PairedEffectEstimator``
prices per-sample evaluation stochasticity when both sides carry comparable
score vectors, degrading per event to the point estimate when they do not.
Future estimators (K-repeat evals, analytic noise models) plug in behind the
same Protocol.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger
import numpy as np

from gigaevo.memory.cards import Measurement
from gigaevo.memory.context.evidence import oriented_delta
from gigaevo.programs.metrics.paired import (
    COHERENCE_TOL,
    PairedBootstrap,
    PairedComparison,
)

if TYPE_CHECKING:
    from gigaevo.memory.write.stats import InjectionOutcome


@runtime_checkable
class EffectEstimator(Protocol):
    """Strategy seam: one injection outcome -> oriented effect measurement."""

    def estimate(
        self, outcome: InjectionOutcome, *, higher_is_better: bool
    ) -> Measurement:
        """Oriented child-vs-base effect with its standard error.

        Callers guarantee ``outcome.fitness`` and ``outcome.base_fitness`` are
        present; ``se=0`` means the effect is exact.
        """
        ...


# Plain classes, not dataclasses: the ${ref:} write-back structured-wraps
# dataclass instances (OmegaConf), corrupting the config tree at resolve time.
class PointEffectEstimator:
    """Historical exact-delta crediting: the oriented effect with ``se=0``."""

    def estimate(
        self, outcome: InjectionOutcome, *, higher_is_better: bool
    ) -> Measurement:
        delta = oriented_delta(outcome.fitness, outcome.base_fitness, higher_is_better)
        if delta is None:
            raise ValueError("estimate() requires fitness and base_fitness")
        return Measurement(value=delta, se=0.0)


class PairedEffectEstimator:
    """Paired per-sample crediting: point-identical value, paired-comparison se.

    ``value`` stays the oriented scalar delta — the per-vector coherence checks
    below guarantee it matches the paired-vector mean within tolerance — so
    switching estimators changes uncertainty, never the reward itself. ``se``
    comes from ``comparison.estimate`` over the outcome's score vectors; it is
    orientation-invariant, so the vectors are never flipped.

    Any missing/unusable/incoherent vector degrades that one event to ``se=0``.
    Degradations are counted per reason (``degraded``) and logged at debug,
    never raised — a mixed pool of vector-carrying and scalar-only outcomes
    still credits every event.
    """

    def __init__(self, comparison: PairedComparison | None = None) -> None:
        self.comparison = comparison if comparison is not None else PairedBootstrap()
        self.degraded: Counter[str] = Counter()

    def estimate(
        self, outcome: InjectionOutcome, *, higher_is_better: bool
    ) -> Measurement:
        delta = oriented_delta(outcome.fitness, outcome.base_fitness, higher_is_better)
        if delta is None:
            raise ValueError("estimate() requires fitness and base_fitness")
        return Measurement(value=delta, se=self._paired_se(outcome))

    def _paired_se(self, outcome: InjectionOutcome) -> float:
        if outcome.child_scores is None or outcome.base_scores is None:
            return self._degrade(outcome, "missing_vector")
        child = np.asarray(outcome.child_scores, dtype=float)
        base = np.asarray(outcome.base_scores, dtype=float)
        if (
            child.size == 0
            or base.size == 0
            or not np.isfinite(child).all()
            or not np.isfinite(base).all()
        ):
            return self._degrade(outcome, "unusable_vector")
        if child.shape != base.shape:
            return self._degrade(outcome, "length_mismatch")
        if (
            outcome.fitness is None
            or outcome.base_fitness is None
            or abs(float(child.mean()) - float(outcome.fitness)) > COHERENCE_TOL
            or abs(float(base.mean()) - float(outcome.base_fitness)) > COHERENCE_TOL
        ):
            return self._degrade(outcome, "incoherent_vector")
        se = float(self.comparison.estimate(child, base).se)
        if not np.isfinite(se) or se < 0.0:
            return self._degrade(outcome, "degenerate_se")
        return se

    def _degrade(self, outcome: InjectionOutcome, reason: str) -> float:
        self.degraded[reason] += 1
        logger.debug(
            "[Memory][Crediting] paired se degraded to 0 for program {} ({})",
            outcome.id,
            reason,
        )
        return 0.0
