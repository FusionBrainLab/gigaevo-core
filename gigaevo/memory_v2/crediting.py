"""Uncertainty-honest effect estimators for the legacy writer adapter."""

from __future__ import annotations

from collections import Counter
import math
from typing import TYPE_CHECKING

from loguru import logger
import numpy as np

from gigaevo.memory.cards import Measurement
from gigaevo.memory.context.evidence import oriented_delta
from gigaevo.programs.metrics.paired import COHERENCE_TOL

if TYPE_CHECKING:
    from gigaevo.memory.write.stats import InjectionOutcome


class HonestPointEffectEstimator:
    """Scalar delta whose unmeasured uncertainty is unknown, never zero."""

    def estimate(
        self, outcome: InjectionOutcome, *, higher_is_better: bool
    ) -> Measurement:
        delta = oriented_delta(outcome.fitness, outcome.base_fitness, higher_is_better)
        if delta is None:
            raise ValueError("estimate() requires fitness and base_fitness")
        return Measurement(value=delta, se=None)


class AnalyticPairedEffectEstimator:
    """Paired mean effect with the analytic standard error of paired deltas."""

    def __init__(self) -> None:
        self.degraded: Counter[str] = Counter()

    def estimate(
        self, outcome: InjectionOutcome, *, higher_is_better: bool
    ) -> Measurement:
        delta = oriented_delta(outcome.fitness, outcome.base_fitness, higher_is_better)
        if delta is None:
            raise ValueError("estimate() requires fitness and base_fitness")
        return Measurement(
            value=delta,
            se=self._paired_se(outcome, higher_is_better=higher_is_better),
        )

    def _paired_se(
        self, outcome: InjectionOutcome, *, higher_is_better: bool
    ) -> float | None:
        if (
            not outcome.child_score_signature
            or outcome.child_score_signature != outcome.base_score_signature
        ):
            return self._degrade(outcome, "cohort_mismatch")
        if outcome.child_scores is None or outcome.base_scores is None:
            return self._degrade(outcome, "missing_vector")
        child = np.asarray(outcome.child_scores, dtype=float)
        base = np.asarray(outcome.base_scores, dtype=float)
        if child.size == 0 or base.size == 0 or child.ndim != 1 or base.ndim != 1:
            return self._degrade(outcome, "unusable_vector")
        if not np.isfinite(child).all() or not np.isfinite(base).all():
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
        if len(child) < 2:
            return self._degrade(outcome, "insufficient_pairs")
        differences = child - base if higher_is_better else base - child
        se = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
        if not math.isfinite(se) or se < 0.0:
            return self._degrade(outcome, "degenerate_se")
        return se

    def _degrade(self, outcome: InjectionOutcome, reason: str) -> float | None:
        self.degraded[reason] += 1
        logger.debug(
            "[MemoryV2][Crediting] paired se unknown for program {} ({})",
            outcome.id,
            reason,
        )
        return None
