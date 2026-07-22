"""Honest OPE for the conditional card-offer gate, not the retrieval policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory_v2.models import CardSnapshot, CausalObservation, EvolutionContext


def _worst_gain(row: CausalObservation) -> float:
    reward = row.context.reward
    parent = row.context.parent_metrics[reward.primary_metric]
    return (
        reward.metric_lower_bound - parent
        if reward.higher_is_better
        else parent - reward.metric_upper_bound
    )


def _optimistic_shrinkage(
    weight: float | np.ndarray, shrinkage: float
) -> float | np.ndarray:
    """DRos optimistic-shrinkage weight (Su & Wang 2020): ``lam*w / (lam + w^2)``.

    Hump-shaped in ``weight`` (peaks at ``w = sqrt(lam)`` then drives large
    weights back toward zero), always bounded above by the raw weight, and the
    identity as ``shrinkage -> inf`` (no shrinkage). Accepts a scalar or array.
    """
    if shrinkage == math.inf:
        return weight
    return shrinkage * weight / (shrinkage + weight**2)


def _run_clustered_variance(scores: np.ndarray, run_ids: Sequence[str]) -> float | None:
    """Run-clustered variance of the mean, or None with fewer than two runs."""

    estimate = float(np.mean(scores))
    centered_by_run: dict[str, float] = defaultdict(float)
    for run_id, score in zip(run_ids, scores.tolist()):
        centered_by_run[run_id] += score - estimate
    clusters = len(centered_by_run)
    if clusters < 2:
        return None
    centered = np.asarray(list(centered_by_run.values()), dtype=float)
    return clusters / (clusters - 1.0) * float(centered @ centered) / len(scores) ** 2


def _shrinkage_mse(
    shrinkage: float,
    raw_weights: np.ndarray,
    baselines: np.ndarray,
    corrections: np.ndarray,
    run_ids: Sequence[str],
    reference: float,
) -> float:
    scores = baselines + _optimistic_shrinkage(raw_weights, shrinkage) * corrections
    bias = float(np.mean(scores)) - reference
    clustered = _run_clustered_variance(scores, run_ids)
    if clustered is not None:
        variance = clustered
    elif raw_weights.size > 1:
        variance = float(np.var(scores, ddof=1)) / raw_weights.size
    else:
        variance = 0.0
    return bias * bias + variance


def _select_shrinkage(
    raw_weights: np.ndarray,
    baselines: np.ndarray,
    corrections: np.ndarray,
    run_ids: Sequence[str],
) -> float:
    """Pick the DRos lambda minimizing the estimated MSE of the point estimate.

    The unshrunk DR estimate is the low-bias reference; the variance term is
    run-clustered to match the reported cluster-robust SE. Candidate lambdas are
    the observed squared weights (the data-driven scale at which shrinkage
    engages); no-shrinkage (``lambda -> inf``) seeds the search, so auto never
    shrinks unless a finite lambda strictly lowers the estimated MSE. The
    continuous MSE optimum can fall between observed squared weights -- the grid
    is deliberately data-driven rather than a fixed geometric ladder.
    """
    reference = float(np.mean(baselines + raw_weights * corrections))
    best_lambda = math.inf
    best_mse = _shrinkage_mse(
        math.inf, raw_weights, baselines, corrections, run_ids, reference
    )
    for candidate in np.unique(raw_weights[raw_weights > 0.0] ** 2):
        mse = _shrinkage_mse(
            float(candidate), raw_weights, baselines, corrections, run_ids, reference
        )
        if mse < best_mse:
            best_mse = mse
            best_lambda = float(candidate)
    return best_lambda


class PreDecisionUnit(BaseModel):
    """Target policies can inspect only variables frozen before treatment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    card: CardSnapshot
    context: EvolutionContext
    behavior_offer_probability: float = Field(gt=0.0, lt=1.0)


class ConditionalOfferOpeReport(BaseModel):
    """DR estimate under the behavior proposal distribution.

    ``effective_sample_size`` and ``maximum_importance_weight`` are computed on
    the shrinkage-adjusted weights; under ``shrinkage="auto"`` the cluster-robust
    SE is conditional on the selected lambda and omits post-selection variability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    endpoint: str
    estimand: str = (
        "one-decision target offer under the logged behavior proposal distribution, "
        "followed by the natural downstream evolution policy"
    )
    estimate: float
    cluster_robust_se: float | None = Field(default=None, ge=0.0)
    clusters: int = Field(ge=1)
    observations: int = Field(ge=1)
    effective_sample_size: float = Field(gt=0.0)
    maximum_importance_weight: float = Field(ge=0.0)
    minimum_behavior_probability: float = Field(gt=0.0, lt=1.0)


class ConditionalOfferDREvaluator:
    """Prequential DR point estimate with run-cluster uncertainty when available."""

    def evaluate_reward(
        self,
        observations: Sequence[CausalObservation],
        *,
        target_offer_probability: Callable[[PreDecisionUnit], float],
        shrinkage: float | Literal["auto"] | None = None,
    ) -> ConditionalOfferOpeReport:
        endpoints = {row.context.reward.endpoint for row in observations}
        if len(endpoints) > 1:
            raise ValueError("reward OPE mixes endpoint definitions")
        return self._evaluate(
            observations,
            target_offer_probability=target_offer_probability,
            endpoint=next(iter(endpoints), "bounded_proximal_utility"),
            observed=lambda row: (
                _worst_gain(row) if row.invalid else row.measurement.value  # type: ignore[union-attr]
            ),
            q0=lambda row: row.reward_q_hat_control,
            q1=lambda row: row.reward_q_hat_treated,
            shrinkage=shrinkage,
        )

    def evaluate_invalidity(
        self,
        observations: Sequence[CausalObservation],
        *,
        target_offer_probability: Callable[[PreDecisionUnit], float],
        shrinkage: float | Literal["auto"] | None = None,
    ) -> ConditionalOfferOpeReport:
        return self._evaluate(
            observations,
            target_offer_probability=target_offer_probability,
            endpoint="invalid_probability",
            observed=lambda row: float(row.invalid),
            q0=lambda row: row.risk_q_hat_control,
            q1=lambda row: row.risk_q_hat_treated,
            shrinkage=shrinkage,
        )

    def _evaluate(
        self,
        observations: Sequence[CausalObservation],
        *,
        target_offer_probability: Callable[[PreDecisionUnit], float],
        endpoint: str,
        observed: Callable[[CausalObservation], float],
        q0: Callable[[CausalObservation], float],
        q1: Callable[[CausalObservation], float],
        shrinkage: float | Literal["auto"] | None = None,
    ) -> ConditionalOfferOpeReport:
        if not observations:
            raise ValueError("conditional-offer OPE requires terminal observations")
        raw_weights: list[float] = []
        baselines: list[float] = []
        corrections: list[float] = []
        run_ids: list[str] = []
        minimum_behavior = 1.0
        for row in observations:
            if not row.invalid and row.measurement is None:
                raise ValueError("valid OPE row is missing its reward measurement")
            unit = PreDecisionUnit(
                decision_id=row.decision_id,
                card=row.card,
                context=row.context,
                behavior_offer_probability=row.offer_propensity,
            )
            target = float(target_offer_probability(unit))
            if not 0.0 <= target <= 1.0:
                raise ValueError("target offer probability must lie in [0, 1]")
            behavior = row.offer_propensity
            minimum_behavior = min(minimum_behavior, behavior, 1.0 - behavior)
            target_action = target if row.treatment else 1.0 - target
            behavior_action = behavior if row.treatment else 1.0 - behavior
            raw_weights.append(target_action / behavior_action)
            baselines.append(target * q1(row) + (1.0 - target) * q0(row))
            corrections.append(observed(row) - (q1(row) if row.treatment else q0(row)))
            run_ids.append(row.context.run_id)
        raw = np.asarray(raw_weights, dtype=float)
        baseline_array = np.asarray(baselines, dtype=float)
        correction_array = np.asarray(corrections, dtype=float)
        if shrinkage is None:
            shrinkage_lambda = math.inf
        elif shrinkage == "auto":
            shrinkage_lambda = _select_shrinkage(
                raw, baseline_array, correction_array, run_ids
            )
        elif isinstance(shrinkage, str):
            raise ValueError(f"unknown shrinkage mode {shrinkage!r}")
        else:
            shrinkage_lambda = float(shrinkage)
            if math.isnan(shrinkage_lambda) or shrinkage_lambda <= 0.0:
                raise ValueError("shrinkage lambda must be positive")
        weight_array = np.asarray(
            _optimistic_shrinkage(raw, shrinkage_lambda), dtype=float
        )
        scores = baseline_array + weight_array * correction_array
        estimate = float(np.mean(scores))
        cluster_variance = _run_clustered_variance(scores, run_ids)
        cluster_se = (
            math.sqrt(max(cluster_variance, 0.0))
            if cluster_variance is not None
            else None
        )
        clusters = len(set(run_ids))
        squared_weight_sum = float(np.sum(weight_array**2))
        if squared_weight_sum == 0.0:
            raise ValueError("the realized log contains no target-policy action mass")
        ess = float(weight_array.sum() ** 2 / squared_weight_sum)
        return ConditionalOfferOpeReport(
            endpoint=endpoint,
            estimate=estimate,
            cluster_robust_se=cluster_se,
            clusters=clusters,
            observations=len(scores),
            effective_sample_size=ess,
            maximum_importance_weight=float(weight_array.max()),
            minimum_behavior_probability=minimum_behavior,
        )
