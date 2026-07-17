"""Honest OPE for the conditional card-offer gate, not the retrieval policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import math

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


class PreDecisionUnit(BaseModel):
    """Target policies can inspect only variables frozen before treatment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    card: CardSnapshot
    context: EvolutionContext
    behavior_offer_probability: float = Field(gt=0.0, lt=1.0)


class ConditionalOfferOpeReport(BaseModel):
    """DR estimate under the behavior proposal distribution."""

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
        )

    def evaluate_invalidity(
        self,
        observations: Sequence[CausalObservation],
        *,
        target_offer_probability: Callable[[PreDecisionUnit], float],
    ) -> ConditionalOfferOpeReport:
        return self._evaluate(
            observations,
            target_offer_probability=target_offer_probability,
            endpoint="invalid_probability",
            observed=lambda row: float(row.invalid),
            q0=lambda row: row.risk_q_hat_control,
            q1=lambda row: row.risk_q_hat_treated,
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
    ) -> ConditionalOfferOpeReport:
        if not observations:
            raise ValueError("conditional-offer OPE requires terminal observations")
        scores: list[float] = []
        weights: list[float] = []
        by_run: dict[str, list[float]] = defaultdict(list)
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
            weight = target_action / behavior_action
            baseline = target * q1(row) + (1.0 - target) * q0(row)
            score = baseline + weight * (
                observed(row) - (q1(row) if row.treatment else q0(row))
            )
            scores.append(score)
            weights.append(weight)
            by_run[row.context.run_id].append(score)
        estimate = float(np.mean(scores))
        clusters = len(by_run)
        if clusters >= 2:
            centered_cluster_sums = np.asarray(
                [
                    sum(value - estimate for value in values)
                    for values in by_run.values()
                ],
                dtype=float,
            )
            cluster_variance = (
                clusters
                / (clusters - 1.0)
                * float(centered_cluster_sums @ centered_cluster_sums)
                / len(scores) ** 2
            )
            cluster_se = math.sqrt(max(cluster_variance, 0.0))
        else:
            cluster_se = None
        weight_array = np.asarray(weights, dtype=float)
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
