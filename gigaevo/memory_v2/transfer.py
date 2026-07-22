"""Small scale-free cross-task usefulness model for shared memory cards."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import ndtr

from gigaevo.memory.cards import Card, CardUseTrial
from gigaevo.memory_v2.models import canonical_digest
from gigaevo.memory_v2.posterior import (
    PosteriorFitError,
    StableBayesianLogisticRegressor,
    TerminalUtilityPosteriorConfig,
)


class CrossTaskUsefulnessConfig(BaseModel):
    """Fixed priors for the task/card binary-success transfer head."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    baseline_prior_sd: float = Field(default=1.5, gt=0.0)
    global_card_prior_sd: float = Field(default=0.75, gt=0.0)
    task_card_prior_sd: float = Field(default=1.0, gt=0.0)
    minimum_trials_per_arm: int = Field(default=2, ge=1)
    max_reward_prior_shift_sd: float = Field(default=0.5, gt=0.0, le=1.0)


@dataclass(frozen=True)
class CrossTaskUsefulnessFit:
    helpful_probability: dict[str, float]
    observations: int

    def reward_intercept_means(
        self,
        *,
        card_effect_prior_sd: float,
        max_shift_sd: float,
    ) -> dict[str, float]:
        return {
            card_id: (max_shift_sd * card_effect_prior_sd * (2.0 * probability - 1.0))
            for card_id, probability in self.helpful_probability.items()
        }


class CrossTaskUsefulnessModel:
    """Hierarchical logistic model over randomized binary card outcomes.

    Each run/card stratum receives its own baseline, each task/card pair receives
    a treatment deviation, and the card treatment main effect is shared across
    tasks. Consequently differing run baselines and offer mixes cannot create a
    pooled treatment effect. No continuous reward magnitude or behavior
    coordinate crosses task boundaries.
    """

    def __init__(
        self,
        config: CrossTaskUsefulnessConfig,
        posterior_config: TerminalUtilityPosteriorConfig,
    ) -> None:
        self.config = config
        self._regressor = StableBayesianLogisticRegressor(posterior_config)

    def evidence_digest(self, cards: tuple[Card, ...], *, current_run_id: str) -> str:
        return canonical_digest(
            {
                "config": self.config.model_dump(mode="json"),
                "trials": {
                    card.id: [
                        trial.model_dump(mode="json")
                        for trial in card.use_trials
                        if trial.run_id != current_run_id
                    ]
                    for card in sorted(cards, key=lambda row: row.id)
                },
            }
        )

    def fit(
        self,
        cards: tuple[Card, ...],
        *,
        target_task_key: str,
        current_run_id: str,
    ) -> CrossTaskUsefulnessFit:
        card_ids = tuple(sorted({card.id for card in cards}))
        neutral = CrossTaskUsefulnessFit(
            helpful_probability={card_id: 0.5 for card_id in card_ids},
            observations=0,
        )
        if not card_ids:
            return neutral

        grouped: dict[tuple[str, str, str], list[CardUseTrial]] = defaultdict(list)
        for card in cards:
            for trial in card.use_trials:
                if trial.run_id != current_run_id:
                    grouped[(trial.task_key, trial.run_id, card.id)].append(trial)

        minimum = self.config.minimum_trials_per_arm
        supported = {
            stratum: tuple(trials)
            for stratum, trials in grouped.items()
            if sum(trial.treatment for trial in trials) >= minimum
            and sum(not trial.treatment for trial in trials) >= minimum
        }
        if not supported:
            return neutral

        strata = tuple(sorted(supported))
        stratum_index = {stratum: index for index, stratum in enumerate(strata)}
        task_pairs = tuple(sorted({(task, card) for task, _, card in strata}))
        task_pair_index = {pair: index for index, pair in enumerate(task_pairs)}
        card_index = {card_id: index for index, card_id in enumerate(card_ids)}
        stratum_count = len(strata)
        task_pair_count = len(task_pairs)
        card_count = len(card_ids)
        dimension = stratum_count + card_count + task_pair_count
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for stratum in strata:
            baseline_offset = stratum_index[stratum]
            task_pair_offset = task_pair_index[(stratum[0], stratum[2])]
            card_offset = card_index[stratum[2]]
            for trial in supported[stratum]:
                row = np.zeros(dimension, dtype=float)
                row[baseline_offset] = 1.0
                if trial.treatment:
                    row[stratum_count + card_offset] = 1.0
                    row[stratum_count + card_count + task_pair_offset] = 1.0
                rows.append(row)
                targets.append(float(trial.success))

        design = np.vstack(rows)
        prior_mean = np.zeros(dimension, dtype=float)
        prior_variance = np.empty(dimension, dtype=float)
        prior_variance[:stratum_count] = self.config.baseline_prior_sd**2
        prior_variance[stratum_count : stratum_count + card_count] = (
            self.config.global_card_prior_sd**2
        )
        prior_variance[stratum_count + card_count :] = self.config.task_card_prior_sd**2
        try:
            fitted = self._regressor.fit(
                design,
                np.asarray(targets, dtype=float),
                prior_mean,
                prior_variance,
            )
        except PosteriorFitError as exc:
            logger.warning(
                "[MemoryV2][Transfer] usefulness fit failed; using neutral priors: {}",
                exc,
            )
            return neutral

        probabilities: dict[str, float] = {}
        for card_id in card_ids:
            global_index = stratum_count + card_index[card_id]
            mean = float(fitted.mean[global_index])
            variance = float(fitted.covariance[global_index, global_index])
            target_pair = (target_task_key, card_id)
            if target_pair in task_pair_index:
                deviation_index = (
                    stratum_count + card_count + task_pair_index[target_pair]
                )
                mean += float(fitted.mean[deviation_index])
                variance += float(fitted.covariance[deviation_index, deviation_index])
                variance += 2.0 * float(
                    fitted.covariance[global_index, deviation_index]
                )
            else:
                variance += self.config.task_card_prior_sd**2
            probabilities[card_id] = float(ndtr(mean / np.sqrt(max(variance, 1e-12))))

        return CrossTaskUsefulnessFit(
            helpful_probability=probabilities,
            observations=len(targets),
        )
