"""Prequential reward-head calibration diagnostics for memory-v2 ledgers.

Read-only. The consumer is the analyst deciding whether the embedding prior
(item 2) and empirical-Bayes prior scales (item 3) improve the Gaussian reward
head. This mirrors ``calibration.py`` (the safety/Bernoulli head) for the
reward/Gaussian head and reuses its ledger substrate and clustering template.

Three diagnostics per typed environment:

* ``predictive_calibration`` (A) refits the reward posterior one step ahead on
  the strictly-earlier valid history within each trajectory and scores the
  realized model-scale gain (PIT, central coverage, NLL, reliability bins). It is
  embedding-free by construction and needs no embeddings.
* ``cold_start_loco`` (B) holds out one card lineage at a time, refits on every
  other card's valid rows with the embedding prior on versus off, and predicts
  the held-out card's treated deliveries. The embedding prior earns its place
  only if it lowers held-out cold-start NLL.
* ``embedding_neighborhood`` (C) slices the leave-one-card-out predictions by
  nearest-neighbour cosine-similarity quantile, exposing regions where embedding
  similarity fails to track the realized effect.

Model selection on the same trajectories is retrospective, so the emitted losses
are development estimates rather than independent validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import math
from typing import Any

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import binomtest, kstest

from gigaevo.memory_v2.calibration import CalibrationTrajectory, SafetyEnvironmentKey
from gigaevo.memory_v2.features import (
    EmbeddingPriorConfig,
    FeatureConfig,
    FeatureSpace,
)
from gigaevo.memory_v2.models import CausalObservation
from gigaevo.memory_v2.posterior import (
    BayesianResidualScaleGaussianRegressor,
    GaussianPosterior,
    TerminalUtilityPosteriorConfig,
    _gain_to_model_scale,
)

REWARD_CALIBRATION_SCHEMA = "gigaevo.memory_v2.reward_calibration/v1"


def _feature_config(
    group: Sequence[CalibrationTrajectory], embedding_prior: EmbeddingPriorConfig | None
) -> FeatureConfig:
    """Derive the one feature schema the environment's decisions all recorded."""

    behavior_keys = {
        tuple(row.key for row in decision.context.map_elites.coordinates)
        for trajectory in group
        for decision in trajectory.decisions
    }
    card_kind = {
        decision.card_kind_contrast
        for trajectory in group
        for decision in trajectory.decisions
    }
    citation = {
        decision.citation_contrast
        for trajectory in group
        for decision in trajectory.decisions
    }
    retrieval = {
        decision.applicability.specification.retrieval_applicability_contrast
        for trajectory in group
        for decision in trajectory.decisions
    }
    if (
        len(behavior_keys) != 1
        or len(card_kind) != 1
        or len(citation) != 1
        or len(retrieval) != 1
    ):
        raise ValueError("environment mixes reward feature schemas")
    return FeatureConfig(
        behavior_keys=next(iter(behavior_keys)),
        card_kind_contrast=next(iter(card_kind)),
        retrieval_applicability_contrast=next(iter(retrieval)),
        citation_contrast=next(iter(citation)),
        embedding_prior=embedding_prior,
    )


def _value_variance(
    observation: CausalObservation, config: TerminalUtilityPosteriorConfig
) -> tuple[float, float]:
    """Model-scale gain and predictive measurement variance for a valid row.

    Mirrors the live reward fit (posterior.py): the gain is normalized by the
    reward scale and mapped to model scale, and the measurement standard
    deviation is the reported standard error over the scale, falling back to the
    configured unknown-measurement standard deviation when none was logged.
    """

    if observation.measurement is None:
        raise ValueError("a valid terminal row is missing its gain measurement")
    scale = observation.context.reward.scale
    standard_error = observation.measurement.se
    measurement_sd = (
        config.unknown_measurement_sd
        if standard_error is None
        else standard_error / scale
    )
    value = _gain_to_model_scale(
        observation.measurement.value / scale, observation.context
    )
    return value, measurement_sd * measurement_sd


def _design(space: FeatureSpace, observation: CausalObservation, *, embedding: bool):
    return space.design(
        observation.card,
        observation.context,
        observation.treatment,
        rag_contrast=observation.rag_applicability.contrast,
        use_contrast=observation.use_contrast,
        embedding=embedding,
    )


def _fit(
    regressor: BayesianResidualScaleGaussianRegressor,
    space: FeatureSpace,
    designs: Sequence[np.ndarray],
    values: Sequence[float],
    variances: Sequence[float],
) -> GaussianPosterior:
    design = (
        np.stack(list(designs))
        if designs
        else np.empty((0, space.outcome_dim), dtype=float)
    )
    return regressor.fit(
        design,
        np.asarray(values, dtype=float),
        np.sqrt(np.asarray(variances, dtype=float)),
        space,
    )


def _predict(
    posterior: GaussianPosterior, design: np.ndarray, measurement_variance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact predictive mixture: per-component mean, variance, and weight.

    The residual-scale posterior is a finite scale mixture, so the one-step
    predictive is a Gaussian mixture rather than a single Gaussian. Each
    component contributes its coefficient uncertainty, its residual variance,
    and the row's measurement variance.
    """

    means = np.array(
        [float(design @ component.mean) for component in posterior.components]
    )
    variances = np.array(
        [
            float(design @ component.covariance @ design)
            + component.residual_sd * component.residual_sd
            + measurement_variance
            for component in posterior.components
        ]
    )
    weights = np.array([component.probability for component in posterior.components])
    return means, variances, weights


def _mixture_mean_sd(
    means: np.ndarray, variances: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """First two moments of the predictive mixture."""

    mean = float(weights @ means)
    variance = float(weights @ (variances + means * means)) - mean * mean
    return mean, math.sqrt(max(variance, 0.0))


def _pit_nll(
    means: np.ndarray, variances: np.ndarray, weights: np.ndarray, value: float
) -> tuple[float, float]:
    """Exact Gaussian-mixture PIT and negative log likelihood at ``value``."""

    standardized = (value - means) / np.sqrt(variances)
    pit = float(weights @ ndtr(standardized))
    log_density = (
        np.log(weights)
        - 0.5 * np.log(2.0 * math.pi * variances)
        - 0.5 * standardized * standardized
    )
    return pit, -float(logsumexp(log_density))


def _coverage(pits: np.ndarray, coverage_levels: Sequence[float]) -> list[dict]:
    """Central predictive-interval coverage from the probability-integral transform.

    A calibrated predictive of any shape has uniform PITs, so its central
    ``level`` interval is exactly ``[(1 - level) / 2, (1 + level) / 2]``. This
    reduces to the Gaussian z-interval when the predictive is a single Gaussian
    and stays exact for the mixture.
    """

    result: list[dict[str, float | int]] = []
    for level in coverage_levels:
        lower = 0.5 - 0.5 * level
        upper = 0.5 + 0.5 * level
        result.append(
            {
                "level": float(level),
                "empirical": float(np.mean((pits >= lower) & (pits <= upper))),
                "observations": int(len(pits)),
            }
        )
    return result


def _clustered_se(values: np.ndarray, clusters: np.ndarray) -> float | None:
    grouped = np.asarray(
        [values[clusters == key].mean() for key in np.unique(clusters)]
    )
    if len(grouped) < 2:
        return None
    return float(grouped.std(ddof=1) / math.sqrt(len(grouped)))


def _reliability_bins(predicted: np.ndarray, actual: np.ndarray) -> list[dict]:
    bins: list[dict[str, float | int]] = []
    order = np.argsort(predicted)
    for indices in np.array_split(order, min(5, len(order))):
        if not len(indices):
            continue
        bins.append(
            {
                "observations": int(len(indices)),
                "mean_predicted": float(predicted[indices].mean()),
                "realized_mean": float(actual[indices].mean()),
            }
        )
    return bins


def _score_predictions(
    pits: np.ndarray,
    nlls: np.ndarray,
    means: np.ndarray,
    values: np.ndarray,
    sds: np.ndarray,
    trajectories: np.ndarray,
    coverage_levels: Sequence[float],
) -> dict[str, Any]:
    uniformity = kstest(pits, "uniform")
    return {
        "observations": int(len(values)),
        "predicted_mean_bias": float(means.mean() - values.mean()),
        "rmse": float(math.sqrt(np.mean((means - values) ** 2))),
        "sharpness": float(sds.mean()),
        "mean_nll": float(nlls.mean()),
        "trajectory_clustered_nll_se": _clustered_se(nlls, trajectories),
        "pit_mean": float(pits.mean()),
        "pit_ks_statistic": float(uniformity.statistic),
        "pit_ks_pvalue": float(uniformity.pvalue),
        "coverage": _coverage(pits, coverage_levels),
        "reliability_bins": _reliability_bins(means, values),
    }


def _arm(
    mask: np.ndarray,
    pits: np.ndarray,
    nlls: np.ndarray,
    means: np.ndarray,
    values: np.ndarray,
    coverage_levels: Sequence[float],
) -> dict[str, Any]:
    if not np.any(mask):
        return {"observations": 0}
    return {
        "observations": int(mask.sum()),
        "predicted_mean_bias": float(means[mask].mean() - values[mask].mean()),
        "mean_nll": float(nlls[mask].mean()),
        "pit_mean": float(pits[mask].mean()),
        "coverage": _coverage(pits[mask], coverage_levels),
    }


def _predictive_calibration(
    group: Sequence[CalibrationTrajectory],
    config: TerminalUtilityPosteriorConfig,
    core_config: FeatureConfig,
    coverage_levels: Sequence[float],
) -> dict[str, Any] | None:
    """Prequential one-step-ahead predictive calibration, embedding-free."""

    regressor = BayesianResidualScaleGaussianRegressor(config)
    pits: list[float] = []
    nlls: list[float] = []
    means: list[float] = []
    values: list[float] = []
    sds: list[float] = []
    treated: list[bool] = []
    trajectories: list[str] = []
    for trajectory in group:
        observations = sorted(
            (row for row in trajectory.observations if not row.invalid),
            key=lambda row: (row.event_ordinal, row.decision_id),
        )
        if len(observations) < 2:
            continue
        space = FeatureSpace(core_config, tuple(row.card for row in observations))
        designs = [_design(space, row, embedding=False) for row in observations]
        targets = [_value_variance(row, config) for row in observations]
        # One-step-ahead history is STRICTLY earlier by event_ordinal, mirroring
        # the sibling head's ``event_ordinal >= decision.event_ordinal`` future
        # guard: observations sharing a scored row's ordinal are contemporaneous,
        # not predecessors, so a row whose only earlier neighbours are tied has
        # no scorable prequential history and is skipped.
        for index in range(1, len(observations)):
            cutoff = observations[index].event_ordinal
            history = [
                position
                for position in range(index)
                if observations[position].event_ordinal < cutoff
            ]
            if not history:
                continue
            posterior = _fit(
                regressor,
                space,
                [designs[position] for position in history],
                [targets[position][0] for position in history],
                [targets[position][1] for position in history],
            )
            value, variance = targets[index]
            components = _predict(posterior, designs[index], variance)
            mean, sd = _mixture_mean_sd(*components)
            pit, nll = _pit_nll(*components, value)
            pits.append(pit)
            nlls.append(nll)
            means.append(mean)
            values.append(value)
            sds.append(sd)
            treated.append(observations[index].treatment)
            trajectories.append(trajectory.trajectory_id)
    if not values:
        return None
    pit_array = np.asarray(pits, dtype=float)
    nll_array = np.asarray(nlls, dtype=float)
    mean_array = np.asarray(means, dtype=float)
    value_array = np.asarray(values, dtype=float)
    sd_array = np.asarray(sds, dtype=float)
    treated_array = np.asarray(treated, dtype=bool)
    trajectory_array = np.asarray(trajectories)
    scored = _score_predictions(
        pit_array,
        nll_array,
        mean_array,
        value_array,
        sd_array,
        trajectory_array,
        coverage_levels,
    )
    scored["treated"] = _arm(
        treated_array,
        pit_array,
        nll_array,
        mean_array,
        value_array,
        coverage_levels,
    )
    scored["control"] = _arm(
        ~treated_array,
        pit_array,
        nll_array,
        mean_array,
        value_array,
        coverage_levels,
    )
    return scored


def _cold_start_loco(
    group: Sequence[CalibrationTrajectory],
    config: TerminalUtilityPosteriorConfig,
    plain_config: FeatureConfig,
    embed_config: FeatureConfig,
    card_embeddings: Mapping[str, np.ndarray],
    coverage_levels: Sequence[float],
) -> tuple[dict[str, Any] | None, dict[str, np.ndarray]]:
    """Leave-one-card-out cold-start, embedding prior on versus off."""

    observations = [
        row
        for trajectory in group
        for row in trajectory.observations
        if not row.invalid
    ]
    trajectory_of = {
        row.decision_id: trajectory.trajectory_id
        for trajectory in group
        for row in trajectory.observations
    }
    cards = tuple(row.card for row in observations)
    embeddings = {
        key: np.asarray(value, dtype=float) for key, value in card_embeddings.items()
    }
    plain_space = FeatureSpace(plain_config, cards)
    embed_space = FeatureSpace(embed_config, cards, embeddings=embeddings)
    regressor = BayesianResidualScaleGaussianRegressor(config)
    bank_of = {
        row.decision_id: plain_space.bank_lineage_id(row.card) for row in observations
    }

    bank_ids = sorted(set(bank_of.values()))
    per_card: list[dict[str, Any]] = []
    resolved_embeddings: dict[str, np.ndarray] = {}
    for held_out in bank_ids:
        held = [
            row
            for row in observations
            if bank_of[row.decision_id] == held_out and row.treatment
        ]
        train = [row for row in observations if bank_of[row.decision_id] != held_out]
        if not held or not train:
            continue
        train_values, train_variances = zip(
            *(_value_variance(row, config) for row in train)
        )
        plain_posterior = _fit(
            regressor,
            plain_space,
            [_design(plain_space, row, embedding=False) for row in train],
            train_values,
            train_variances,
        )
        embed_posterior = _fit(
            regressor,
            embed_space,
            [_design(embed_space, row, embedding=True) for row in train],
            train_values,
            train_variances,
        )
        plain_nll: list[float] = []
        embed_nll: list[float] = []
        embed_pit: list[float] = []
        for row in held:
            value, variance = _value_variance(row, config)
            plain_components = _predict(
                plain_posterior, _design(plain_space, row, embedding=False), variance
            )
            embed_components = _predict(
                embed_posterior, _design(embed_space, row, embedding=True), variance
            )
            _, nll_plain = _pit_nll(*plain_components, value)
            pit, nll_embed = _pit_nll(*embed_components, value)
            plain_nll.append(nll_plain)
            embed_nll.append(nll_embed)
            embed_pit.append(pit)
        nll_core = float(np.mean(plain_nll))
        nll_embed = float(np.mean(embed_nll))
        resolved_embeddings[held_out] = embed_space._embedding[held_out]
        per_card.append(
            {
                "bank_card_id": held_out,
                "observations": len(held),
                "trajectory": trajectory_of.get(
                    held[0].decision_id, held[0].decision_id
                ),
                "nll_core": nll_core,
                "nll_embed": nll_embed,
                "delta_nll": nll_embed - nll_core,
                "pit_mean": float(np.mean(embed_pit)),
                "coverage": _coverage(
                    np.asarray(embed_pit, dtype=float), coverage_levels
                ),
            }
        )
    if not per_card:
        return None, {}
    deltas = np.asarray([row["delta_nll"] for row in per_card], dtype=float)
    clusters = np.asarray([row["trajectory"] for row in per_card])
    # The sign test clusters on the trajectory, matching the clustered standard
    # error: cards sharing a trajectory share training rows, so a card-level test
    # would overstate the number of independent replicates.
    trajectory_deltas = np.asarray(
        [deltas[clusters == key].mean() for key in np.unique(clusters)]
    )
    favorable_trajectories = int(np.sum(trajectory_deltas < 0.0))
    sign_test = binomtest(favorable_trajectories, len(trajectory_deltas), 0.5)
    report = {
        "cards": len(per_card),
        "mean_delta_nll": float(deltas.mean()),
        "mean_nll_core": float(np.mean([row["nll_core"] for row in per_card])),
        "mean_nll_embed": float(np.mean([row["nll_embed"] for row in per_card])),
        "cards_favoring_embedding": int(np.sum(deltas < 0.0)),
        "trajectories": int(len(trajectory_deltas)),
        "trajectories_favoring_embedding": favorable_trajectories,
        "sign_test_unit": "trajectory",
        "sign_test_pvalue": float(sign_test.pvalue),
        "trajectory_clustered_delta_nll_se": _clustered_se(deltas, clusters),
        "embedding_lowers_cold_start_nll": bool(deltas.mean() < 0.0),
        "per_card": per_card,
    }
    return report, resolved_embeddings


def _embedding_neighborhood(
    per_card: Sequence[dict[str, Any]],
    embeddings: Mapping[str, np.ndarray],
    coverage_levels: Sequence[float],
) -> dict[str, Any]:
    """Slice cold-start cards by nearest-neighbour cosine-similarity quantile."""

    bank_ids = [row["bank_card_id"] for row in per_card]
    matrix = np.stack([embeddings[bank_id] for bank_id in bank_ids])
    norms = np.linalg.norm(matrix, axis=1)
    unit = matrix / np.where(norms > 0.0, norms, 1.0)[:, None]
    cosine = unit @ unit.T
    np.fill_diagonal(cosine, -np.inf)
    nearest = cosine.max(axis=1) if len(bank_ids) > 1 else np.zeros(len(bank_ids))
    order = np.argsort(nearest)
    bins: list[dict[str, Any]] = []
    for indices in np.array_split(order, min(3, len(order))):
        if not len(indices):
            continue
        members = [per_card[position] for position in indices]
        deltas = np.asarray([row["delta_nll"] for row in members], dtype=float)
        bins.append(
            {
                "nearest_similarity_lower": float(nearest[indices].min()),
                "nearest_similarity_upper": float(nearest[indices].max()),
                "cards": len(members),
                "mean_delta_nll": float(deltas.mean()),
                "mean_nll_embed": float(np.mean([row["nll_embed"] for row in members])),
            }
        )
    return {"similarity_metric": "nearest_neighbour_cosine_quantile", "bins": bins}


def reward_head_calibration(
    trajectories: Sequence[CalibrationTrajectory],
    *,
    config: TerminalUtilityPosteriorConfig = TerminalUtilityPosteriorConfig(),
    coverage_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
    card_embeddings: Mapping[str, np.ndarray] | None = None,
    embedding_prior: EmbeddingPriorConfig | None = None,
    min_observations: int = 30,
) -> dict[str, Any]:
    """Reward-head calibration and embedding cold-start diagnostics per environment."""

    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    if not coverage_levels or any(not 0.0 < level < 1.0 for level in coverage_levels):
        raise ValueError("coverage levels must lie strictly between zero and one")
    if (card_embeddings is None) != (embedding_prior is None):
        raise ValueError(
            "card_embeddings and embedding_prior must be supplied together"
        )

    grouped: dict[SafetyEnvironmentKey, list[CalibrationTrajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.environment_key, []).append(trajectory)

    reports: list[dict[str, Any]] = []
    for key, group in sorted(
        grouped.items(), key=lambda item: item[0].environment.digest
    ):
        trajectory_ids = [trajectory.trajectory_id for trajectory in group]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError(
                f"environment contains duplicate trajectory snapshots: {key.label}"
            )
        decision_ids = [
            decision.decision_id
            for trajectory in group
            for decision in trajectory.decisions
        ]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError(
                f"environment contains duplicate causal decisions: {key.label}"
            )
        core_config = _feature_config(group, None)
        valid_observations = sum(
            not row.invalid for trajectory in group for row in trajectory.observations
        )
        report: dict[str, Any] = {
            "environment": key.as_dict(),
            "trajectory_count": len(group),
            "valid_observations": valid_observations,
        }
        if valid_observations < min_observations:
            report["status"] = "insufficient_evidence"
            report["predictive_calibration"] = None
            report["cold_start_loco"] = None
            report["embedding_neighborhood"] = None
            reports.append(report)
            continue
        report["predictive_calibration"] = _predictive_calibration(
            group, config, core_config, coverage_levels
        )
        if card_embeddings is None or embedding_prior is None:
            report["cold_start_loco"] = None
            report["embedding_neighborhood"] = None
        else:
            loco, resolved = _cold_start_loco(
                group,
                config,
                core_config,
                _feature_config(group, embedding_prior),
                card_embeddings,
                coverage_levels,
            )
            report["cold_start_loco"] = loco
            report["embedding_neighborhood"] = (
                _embedding_neighborhood(loco["per_card"], resolved, coverage_levels)
                if loco is not None
                else None
            )
        report["status"] = (
            "insufficient_evidence"
            if report["predictive_calibration"] is None
            and report["cold_start_loco"] is None
            else "development_estimate"
        )
        reports.append(report)

    return {
        "schema": REWARD_CALIBRATION_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_metric": (
            "prequential Gaussian-mixture predictive negative log likelihood"
        ),
        "min_observations": min_observations,
        "coverage_levels": [float(level) for level in coverage_levels],
        "embedding_prior": (
            embedding_prior.model_dump(mode="json") if embedding_prior else None
        ),
        "scope_note": (
            "Reward-head calibration is grouped by the complete typed environment "
            "fingerprint. Predictive calibration refits the reward posterior one "
            "step ahead on valid history strictly earlier by event ordinal within "
            "each trajectory; contemporaneous rows sharing a scored row's ordinal "
            "and the first valid row have no prequential history and are not "
            "scored. The refit conditions on realized earlier outcomes, so it "
            "measures the reward head's one-step-ahead predictive calibration, not "
            "the live decision-time posterior, whose delayed reward feedback "
            "withholds not-yet-reconciled outcomes. Prequential probability-integral transforms are serially "
            "dependent, so the reported Kolmogorov-Smirnov p-value is nominal under "
            "an independence assumption the design violates; read the statistic "
            "magnitude rather than its p-value. Cold-start leave-one-card-out refits "
            "on every other card's "
            "valid rows with the embedding prior on versus off and predicts the "
            "held-out card's treated deliveries; the embedding prior is credited "
            "only when it lowers held-out cold-start negative log likelihood. "
            "Neighbourhood bins are nearest-neighbour cosine-similarity quantiles. "
            "Selecting on these same trajectories is retrospective, so the reported "
            "losses are development estimates rather than independent validation."
        ),
        "sources": [str(trajectory.ledger_path) for trajectory in trajectories],
        "groups": reports,
    }
