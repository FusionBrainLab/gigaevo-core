"""Coherent hierarchical terminal-utility posterior for memory-v2 decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logit, logsumexp, ndtr
from scipy.stats import norm

from gigaevo.memory_v2.features import FeatureSpace, HierarchicalFeatureMap
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    EvolutionContext,
    PosteriorPrediction,
    RewardDefinition,
    canonical_digest,
    validate_reward_observations,
)

_SAFETY_INTEGRATION_LIMIT = 10.0
_MIN_SAFETY_CONDITIONAL_SD_RATIO = 1e-5
_MAX_SAFETY_BREAKPOINTS = 32


class PosteriorFitError(RuntimeError):
    """Inference failed a numerical integrity check; the policy must abstain."""


def _normalized_gain_bounds(context: EvolutionContext) -> tuple[float, float]:
    reward = context.reward
    try:
        parent = context.parent_metrics[reward.primary_metric]
    except KeyError as exc:
        raise ValueError(
            f"parent metrics omit primary metric {reward.primary_metric!r}"
        ) from exc
    if reward.higher_is_better:
        lower = (reward.metric_lower_bound - parent) / reward.scale
        upper = (reward.metric_upper_bound - parent) / reward.scale
    else:
        lower = (parent - reward.metric_upper_bound) / reward.scale
        upper = (parent - reward.metric_lower_bound) / reward.scale
    if not -1.0 - 1e-9 <= lower < upper <= 1.0 + 1e-9:
        raise ValueError("parent fitness is outside the configured reward bounds")
    return max(lower, -1.0), min(upper, 1.0)


def _gain_to_model_scale(value: float, context: EvolutionContext) -> float:
    lower, upper = _normalized_gain_bounds(context)
    if value < lower - 1e-9 or value > upper + 1e-9:
        raise ValueError(
            f"normalized gain {value:.6g} is outside [{lower:.6g}, {upper:.6g}]"
        )
    return min(max(value, lower), upper)


def _latent_to_gain(
    centered_latent: np.ndarray | float,
    context: EvolutionContext,
) -> np.ndarray:
    lower, upper = _normalized_gain_bounds(context)
    return np.clip(np.asarray(centered_latent, dtype=float), lower, upper)


def _bisect_sign_change(
    function: Callable[[float], float], left: float, right: float
) -> float:
    left_value = function(left)
    for _ in range(80):
        middle = 0.5 * (left + right)
        middle_value = function(middle)
        if middle_value == 0.0:
            return middle
        if (left_value <= 0.0) == (middle_value <= 0.0):
            left = middle
            left_value = middle_value
        else:
            right = middle
    return 0.5 * (left + right)


def _joint_gaussian_boundary_probability(
    mean: np.ndarray,
    covariance: np.ndarray,
    boundary: Callable[[float], float],
    *,
    tolerance: float,
) -> tuple[float, float]:
    """Adaptively integrate P(eta1 <= boundary(eta0)) with an error estimate."""

    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("safety integration requires a bivariate Gaussian")
    if tolerance <= 0.0:
        raise ValueError("safety integration tolerance must be positive")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise PosteriorFitError("safety moments are not finite")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
        raise PosteriorFitError("safety covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest_eigenvalue = float(eigenvalues[-1])
    if largest_eigenvalue <= 0.0 or float(eigenvalues[0]) < -1e-12 * max(
        largest_eigenvalue, 1.0
    ):
        raise PosteriorFitError("safety covariance is not positive semidefinite")
    variance0 = float(covariance[0, 0])
    variance1 = float(covariance[1, 1])
    if variance0 <= 0.0 or variance1 <= 0.0:
        raise PosteriorFitError("safety covariance has a non-positive variance")
    sd0 = math.sqrt(variance0)
    sd1 = math.sqrt(variance1)
    covariance01 = float(covariance[0, 1])
    slope = covariance01 / sd0
    conditional_variance = sd1 * sd1 - slope * slope
    if conditional_variance <= 0.0:
        raise PosteriorFitError("safety covariance is not positive semidefinite")
    conditional_sd = math.sqrt(conditional_variance)
    if conditional_sd / sd1 < _MIN_SAFETY_CONDITIONAL_SD_RATIO:
        raise PosteriorFitError(
            "safety covariance is too close to singular for certified integration"
        )

    def gap(z_value: float) -> float:
        eta0 = float(mean[0] + sd0 * z_value)
        return float(boundary(eta0) - (mean[1] + slope * z_value))

    grid = np.linspace(
        -_SAFETY_INTEGRATION_LIMIT,
        _SAFETY_INTEGRATION_LIMIT,
        801,
    )
    gaps = [gap(float(value)) for value in grid]
    breakpoints: list[float] = []
    for left, right, left_gap, right_gap in zip(
        grid[:-1], grid[1:], gaps[:-1], gaps[1:]
    ):
        if (left_gap < 0.0 < right_gap) or (right_gap < 0.0 < left_gap):
            breakpoints.append(_bisect_sign_change(gap, float(left), float(right)))
    for index in range(1, len(grid) - 1):
        left_gap, middle_gap, right_gap = gaps[index - 1 : index + 2]
        if not all(math.isfinite(value) for value in (left_gap, middle_gap, right_gap)):
            continue
        left_slope = middle_gap - left_gap
        right_slope = right_gap - middle_gap
        if left_slope * right_slope >= 0.0:
            continue
        direction = 1.0 if left_slope < 0.0 < right_slope else -1.0
        result = minimize_scalar(
            lambda value: direction * gap(float(value)),
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if result.success:
            breakpoints.append(float(result.x))
    breakpoints = sorted(
        {
            round(point, 14)
            for point in breakpoints
            if -_SAFETY_INTEGRATION_LIMIT < point < _SAFETY_INTEGRATION_LIMIT
        }
    )
    if len(breakpoints) > _MAX_SAFETY_BREAKPOINTS:
        raise PosteriorFitError("safety boundary has too many integration breakpoints")

    def integrand(z_value: float) -> float:
        gap_value = gap(z_value)
        conditional = float(ndtr(gap_value / conditional_sd))
        return float(norm.pdf(z_value) * conditional)

    estimate, quadrature_error = quad(
        integrand,
        -_SAFETY_INTEGRATION_LIMIT,
        _SAFETY_INTEGRATION_LIMIT,
        points=breakpoints or None,
        epsabs=tolerance * 0.25,
        epsrel=tolerance * 0.25,
        limit=400,
    )
    tail_error = float(2.0 * ndtr(-_SAFETY_INTEGRATION_LIMIT))
    error = float(quadrature_error + tail_error)
    if not math.isfinite(estimate) or not math.isfinite(error):
        raise PosteriorFitError("safety integration returned a non-finite result")
    if error > tolerance:
        raise PosteriorFitError(
            f"safety integration error {error:.3g} exceeds {tolerance:.3g}"
        )
    return min(max(float(estimate), 0.0), 1.0), error


def _risk_difference_boundary(eta0: float, difference: float) -> float:
    target = float(expit(eta0)) + difference
    if target <= 0.0:
        return -math.inf
    if target >= 1.0:
        return math.inf
    return float(logit(target))


def _effective_absolute_invalidity_limit(value: float | None) -> float:
    return 1.0 if value is None else value


def _deterministic_safety_summary(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    max_treated_invalid_probability: float | None,
    max_incremental_invalid_probability: float,
    alpha: float,
    integration_tolerance: float,
) -> tuple[float, float, float, float]:
    treated_upper, safe_probability, safe_error = _deterministic_safety_gate(
        mean,
        covariance,
        max_treated_invalid_probability=max_treated_invalid_probability,
        max_incremental_invalid_probability=max_incremental_invalid_probability,
        alpha=alpha,
        integration_tolerance=integration_tolerance,
    )

    def difference_cdf_lower(value: float) -> float:
        estimate, error = _joint_gaussian_boundary_probability(
            mean,
            covariance,
            lambda eta0: _risk_difference_boundary(eta0, value),
            tolerance=integration_tolerance,
        )
        return max(0.0, estimate - error)

    target = 1.0 - alpha
    lower, upper = -1.0 + 1e-10, 1.0 - 1e-10
    if difference_cdf_lower(lower) >= target:
        difference_upper = -1.0
    elif difference_cdf_lower(upper) <= target:
        difference_upper = 1.0
    else:
        difference_upper = float(
            brentq(
                lambda value: difference_cdf_lower(value) - target,
                lower,
                upper,
                xtol=1e-7,
            )
        )
    return treated_upper, difference_upper, safe_probability, safe_error


def _deterministic_safety_gate(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    max_treated_invalid_probability: float | None,
    max_incremental_invalid_probability: float,
    alpha: float,
    integration_tolerance: float,
) -> tuple[float, float, float]:
    """Return the exact quantities required to certify one safety gate."""

    absolute_limit = _effective_absolute_invalidity_limit(
        max_treated_invalid_probability
    )
    treated_limit = (
        -math.inf
        if absolute_limit <= 0.0
        else math.inf
        if absolute_limit >= 1.0
        else float(logit(absolute_limit))
    )

    def safe_boundary(eta0: float) -> float:
        return min(
            treated_limit,
            _risk_difference_boundary(eta0, max_incremental_invalid_probability),
        )

    safe_estimate, safe_error = _joint_gaussian_boundary_probability(
        mean,
        covariance,
        safe_boundary,
        tolerance=integration_tolerance,
    )
    safe_probability = max(0.0, safe_estimate - safe_error)
    treated_sd = math.sqrt(max(float(covariance[1, 1]), 0.0))
    treated_upper = float(expit(mean[1] + norm.ppf(1.0 - alpha) * treated_sd))
    return treated_upper, safe_probability, safe_error


def _inverse_spd(matrix: np.ndarray, *, jitter: float) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=float)
    current = jitter
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(matrix + current * identity)
            inverse_chol = np.linalg.solve(chol, identity)
            return inverse_chol.T @ inverse_chol
        except np.linalg.LinAlgError:
            current *= 10.0
    raise PosteriorFitError("posterior precision is not positive definite")


def _logdet_spd(matrix: np.ndarray, *, jitter: float) -> float:
    identity = np.eye(matrix.shape[0], dtype=float)
    current = jitter
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(matrix + current * identity)
            return float(2.0 * np.log(np.diag(chol)).sum())
        except np.linalg.LinAlgError:
            current *= 10.0
    raise PosteriorFitError("matrix log determinant is undefined")


def _cholesky_spd(matrix: np.ndarray, *, jitter: float) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=float)
    current = jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + current * identity)
        except np.linalg.LinAlgError:
            current *= 10.0
    raise PosteriorFitError("posterior covariance is not positive definite")


class TerminalUtilityPosteriorConfig(BaseModel):
    """Numerical and prior schema for the bounded v2 core."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    baseline_prior_sd: float = Field(default=0.75, gt=0.0)
    shared_effect_prior_sd: float = Field(default=0.35, gt=0.0)
    card_effect_prior_sd: float = Field(default=0.25, gt=0.0)
    reward_residual_sd_initial: float = Field(default=0.20, gt=0.0)
    reward_residual_sd_bounds: tuple[float, float] = (0.01, 5.0)
    reward_residual_log_prior_sd: float = Field(default=0.75, gt=0.0)
    reward_residual_mode_grid_points: int = Field(default=17, ge=9, le=41)
    reward_residual_quadrature_mass_rtol: float = Field(default=1e-5, gt=0.0)
    reward_residual_moment_rtol: float = Field(default=1e-3, gt=0.0)
    reward_coefficient_mean_tolerance: float = Field(default=1e-3, gt=0.0)
    reward_residual_upper_boundary_mass_limit: float = Field(default=1e-4, gt=0.0)
    reward_residual_quadrature_max_depth: int = Field(default=8, ge=1, le=12)
    unknown_measurement_sd: float = Field(default=0.15, gt=0.0)
    reference_offer_probability: float = Field(default=0.5, gt=0.0, lt=1.0)
    safety_integration_tolerance: float = Field(default=1e-8, gt=0.0, lt=1e-3)
    invalidity_prior_probability: float = Field(default=0.05, gt=0.0, lt=0.5)
    safety_baseline_prior_sd: float = Field(default=0.15, gt=0.0)
    safety_shared_effect_prior_mean: float = 0.0
    safety_shared_effect_prior_sd: float = Field(default=0.20, gt=0.0)
    safety_card_effect_prior_sd: float = Field(default=0.60, gt=0.0)
    optimizer_max_iterations: int = Field(default=400, ge=50)
    optimizer_gradient_tolerance: float = Field(default=1e-6, gt=0.0)
    max_hessian_condition: float = Field(default=1e12, gt=1.0)
    jitter: float = Field(default=1e-9, gt=0.0)


@dataclass(frozen=True)
class GaussianPosteriorComponent:
    mean: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray
    residual_sd: float
    probability: float


@dataclass(frozen=True)
class GaussianPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    residual_sd: float
    residual_variance_mean: float
    log_marginal: float
    observations: int
    card_effect_sd: float
    optimizer_method: str
    optimizer_success: bool
    optimizer_iterations: int
    hyperparameters_at_boundary: bool
    residual_scale_ess: float
    residual_quadrature_error: float
    residual_moment_error: float
    coefficient_mean_error: float
    residual_boundary_probability: float
    residual_upper_boundary_probability: float
    components: tuple[GaussianPosteriorComponent, ...]

    def sample_many(
        self, rng: np.random.Generator, samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if samples < 1:
            raise ValueError("samples must be positive")
        probabilities = np.asarray(
            [component.probability for component in self.components], dtype=float
        )
        indices = rng.choice(len(self.components), size=samples, p=probabilities)
        coefficients = np.empty((samples, len(self.mean)), dtype=float)
        residual_sds = np.empty(samples, dtype=float)
        for index in np.unique(indices):
            selected = np.flatnonzero(indices == index)
            component = self.components[int(index)]
            coefficients[selected] = (
                component.mean
                + rng.standard_normal((len(selected), len(self.mean)))
                @ component.factor.T
            )
            residual_sds[selected] = component.residual_sd
        return coefficients, residual_sds


@dataclass(frozen=True)
class LogisticPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    observations: int
    objective: float
    gradient_norm: float
    hessian_condition: float
    optimizer_method: str
    optimizer_iterations: int


class BayesianResidualScaleGaussianRegressor:
    """Adaptive posterior mixture over log residual scale."""

    def __init__(self, config: TerminalUtilityPosteriorConfig) -> None:
        self.config = config

    def fit(
        self,
        design: np.ndarray,
        values: np.ndarray,
        measurement_sd: np.ndarray,
        space: FeatureSpace,
    ) -> GaussianPosterior:
        dimension = space.outcome_dim
        if design.shape != (len(values), dimension):
            raise ValueError("reward design shape is inconsistent")
        if measurement_sd.shape != values.shape:
            raise ValueError("measurement standard errors are misaligned")
        lower, upper = self.config.reward_residual_sd_bounds
        if lower <= 0.0 or upper <= lower:
            raise ValueError("reward residual bounds must be positive and ordered")
        prior_center = math.log(self.config.reward_residual_sd_initial)
        prior_sd = self.config.reward_residual_log_prior_sd
        log_lower, log_upper = math.log(lower), math.log(upper)
        if log_upper <= log_lower:
            raise ValueError("reward residual prior has no support inside its bounds")

        cache: dict[float, tuple[np.ndarray, np.ndarray, float, float]] = {}

        def evaluate(
            log_residual_sd: float,
        ) -> tuple[np.ndarray, np.ndarray, float, float]:
            key = float(log_residual_sd)
            cached = cache.get(key)
            if cached is not None:
                return cached
            conditional = self._conditional(
                design,
                values,
                measurement_sd,
                space,
                residual_sd=float(math.exp(key)),
            )
            log_kernel = conditional[2] + float(
                norm.logpdf(key, loc=prior_center, scale=prior_sd)
            )
            result = (conditional[0], conditional[1], conditional[2], log_kernel)
            cache[key] = result
            return result

        scan = np.linspace(
            log_lower, log_upper, self.config.reward_residual_mode_grid_points
        )
        scan_values = np.asarray([evaluate(float(node))[3] for node in scan])
        local_indices = [
            index
            for index in range(1, len(scan) - 1)
            if scan_values[index] >= scan_values[index - 1]
            and scan_values[index] >= scan_values[index + 1]
        ]
        global_scan_index = int(np.argmax(scan_values))
        if 0 < global_scan_index < len(scan) - 1:
            local_indices.append(global_scan_index)
        modes: list[tuple[float, float]] = [
            (float(scan[0]), float(scan_values[0])),
            (float(scan[-1]), float(scan_values[-1])),
        ]
        mode_success = True
        for index in sorted(set(local_indices)):
            result = minimize_scalar(
                lambda node: -evaluate(float(node))[3],
                bounds=(float(scan[index - 1]), float(scan[index + 1])),
                method="bounded",
                options={"xatol": 1e-10},
            )
            mode_success = mode_success and bool(result.success)
            modes.append((float(result.x), float(-result.fun)))
        modes.sort(key=lambda row: row[1], reverse=True)
        global_mode, maximum = modes[0]
        retained_modes = [mode for mode, height in modes if height >= maximum - 24.0]

        symmetric_distance = min(global_mode - log_lower, log_upper - global_mode)
        if symmetric_distance > 1e-8:
            h = min(0.02 * (log_upper - log_lower), 0.25 * symmetric_distance)
            curvature = -1.0 / (prior_sd * prior_sd)
            for _ in range(3):
                curvature = (
                    evaluate(global_mode + h)[3]
                    - 2.0 * maximum
                    + evaluate(global_mode - h)[3]
                ) / (h * h)
                if not math.isfinite(curvature) or curvature >= 0.0:
                    curvature = -1.0 / (prior_sd * prior_sd)
                    break
                local_sd = 1.0 / math.sqrt(-curvature)
                h = max(1e-5, min(0.25 * local_sd, h))
        else:
            curvature = -1.0 / (prior_sd * prior_sd)
        local_sd = 1.0 / math.sqrt(-curvature)

        span = log_upper - log_lower
        lower_boundary_cut = log_lower + 0.01 * span
        upper_boundary_cut = log_upper - 0.01 * span
        breakpoints = {
            log_lower,
            lower_boundary_cut,
            upper_boundary_cut,
            log_upper,
            *retained_modes,
        }
        for mode in retained_modes:
            mode_height = evaluate(mode)[3]
            for drop in (0.5, 2.0, 8.0, 24.0):
                target = mode_height - drop
                expected = math.sqrt(2.0 * drop) * local_sd
                for direction, boundary in ((-1.0, log_lower), (1.0, log_upper)):
                    probe = min(
                        max(mode + direction * 2.0 * expected, log_lower),
                        log_upper,
                    )
                    while probe != boundary and evaluate(probe)[3] > target:
                        distance = abs(probe - mode) * 2.0
                        probe = min(
                            max(mode + direction * distance, log_lower),
                            log_upper,
                        )
                    if evaluate(probe)[3] <= target < mode_height:
                        root = brentq(
                            lambda node: evaluate(float(node))[3] - target,
                            min(mode, probe),
                            max(mode, probe),
                            xtol=1e-10,
                        )
                        breakpoints.add(float(root))

        legendre: dict[int, tuple[np.ndarray, np.ndarray]] = {
            points: np.polynomial.legendre.leggauss(points) for points in (8, 16)
        }

        def rule(
            left: float, right: float, points: int
        ) -> tuple[
            float,
            list[tuple[float, float, tuple[np.ndarray, np.ndarray, float, float]]],
        ]:
            raw_nodes, raw_weights = legendre[points]
            half_width = 0.5 * (right - left)
            midpoint = 0.5 * (right + left)
            nodes = midpoint + half_width * raw_nodes
            weights = half_width * raw_weights
            rows = [
                (float(node), float(weight), evaluate(float(node)))
                for node, weight in zip(nodes, weights)
            ]
            mass = sum(
                weight * math.exp(conditional[3] - maximum)
                for _, weight, conditional in rows
            )
            return mass, rows

        accepted: list[
            tuple[
                list[tuple[float, float, tuple[np.ndarray, np.ndarray, float, float]]],
                list[tuple[float, float, tuple[np.ndarray, np.ndarray, float, float]]],
            ]
        ] = []
        refinement_failed = False

        def refine(left: float, right: float, depth: int) -> None:
            nonlocal refinement_failed
            coarse_mass, coarse_rows = rule(left, right, 8)
            fine_mass, fine_rows = rule(left, right, 16)
            tolerance = 1e-10 + 1e-6 * fine_mass
            if abs(fine_mass - coarse_mass) <= tolerance:
                accepted.append((coarse_rows, fine_rows))
                return
            if depth >= self.config.reward_residual_quadrature_max_depth:
                refinement_failed = True
                accepted.append((coarse_rows, fine_rows))
                return
            midpoint = 0.5 * (left + right)
            refine(left, midpoint, depth + 1)
            refine(midpoint, right, depth + 1)

        ordered_breakpoints = sorted(breakpoints)
        for left, right in zip(ordered_breakpoints, ordered_breakpoints[1:]):
            if right > left:
                refine(left, right, 0)

        coarse_rows = [row for coarse, _ in accepted for row in coarse]
        fine_rows = [row for _, fine in accepted for row in fine]

        def summarize(rows):
            log_terms = np.asarray(
                [conditional[3] + math.log(weight) for _, weight, conditional in rows]
            )
            log_normalization = float(logsumexp(log_terms))
            probabilities = np.exp(log_terms - log_normalization)
            coefficient_mean = sum(
                float(probability) * conditional[0]
                for probability, (_, _, conditional) in zip(probabilities, rows)
            )
            residual_variance = sum(
                float(probability) * math.exp(2.0 * node)
                for probability, (node, _, _) in zip(probabilities, rows)
            )
            return (
                log_normalization,
                probabilities,
                coefficient_mean,
                float(residual_variance),
            )

        coarse_summary = summarize(coarse_rows)
        fine_summary = summarize(fine_rows)
        normalization = fine_summary[0]
        probabilities = fine_summary[1]
        mean = fine_summary[2]
        residual_variance_mean = fine_summary[3]

        covariance = np.zeros((dimension, dimension), dtype=float)
        for probability, (_, _, conditional) in zip(probabilities, fine_rows):
            centered = conditional[0] - mean
            covariance += float(probability) * (
                conditional[1] + np.outer(centered, centered)
            )
        posterior_sd = np.sqrt(np.maximum(np.diag(covariance), self.config.jitter))
        mass_error = abs(math.expm1(coarse_summary[0] - normalization))
        residual_moment_error = abs(coarse_summary[3] - residual_variance_mean) / max(
            residual_variance_mean, self.config.jitter
        )
        coefficient_mean_error = float(
            np.max(np.abs(coarse_summary[2] - mean) / posterior_sd)
        )
        lower_boundary_probability = sum(
            float(probability)
            for probability, (node, _, _) in zip(probabilities, fine_rows)
            if node < lower_boundary_cut
        )
        upper_boundary_probability = sum(
            float(probability)
            for probability, (node, _, _) in zip(probabilities, fine_rows)
            if node > upper_boundary_cut
        )
        boundary_probability = lower_boundary_probability + upper_boundary_probability
        boundary = (
            not math.isfinite(upper_boundary_probability)
            or upper_boundary_probability
            >= self.config.reward_residual_upper_boundary_mass_limit
        )
        unhealthy = (
            not mode_success
            or refinement_failed
            or not all(
                math.isfinite(value)
                for value in (
                    mass_error,
                    residual_moment_error,
                    coefficient_mean_error,
                )
            )
            or mass_error > self.config.reward_residual_quadrature_mass_rtol
            or residual_moment_error > self.config.reward_residual_moment_rtol
            or coefficient_mean_error > self.config.reward_coefficient_mean_tolerance
        )

        keep = probabilities >= float(np.max(probabilities)) * 1e-12
        kept_probabilities = probabilities[keep]
        kept_probabilities /= kept_probabilities.sum()
        components = tuple(
            GaussianPosteriorComponent(
                mean=conditional[0],
                covariance=conditional[1],
                factor=_cholesky_spd(conditional[1], jitter=self.config.jitter),
                residual_sd=float(math.exp(node)),
                probability=float(probability),
            )
            for probability, (node, _, conditional) in zip(
                kept_probabilities,
                (row for include, row in zip(keep, fine_rows) if include),
            )
        )
        residual_scale_ess = float(1.0 / np.sum(kept_probabilities**2))
        return GaussianPosterior(
            mean=mean,
            covariance=covariance,
            residual_sd=math.sqrt(residual_variance_mean),
            residual_variance_mean=residual_variance_mean,
            log_marginal=normalization,
            observations=len(values),
            card_effect_sd=self.config.card_effect_prior_sd,
            optimizer_method="adaptive_log_sigma_gauss_legendre",
            optimizer_success=not unhealthy,
            optimizer_iterations=len(cache),
            hyperparameters_at_boundary=boundary,
            residual_scale_ess=residual_scale_ess,
            residual_quadrature_error=float(mass_error),
            residual_moment_error=float(residual_moment_error),
            coefficient_mean_error=coefficient_mean_error,
            residual_boundary_probability=float(boundary_probability),
            residual_upper_boundary_probability=float(upper_boundary_probability),
            components=components,
        )

    def _conditional(
        self,
        design: np.ndarray,
        values: np.ndarray,
        measurement_sd: np.ndarray,
        space: FeatureSpace,
        *,
        residual_sd: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        prior_variance = space.prior_variance(
            baseline_sd=self.config.baseline_prior_sd,
            shared_effect_sd=self.config.shared_effect_prior_sd,
            card_effect_sd=self.config.card_effect_prior_sd,
        )
        prior_precision = 1.0 / prior_variance
        variance = residual_sd**2 + measurement_sd**2
        weights = 1.0 / variance
        precision = np.diag(prior_precision) + design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * values)
        covariance = _inverse_spd(precision, jitter=self.config.jitter)
        mean = covariance @ rhs
        quadratic = float(values @ (weights * values) - mean @ precision @ mean)
        log_marginal = -0.5 * (
            len(values) * math.log(2.0 * math.pi)
            + float(np.log(variance).sum())
            + _logdet_spd(precision, jitter=self.config.jitter)
            - float(np.log(prior_precision).sum())
            + quadratic
        )
        return mean, covariance, log_marginal


class StableBayesianLogisticRegressor:
    """Proper-prior logistic MAP/Laplace with fail-closed diagnostics."""

    def __init__(self, config: TerminalUtilityPosteriorConfig) -> None:
        self.config = config

    def fit(
        self,
        design: np.ndarray,
        invalid: np.ndarray,
        prior_mean: np.ndarray,
        prior_variance: np.ndarray,
    ) -> LogisticPosterior:
        dimension = design.shape[1]
        if design.shape[0] != len(invalid):
            raise ValueError("safety design and targets are misaligned")
        if prior_mean.shape != (dimension,) or prior_variance.shape != (dimension,):
            raise ValueError("safety prior shape is inconsistent")
        prior_precision = 1.0 / prior_variance

        def objective(coefficients: np.ndarray) -> float:
            linear = design @ coefficients
            centered = coefficients - prior_mean
            return float(
                np.logaddexp(0.0, linear).sum()
                - invalid @ linear
                + 0.5 * np.sum(prior_precision * centered * centered)
            )

        def gradient(coefficients: np.ndarray) -> np.ndarray:
            probability = expit(design @ coefficients)
            return design.T @ (probability - invalid) + prior_precision * (
                coefficients - prior_mean
            )

        optimizer_method = "prior"
        optimizer_iterations = 0
        if len(invalid):
            result = minimize(
                objective,
                prior_mean,
                jac=gradient,
                method="L-BFGS-B",
                options={
                    "maxiter": self.config.optimizer_max_iterations,
                    "ftol": 1e-12,
                    "gtol": self.config.optimizer_gradient_tolerance,
                    "maxls": 50,
                },
            )
            coefficients = np.asarray(result.x, dtype=float)
            gradient_norm = float(np.linalg.norm(gradient(coefficients), ord=np.inf))
            optimizer_method = "L-BFGS-B"
            optimizer_iterations = int(result.nit)
            if (
                not result.success
                or not np.isfinite(result.fun)
                or gradient_norm > 10.0 * self.config.optimizer_gradient_tolerance
            ):

                def hessian_product(
                    coefficients: np.ndarray, vector: np.ndarray
                ) -> np.ndarray:
                    probability = expit(design @ coefficients)
                    curvature = probability * (1.0 - probability)
                    return design.T @ (curvature * (design @ vector)) + (
                        prior_precision * vector
                    )

                fallback = minimize(
                    objective,
                    coefficients,
                    jac=gradient,
                    hessp=hessian_product,
                    method="Newton-CG",
                    options={
                        "maxiter": self.config.optimizer_max_iterations,
                        "xtol": self.config.optimizer_gradient_tolerance,
                    },
                )
                fallback_coefficients = np.asarray(fallback.x, dtype=float)
                fallback_gradient = float(
                    np.linalg.norm(gradient(fallback_coefficients), ord=np.inf)
                )
                if not np.isfinite(fallback.fun) or fallback_gradient > max(
                    1e-5, 10.0 * self.config.optimizer_gradient_tolerance
                ):
                    raise PosteriorFitError(
                        "safety MAP failed convergence: "
                        f"lbfgs_success={result.success}, "
                        f"newton_success={fallback.success}, "
                        f"gradient_inf={fallback_gradient:.3g}"
                    )
                coefficients = fallback_coefficients
                gradient_norm = fallback_gradient
                optimizer_method = "Newton-CG"
                optimizer_iterations += int(fallback.nit)
        else:
            coefficients = prior_mean.copy()
            gradient_norm = 0.0
        probability = expit(design @ coefficients)
        curvature = probability * (1.0 - probability)
        hessian = np.diag(prior_precision) + design.T @ (curvature[:, None] * design)
        condition = float(np.linalg.cond(hessian))
        if not np.isfinite(condition) or condition > self.config.max_hessian_condition:
            raise PosteriorFitError(
                f"safety Hessian is ill-conditioned: {condition:.3g}"
            )
        covariance = _inverse_spd(hessian, jitter=self.config.jitter)
        return LogisticPosterior(
            mean=coefficients,
            covariance=covariance,
            observations=len(invalid),
            objective=objective(coefficients),
            gradient_norm=gradient_norm,
            hessian_condition=condition,
            optimizer_method=optimizer_method,
            optimizer_iterations=optimizer_iterations,
        )


class FittedTerminalUtilityPosterior:
    def __init__(
        self,
        *,
        space: FeatureSpace,
        reward: GaussianPosterior,
        safety: LogisticPosterior,
        model_config_hash: str,
        evidence_count: int,
        offer_probability_by_treatment: dict[str, float],
        reference_offer_probability: float,
        safety_integration_tolerance: float,
        reward_definition: RewardDefinition | None,
        semantic_schema_hash: str | None,
    ) -> None:
        self.space = space
        self.reward = reward
        self.safety = safety
        self.model_config_hash = model_config_hash
        self.evidence_count = evidence_count
        self.offer_probability_by_treatment = dict(offer_probability_by_treatment)
        self.reference_offer_probability = reference_offer_probability
        self.safety_integration_tolerance = safety_integration_tolerance
        self.reward_definition = reward_definition
        self.semantic_schema_hash = semantic_schema_hash
        self._safety_factor = _cholesky_spd(safety.covariance, jitter=1e-12)

    def _arm_designs(
        self,
        card: CardSnapshot,
        context: EvolutionContext,
        *,
        rag_applicable: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.reward_definition is not None
            and context.reward != self.reward_definition
        ):
            raise ValueError(
                "prediction reward definition differs from fitted evidence"
            )
        if (
            self.semantic_schema_hash is not None
            and context.map_elites.semantic_schema_hash != self.semantic_schema_hash
        ):
            raise ValueError("prediction semantic schema differs from fitted evidence")
        return (
            self.space.design(
                card,
                context,
                False,
                rag_applicable=rag_applicable,
            ),
            self.space.design(
                card,
                context,
                True,
                rag_applicable=rag_applicable,
            ),
        )

    @staticmethod
    def _draws(
        mean: np.ndarray,
        factor: np.ndarray,
        rng: np.random.Generator,
        samples: int,
    ) -> np.ndarray:
        return mean + rng.standard_normal((samples, len(mean))) @ factor.T

    def sample_usable_effects(
        self,
        cards: Sequence[CardSnapshot],
        context: EvolutionContext,
        rng: np.random.Generator,
        *,
        samples: int,
        applicable_bank_card_ids: frozenset[str] = frozenset(),
    ) -> np.ndarray:
        """Draw shared posterior worlds as a samples-by-cards effect matrix."""

        if samples < 1:
            raise ValueError("samples must be positive")
        if not cards:
            return np.empty((samples, 0), dtype=float)
        reward_draws, _ = self.reward.sample_many(rng, samples)
        safety_draws = self._draws(self.safety.mean, self._safety_factor, rng, samples)
        reward_designs = [
            self._arm_designs(
                card,
                context,
                rag_applicable=card.bank_card_id in applicable_bank_card_ids,
            )
            for card in cards
        ]
        reward_design0 = np.stack([row[0] for row in reward_designs])
        reward_design1 = np.stack([row[1] for row in reward_designs])
        valid0 = _latent_to_gain(reward_draws @ reward_design0.T, context)
        valid1 = _latent_to_gain(reward_draws @ reward_design1.T, context)
        p0 = expit(safety_draws @ reward_design0.T)
        p1 = expit(safety_draws @ reward_design1.T)
        lower, _ = _normalized_gain_bounds(context)
        q0 = (1.0 - p0) * valid0 + p0 * lower
        q1 = (1.0 - p1) * valid1 + p1 * lower
        return q1 - q0

    def predictions(
        self,
        cards: Sequence[CardSnapshot],
        context: EvolutionContext,
        rng: np.random.Generator,
        *,
        samples: int,
        max_treated_invalid_probability: float | None,
        max_incremental_invalid_probability: float,
        safety_alpha: float,
        applicable_bank_card_ids: frozenset[str] = frozenset(),
    ) -> dict[str, PosteriorPrediction]:
        """Summarize every candidate using one coherent set of posterior worlds."""

        if samples < 64:
            raise ValueError("posterior summaries require at least 64 samples")
        reward_draws, reward_residual_sds = self.reward.sample_many(rng, samples)
        safety_draws = self._draws(self.safety.mean, self._safety_factor, rng, samples)
        return {
            card.treatment_id: self._prediction_from_draws(
                card,
                context,
                reward_draws,
                reward_residual_sds,
                safety_draws,
                rag_applicable=card.bank_card_id in applicable_bank_card_ids,
                max_treated_invalid_probability=max_treated_invalid_probability,
                max_incremental_invalid_probability=(
                    max_incremental_invalid_probability
                ),
                safety_alpha=safety_alpha,
            )
            for card in cards
        }

    def prediction(
        self,
        card: CardSnapshot,
        context: EvolutionContext,
        rng: np.random.Generator,
        *,
        samples: int,
        max_treated_invalid_probability: float | None,
        max_incremental_invalid_probability: float,
        safety_alpha: float,
        rag_applicable: bool = False,
    ) -> PosteriorPrediction:
        return self.predictions(
            (card,),
            context,
            rng,
            samples=samples,
            max_treated_invalid_probability=max_treated_invalid_probability,
            max_incremental_invalid_probability=max_incremental_invalid_probability,
            safety_alpha=safety_alpha,
            applicable_bank_card_ids=(
                frozenset({card.bank_card_id}) if rag_applicable else frozenset()
            ),
        )[card.treatment_id]

    def _prediction_from_draws(
        self,
        card: CardSnapshot,
        context: EvolutionContext,
        reward_draws: np.ndarray,
        reward_residual_sds: np.ndarray,
        safety_draws: np.ndarray,
        *,
        rag_applicable: bool,
        max_treated_invalid_probability: float | None,
        max_incremental_invalid_probability: float,
        safety_alpha: float,
    ) -> PosteriorPrediction:
        x0, x1 = self._arm_designs(
            card,
            context,
            rag_applicable=rag_applicable,
        )
        latent0 = reward_draws @ x0
        latent1 = reward_draws @ x1
        valid0 = _latent_to_gain(latent0, context)
        valid1 = _latent_to_gain(latent1, context)
        p0 = expit(safety_draws @ x0)
        p1 = expit(safety_draws @ x1)
        lower, _ = _normalized_gain_bounds(context)
        q0 = (1.0 - p0) * valid0 + p0 * lower
        q1 = (1.0 - p1) * valid1 + p1 * lower
        risk_delta = p1 - p0
        effects = q1 - q0
        absolute_limit = _effective_absolute_invalidity_limit(
            max_treated_invalid_probability
        )
        safe = (p1 <= absolute_limit) & (
            risk_delta <= max_incremental_invalid_probability
        )
        helpful = effects > 0.0
        safety_mean = np.asarray(
            [x0 @ self.safety.mean, x1 @ self.safety.mean],
            dtype=float,
        )
        safety_matrix = np.stack((x0, x1))
        safety_covariance = safety_matrix @ self.safety.covariance @ safety_matrix.T
        try:
            (
                treated_invalid_upper,
                incremental_invalid_upper,
                probability_safe,
                safety_integration_error,
            ) = _deterministic_safety_summary(
                safety_mean,
                safety_covariance,
                max_treated_invalid_probability=max_treated_invalid_probability,
                max_incremental_invalid_probability=(
                    max_incremental_invalid_probability
                ),
                alpha=safety_alpha,
                integration_tolerance=self.safety_integration_tolerance,
            )
        except PosteriorFitError:
            # Preserve the auditable posterior row while excluding this action.
            # A numerical certification failure must not abort evolution or be
            # mistaken for evidence that a card is safe.
            treated_invalid_upper = 1.0
            incremental_invalid_upper = 1.0
            probability_safe = 0.0
            safety_integration_error = 1.0
        within_world_variance = (1.0 - p1) * reward_residual_sds**2 + p1 * (
            1.0 - p1
        ) * (valid1 - lower) ** 2
        predictive_variance = max(
            float(q1.var(ddof=1)) + float(within_world_variance.mean()),
            0.0,
        )
        return PosteriorPrediction(
            treatment_id=card.treatment_id,
            usable_effect_mean=float(effects.mean()),
            usable_effect_sd=float(effects.std(ddof=1)),
            probability_helpful=float(helpful.mean()),
            usable_gain_control_mean=float(q0.mean()),
            usable_gain_treated_mean=float(q1.mean()),
            control_invalid_probability=float(p0.mean()),
            treated_invalid_probability=float(p1.mean()),
            incremental_invalid_probability=float(risk_delta.mean()),
            treated_invalid_upper=treated_invalid_upper,
            incremental_invalid_upper=incremental_invalid_upper,
            probability_safe=probability_safe,
            safety_integration_error=safety_integration_error,
            probability_safe_and_helpful=float((safe & helpful).mean()),
            usable_gain_predictive_sd=math.sqrt(predictive_variance),
        )


class HierarchicalTerminalUtilityPosterior:
    """Fit a bounded valid-gain/invalidity hurdle posterior."""

    MODEL_NAME = "hierarchical-hurdle-terminal-utility"

    def __init__(
        self,
        *,
        feature_map: HierarchicalFeatureMap,
        config: TerminalUtilityPosteriorConfig,
    ) -> None:
        self.feature_map = feature_map
        self.config = config
        self.reward_regressor = BayesianResidualScaleGaussianRegressor(config)
        self.safety_regressor = StableBayesianLogisticRegressor(config)
        feature_payload = feature_map.config.model_dump(mode="json")
        self.model_config_hash = canonical_digest(
            {
                "model": self.MODEL_NAME,
                "features": feature_payload,
                "posterior": config.model_dump(mode="json"),
            }
        )

    def fit(
        self,
        observations: Sequence[CausalObservation],
        candidates: Sequence[CardSnapshot],
        *,
        reward_observations: Sequence[CausalObservation] | None = None,
    ) -> FittedTerminalUtilityPosterior:
        rows = tuple(observations)
        if reward_observations is None:
            if any(row.context.reward.lineage_depth > 1 for row in rows):
                raise ValueError(
                    "depth-greater-than-one reward fitting requires explicit "
                    "matured lineage observations"
                )
            reward_rows = rows
        else:
            reward_rows = tuple(reward_observations)
        validate_reward_observations(rows, reward_rows)
        reward_definitions = {row.context.reward for row in (*rows, *reward_rows)}
        if len(reward_definitions) > 1:
            raise ValueError("posterior evidence mixes reward definitions")
        semantic_schema_hashes = {
            row.context.map_elites.semantic_schema_hash for row in rows
        }
        if len(semantic_schema_hashes) > 1:
            raise ValueError("posterior evidence mixes semantic behavior schemas")
        mismatched_propensities = sorted(
            {
                row.offer_propensity
                for row in rows
                if not math.isclose(
                    row.offer_propensity,
                    self.config.reference_offer_probability,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            }
        )
        if mismatched_propensities:
            raise ValueError(
                "terminal-utility posterior requires one fixed offer probability; "
                f"reference={self.config.reference_offer_probability}, "
                f"observed={mismatched_propensities}"
            )
        all_cards = tuple(
            (
                *(observation.card for observation in rows),
                *(observation.card for observation in reward_rows),
                *candidates,
            )
        )
        space = self.feature_map.space(all_cards)
        safety_design = self._matrix(
            [
                space.design(
                    row.card,
                    row.context,
                    row.treatment,
                    rag_applicable=row.rag_applicable,
                )
                for row in rows
            ],
            space.outcome_dim,
        )
        invalid = np.asarray([float(row.invalid) for row in rows], dtype=float)
        safety_prior_mean = np.zeros(space.outcome_dim, dtype=float)
        safety_prior_mean[0] = float(logit(self.config.invalidity_prior_probability))
        safety_prior_mean[space.baseline_dim] = (
            self.config.safety_shared_effect_prior_mean
        )
        safety_prior_variance = space.prior_variance(
            baseline_sd=self.config.safety_baseline_prior_sd,
            shared_effect_sd=self.config.safety_shared_effect_prior_sd,
            card_effect_sd=self.config.safety_card_effect_prior_sd,
        )
        safety = self.safety_regressor.fit(
            safety_design,
            invalid,
            safety_prior_mean,
            safety_prior_variance,
        )

        valid_rows = tuple(row for row in reward_rows if not row.invalid)
        reward_design = self._matrix(
            [
                space.design(
                    row.card,
                    row.context,
                    row.treatment,
                    rag_applicable=row.rag_applicable,
                )
                for row in valid_rows
            ],
            space.outcome_dim,
        )
        reward_values: list[float] = []
        measurement_sd: list[float] = []
        for row in valid_rows:
            if row.measurement is None:
                raise ValueError("a valid terminal row is missing its gain measurement")
            normalized_value = row.measurement.value / row.context.reward.scale
            normalized_se = (
                self.config.unknown_measurement_sd
                if row.measurement.se is None
                else row.measurement.se / row.context.reward.scale
            )
            reward_values.append(_gain_to_model_scale(normalized_value, row.context))
            measurement_sd.append(normalized_se)
        reward_value_array = np.asarray(reward_values, dtype=float)
        measurement_sd_array = np.asarray(measurement_sd, dtype=float)
        reward = self.reward_regressor.fit(
            reward_design,
            reward_value_array,
            measurement_sd_array,
            space,
        )
        propensity_sum: dict[str, float] = {}
        propensity_count: dict[str, int] = {}
        for row in rows:
            treatment_id = row.card.treatment_id
            propensity_sum[treatment_id] = (
                propensity_sum.get(treatment_id, 0.0) + row.offer_propensity
            )
            propensity_count[treatment_id] = propensity_count.get(treatment_id, 0) + 1
        return FittedTerminalUtilityPosterior(
            space=space,
            reward=reward,
            safety=safety,
            model_config_hash=self.model_config_hash,
            evidence_count=len(rows),
            offer_probability_by_treatment={
                treatment_id: propensity_sum[treatment_id] / count
                for treatment_id, count in propensity_count.items()
            },
            reference_offer_probability=self.config.reference_offer_probability,
            safety_integration_tolerance=self.config.safety_integration_tolerance,
            reward_definition=next(iter(reward_definitions), None),
            semantic_schema_hash=next(iter(semantic_schema_hashes), None),
        )

    @staticmethod
    def _matrix(rows: Sequence[np.ndarray], dimension: int) -> np.ndarray:
        if not rows:
            return np.empty((0, dimension), dtype=float)
        return np.stack(rows).astype(float, copy=False)
