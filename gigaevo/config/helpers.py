"""Tiny helper functions for Hydra config computations."""

from typing import Any

from omegaconf import OmegaConf

from gigaevo.entrypoint.default_pipelines import (
    ContextPipelineBuilder,
    DefaultPipelineBuilder,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.evolution.strategies.map_elites import BehaviorSpace, IslandConfig
from gigaevo.evolution.strategies.models import BinningType
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext


def get_metrics_context(problem_context: ProblemContext) -> MetricsContext:
    """Extract metrics_context from ProblemContext."""
    return problem_context.metrics_context


def get_primary_key(metrics_context: MetricsContext) -> str:
    """Get primary metric key."""
    return metrics_context.get_primary_key()


def is_higher_better(metrics_context: MetricsContext, key: str) -> bool:
    """Check if metric is higher-is-better."""
    return metrics_context.is_higher_better(key)


def get_bounds(metrics_context: MetricsContext, key: str) -> tuple[float, float]:
    """Get bounds for a metric."""
    return metrics_context.get_bounds(key)


def build_behavior_space(
    keys: list[str],
    bounds: list[tuple[float, float]],
    resolutions: list[int],
    binning_types: list[str],
) -> Any:
    """Build a BehaviorSpace from lists of parameters.

    Args:
        keys: List of behavior feature keys (e.g., ['fitness', 'is_valid'])
        bounds: List of (min, max) bounds tuples (e.g., [(0, 1), (0, 1)])
        resolutions: List of resolution integers (e.g., [150, 2])
        binning_types: List of binning type strings (e.g., ['linear', 'linear'])

    Returns:
        BehaviorSpace instance

    Example:
        build_behavior_space(
            keys=['fitness', 'is_valid'],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            resolutions=[150, 2],
            binning_types=['linear', 'linear']
        )
    """

    if (
        len(keys) != len(bounds)
        or len(keys) != len(resolutions)
        or len(keys) != len(binning_types)
    ):
        raise ValueError("All parameter lists must have the same length")

    feature_bounds = {keys[i]: bounds[i] for i in range(len(keys))}
    resolution = {keys[i]: resolutions[i] for i in range(len(keys))}
    binning_types_dict = {
        keys[i]: BinningType(bt) for i, bt in enumerate(binning_types)
    }

    return BehaviorSpace(
        feature_bounds=feature_bounds,
        resolution=resolution,
        binning_types=binning_types_dict,
    )


def build_behavior_space_params(
    keys: list[str],
    bounds: list[tuple[float, float]],
    resolutions: list[int],
    binning_types: list[str] | None = None,
) -> OmegaConf:
    """Build all parameters needed for BehaviorSpace construction.

    This is a convenience helper that takes separate lists and constructs
    the dicts needed for BehaviorSpace feature_bounds, resolution, and binning_types.

    Args:
        keys: List of behavior feature keys (e.g., ['fitness', 'is_valid'])
        bounds: List of (min, max) bounds tuples (e.g., [(0, 1), (0, 1)])
        resolutions: List of resolution integers (e.g., [150, 2])
        binning_types: Optional list of binning type strings (e.g., ['linear', 'linear'])

    Returns:
        OmegaConf DictConfig with keys: feature_bounds, resolution, binning_types

    Example:
        build_behavior_space_params(
            keys=['fitness', 'is_valid'],
            bounds=[(0, 1), (0, 1)],
            resolutions=[150, 2],
            binning_types=['linear', 'linear']
        )
        -> {
            'feature_bounds': {'fitness': (0, 1), 'is_valid': (0, 1)},
            'resolution': {'fitness': 150, 'is_valid': 2},
            'binning_types': {'fitness': 'linear', 'is_valid': 'linear'}
        }
    """
    feature_bounds = {keys[i]: bounds[i] for i in range(len(keys))}
    resolution = {keys[i]: resolutions[i] for i in range(len(keys))}

    binning_types_dict: dict[str, BinningType] = {}
    if binning_types:
        binning_types_dict = {
            keys[i]: BinningType(bt) if isinstance(bt, str) else bt
            for i, bt in enumerate(binning_types)
        }

    return OmegaConf.create(
        {
            "feature_bounds": feature_bounds,
            "resolution": resolution,
            "binning_types": binning_types_dict,
        }
    )


def extract_behavior_keys_from_islands(island_configs: list[IslandConfig]) -> set[str]:
    """Extract all behavior keys from islands."""
    keys = set()
    for island in island_configs:
        keys |= set(island.behavior_space.behavior_keys)
    return keys


def build_dag_from_builder(builder: Any) -> Any:
    """Build DAG blueprint from pipeline builder."""
    return builder.build_blueprint()


def select_pipeline_builder(
    problem_context: ProblemContext,
    evolution_context: EvolutionContext,
) -> ContextPipelineBuilder | DefaultPipelineBuilder:
    """Select appropriate pipeline builder based on problem type."""
    if problem_context.is_contextual:
        return ContextPipelineBuilder(evolution_context)
    return DefaultPipelineBuilder(evolution_context)
