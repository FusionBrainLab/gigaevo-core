"""Tiny helper functions for Hydra config computations."""

from typing import Any

from omegaconf import OmegaConf

from gigaevo.entrypoint.default_pipelines import (
    ContextPipelineBuilder,
    DefaultPipelineBuilder,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.evolution.strategies.map_elites import IslandConfig
from gigaevo.evolution.strategies.models import (
    BehaviorSpace,
    DynamicBehaviorSpace,
    LinearBinning,
)
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec


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
    dynamic: bool = True,
    expansion_buffer_ratio: float = 0.1,
) -> Any:
    """Build a BehaviorSpace from lists of parameters.

    Args:
        keys: List of behavior feature keys (e.g., ['fitness', 'is_valid'])
        bounds: List of (min, max) bounds tuples (e.g., [(0, 1), (0, 1)])
                For dynamic spaces, these act as hard limits (clamping bounds).
        resolutions: List of resolution integers (e.g., [150, 2])
        binning_types: List of binning type strings (e.g., ['linear', 'linear'])
        dynamic: Whether to return a DynamicBehaviorSpace (default: True)
        expansion_buffer_ratio: Buffer ratio for dynamic space (default: 0.1)

    Returns:
        BehaviorSpace instance (or DynamicBehaviorSpace)

    Example:
        build_behavior_space(
            keys=['fitness', 'is_valid'],
            bounds=[(0.0, 1.0), (0.0, 1.0)],  # Hard limits for dynamic adjustment
            resolutions=[150, 2],
            binning_types=['linear', 'linear'],
            dynamic=True
        )
    """

    if (
        len(keys) != len(bounds)
        or len(keys) != len(resolutions)
        or len(keys) != len(binning_types)
    ):
        raise ValueError("All parameter lists must have the same length")

    bins = {}
    for i, key in enumerate(keys):
        min_val, max_val = bounds[i]
        num_bins = resolutions[i]
        b_type = binning_types[i]

        if b_type == "linear":
            strategy = LinearBinning(
                min_val=float(min_val), max_val=float(max_val), num_bins=num_bins
            )
        else:
            # Fallback to linear
            strategy = LinearBinning(
                min_val=float(min_val), max_val=float(max_val), num_bins=num_bins
            )

        bins[key] = strategy

    if dynamic:
        return DynamicBehaviorSpace(
            bins=bins,
            expansion_buffer_ratio=expansion_buffer_ratio,
        )
    return BehaviorSpace(bins=bins)


def build_behavior_space_params(
    keys: list[str],
    bounds: list[tuple[float, float]],
    resolutions: list[int],
    binning_types: list[str] | None = None,
) -> OmegaConf:
    """Build all parameters needed for BehaviorSpace construction.

    This is a convenience helper that takes separate lists and constructs
    the dicts needed for BehaviorSpace.

    Args:
        keys: List of behavior feature keys (e.g., ['fitness', 'is_valid'])
        bounds: List of (min, max) bounds tuples (e.g., [(0, 1), (0, 1)])
        resolutions: List of resolution integers (e.g., [150, 2])
        binning_types: Optional list of binning type strings (e.g., ['linear', 'linear'])

    Returns:
        OmegaConf DictConfig with structure matching BehaviorSpace.bins
    """
    if binning_types is None:
        binning_types = ["linear"] * len(keys)

    bins = {}
    for i, key in enumerate(keys):
        min_val, max_val = bounds[i]
        num_bins = resolutions[i]
        b_type = binning_types[i]

        # Construct a dict representation of the strategy
        strategy_conf = {
            "type": b_type,
            "min_val": min_val,
            "max_val": max_val,
            "num_bins": num_bins,
        }
        bins[key] = strategy_conf

    return OmegaConf.create({"bins": bins})


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


def add_auxiliary_metrics(
    metrics_context: MetricsContext, auxiliary_metrics: dict[str, MetricSpec]
) -> MetricsContext:
    """Add auxiliary metrics to existing context in-place.

    Args:
        metrics_context: The MetricsContext to mutate
        auxiliary_metrics: Dict mapping metric keys to MetricSpec instances

    Returns:
        The mutated metrics_context (for chaining in Hydra)
    """
    for key, spec in auxiliary_metrics.items():
        metrics_context.add_metric(key, spec)
    return metrics_context
