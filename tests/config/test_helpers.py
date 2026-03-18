"""Tests for gigaevo.config.helpers configuration utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from gigaevo.config.helpers import (
    add_auxiliary_metrics,
    build_behavior_space,
    build_behavior_space_params,
    build_dag_from_builder,
    extract_behavior_keys_from_islands,
    get_bounds,
    get_metrics_context,
    get_primary_key,
    is_higher_better,
    select_pipeline_builder,
)
from gigaevo.database.program_storage import ProgramStorage
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
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics_context() -> MetricsContext:
    """Create a test MetricsContext with common metrics."""
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="main metric",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "complexity": MetricSpec(
                description="code complexity",
                higher_is_better=False,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
            "is_valid": MetricSpec(
                description="validity flag",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _make_problem_context(*, is_contextual: bool = False) -> ProblemContext:
    """Create a mock ProblemContext."""
    ctx = MagicMock(spec=ProblemContext)
    ctx.problem_dir = Path("/fake/problem")
    ctx.metrics_context = _make_metrics_context()
    type(ctx).is_contextual = PropertyMock(return_value=is_contextual)
    return ctx


def _make_evolution_context() -> EvolutionContext:
    """Create a mock EvolutionContext."""
    problem_ctx = _make_problem_context()
    storage = MagicMock(spec=ProgramStorage)
    llm_wrapper = MagicMock(spec=MultiModelRouter)

    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=llm_wrapper,
        storage=storage,
    )


# ===================================================================
# Simple accessor functions
# ===================================================================


class TestSimpleAccessors:
    def test_get_metrics_context(self):
        problem_ctx = _make_problem_context()
        mc = get_metrics_context(problem_ctx)
        assert mc is problem_ctx.metrics_context
        assert "fitness" in mc.specs

    def test_get_primary_key(self):
        mc = _make_metrics_context()
        primary = get_primary_key(mc)
        assert primary == "fitness"

    def test_is_higher_better_true(self):
        mc = _make_metrics_context()
        assert is_higher_better(mc, "fitness") is True

    def test_is_higher_better_false(self):
        mc = _make_metrics_context()
        assert is_higher_better(mc, "complexity") is False

    def test_get_bounds(self):
        mc = _make_metrics_context()
        bounds = get_bounds(mc, "fitness")
        assert bounds == (0.0, 1.0)

    def test_get_bounds_different_metric(self):
        mc = _make_metrics_context()
        bounds = get_bounds(mc, "complexity")
        assert bounds == (0.0, 100.0)


# ===================================================================
# build_behavior_space
# ===================================================================


class TestBuildBehaviorSpace:
    def test_basic_static_behavior_space(self):
        space = build_behavior_space(
            keys=["fitness", "complexity"],
            bounds=[(0.0, 1.0), (0.0, 100.0)],
            resolutions=[150, 10],
            binning_types=["linear", "linear"],
            dynamic=False,
        )

        assert isinstance(space, BehaviorSpace)
        assert isinstance(space, DynamicBehaviorSpace) is False
        assert space.behavior_keys == ["fitness", "complexity"]

    def test_basic_dynamic_behavior_space(self):
        space = build_behavior_space(
            keys=["fitness", "complexity"],
            bounds=[(0.0, 1.0), (0.0, 100.0)],
            resolutions=[150, 10],
            binning_types=["linear", "linear"],
            dynamic=True,
        )

        assert isinstance(space, DynamicBehaviorSpace)
        assert space.behavior_keys == ["fitness", "complexity"]

    def test_single_feature_space(self):
        space = build_behavior_space(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
            dynamic=False,
        )

        assert space.behavior_keys == ["fitness"]

    def test_dynamic_space_with_custom_buffer_ratio(self):
        space = build_behavior_space(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
            dynamic=True,
            expansion_buffer_ratio=0.2,
        )

        assert isinstance(space, DynamicBehaviorSpace)
        assert space.expansion_buffer_ratio == 0.2

    def test_mismatched_keys_and_bounds_raises(self):
        with pytest.raises(
            ValueError, match="All parameter lists must have the same length"
        ):
            build_behavior_space(
                keys=["fitness", "complexity"],
                bounds=[(0.0, 1.0)],  # Only 1 bound for 2 keys
                resolutions=[100, 10],
                binning_types=["linear", "linear"],
            )

    def test_mismatched_keys_and_resolutions_raises(self):
        with pytest.raises(
            ValueError, match="All parameter lists must have the same length"
        ):
            build_behavior_space(
                keys=["fitness"],
                bounds=[(0.0, 1.0)],
                resolutions=[100, 10],  # 2 resolutions for 1 key
                binning_types=["linear"],
            )

    def test_mismatched_keys_and_binning_types_raises(self):
        with pytest.raises(
            ValueError, match="All parameter lists must have the same length"
        ):
            build_behavior_space(
                keys=["fitness"],
                bounds=[(0.0, 1.0)],
                resolutions=[100],
                binning_types=["linear", "linear"],  # 2 types for 1 key
            )

    def test_behavior_space_bins_are_linear_binning(self):
        space = build_behavior_space(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
            dynamic=False,
        )

        assert isinstance(space.bins["fitness"], LinearBinning)
        assert space.bins["fitness"].min_val == 0.0
        assert space.bins["fitness"].max_val == 1.0
        assert space.bins["fitness"].num_bins == 100


# ===================================================================
# build_behavior_space_params
# ===================================================================


class TestBuildBehaviorSpaceParams:
    def test_basic_params(self):
        from omegaconf import DictConfig

        params = build_behavior_space_params(
            keys=["fitness", "complexity"],
            bounds=[(0.0, 1.0), (0.0, 100.0)],
            resolutions=[150, 10],
        )

        assert isinstance(params, DictConfig)
        assert "bins" in params
        assert "fitness" in params.bins
        assert "complexity" in params.bins

    def test_params_default_binning_linear(self):
        params = build_behavior_space_params(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
        )

        assert params.bins.fitness.type == "linear"

    def test_params_custom_binning_type(self):
        params = build_behavior_space_params(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
        )

        assert params.bins.fitness.type == "linear"
        assert params.bins.fitness.min_val == 0.0
        assert params.bins.fitness.max_val == 1.0
        assert params.bins.fitness.num_bins == 100


# ===================================================================
# extract_behavior_keys_from_islands
# ===================================================================


class TestExtractBehaviorKeysFromIslands:
    def test_single_island_single_key(self):
        space = build_behavior_space(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
        )
        island = MagicMock(spec=IslandConfig)
        island.behavior_space = space

        keys = extract_behavior_keys_from_islands([island])
        assert keys == {"fitness"}

    def test_single_island_multiple_keys(self):
        space = build_behavior_space(
            keys=["fitness", "complexity"],
            bounds=[(0.0, 1.0), (0.0, 100.0)],
            resolutions=[150, 10],
            binning_types=["linear", "linear"],
        )
        island = MagicMock(spec=IslandConfig)
        island.behavior_space = space

        keys = extract_behavior_keys_from_islands([island])
        assert keys == {"fitness", "complexity"}

    def test_multiple_islands_union_of_keys(self):
        space1 = build_behavior_space(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
        )
        space2 = build_behavior_space(
            keys=["complexity"],
            bounds=[(0.0, 100.0)],
            resolutions=[10],
            binning_types=["linear"],
        )

        island1 = MagicMock(spec=IslandConfig)
        island1.behavior_space = space1
        island2 = MagicMock(spec=IslandConfig)
        island2.behavior_space = space2

        keys = extract_behavior_keys_from_islands([island1, island2])
        assert keys == {"fitness", "complexity"}

    def test_empty_islands_list(self):
        keys = extract_behavior_keys_from_islands([])
        assert keys == set()


# ===================================================================
# build_dag_from_builder
# ===================================================================


class TestBuildDagFromBuilder:
    def test_builds_blueprint_from_default_builder(self):
        evo_ctx = _make_evolution_context()
        builder = DefaultPipelineBuilder(evo_ctx)
        blueprint = build_dag_from_builder(builder)

        assert blueprint is not None
        assert len(blueprint.nodes) > 0

    def test_builds_blueprint_from_context_builder(self):
        evo_ctx = _make_evolution_context()
        builder = ContextPipelineBuilder(evo_ctx)
        blueprint = build_dag_from_builder(builder)

        assert blueprint is not None
        assert "AddContext" in blueprint.nodes


# ===================================================================
# select_pipeline_builder
# ===================================================================


class TestSelectPipelineBuilder:
    def test_selects_default_for_non_contextual_problem(self):
        problem_ctx = _make_problem_context(is_contextual=False)
        evo_ctx = _make_evolution_context()
        evo_ctx.problem_ctx = problem_ctx

        builder = select_pipeline_builder(problem_ctx, evo_ctx)
        assert isinstance(builder, DefaultPipelineBuilder)
        assert not isinstance(builder, ContextPipelineBuilder)

    def test_selects_context_for_contextual_problem(self):
        problem_ctx = _make_problem_context(is_contextual=True)
        evo_ctx = _make_evolution_context()
        evo_ctx.problem_ctx = problem_ctx

        builder = select_pipeline_builder(problem_ctx, evo_ctx)
        assert isinstance(builder, ContextPipelineBuilder)


# ===================================================================
# add_auxiliary_metrics
# ===================================================================


class TestAddAuxiliaryMetrics:
    def test_adds_single_metric(self):
        mc = _make_metrics_context()
        original_count = len(mc.specs)

        new_spec = MetricSpec(
            description="new metric",
            higher_is_better=True,
            lower_bound=0.0,
            upper_bound=10.0,
        )
        result = add_auxiliary_metrics(mc, {"new_metric": new_spec})

        assert result is mc
        assert "new_metric" in mc.specs
        assert len(mc.specs) == original_count + 1

    def test_adds_multiple_metrics(self):
        mc = _make_metrics_context()
        original_count = len(mc.specs)

        specs = {
            "metric1": MetricSpec(
                description="m1",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "metric2": MetricSpec(
                description="m2",
                higher_is_better=False,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
        }
        result = add_auxiliary_metrics(mc, specs)

        assert result is mc
        assert "metric1" in mc.specs
        assert "metric2" in mc.specs
        assert len(mc.specs) == original_count + 2

    def test_returns_same_context_for_chaining(self):
        mc = _make_metrics_context()
        new_spec = MetricSpec(
            description="test",
            higher_is_better=True,
            lower_bound=0.0,
            upper_bound=1.0,
        )

        result = add_auxiliary_metrics(mc, {"test_metric": new_spec})
        assert result is mc

    def test_empty_auxiliary_metrics_noop(self):
        mc = _make_metrics_context()
        original_count = len(mc.specs)

        result = add_auxiliary_metrics(mc, {})
        assert result is mc
        assert len(mc.specs) == original_count
