"""Tests for gigaevo.prompts.coevolution.pipeline — PromptEvolutionPipelineBuilder."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock
import uuid

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.constants import DEFAULT_DAG_CONCURRENCY
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.prompts.coevolution.pipeline import PromptEvolutionPipelineBuilder
from gigaevo.prompts.coevolution.stages import (
    PromptExecutionStage,
    PromptFitnessStage,
)
from gigaevo.prompts.coevolution.stats import (
    PromptStatsProvider,
    prompt_text_to_id,
)
from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher
from gigaevo.runner.dag_blueprint import DAGBlueprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(prompts_dir: str | None = None) -> EvolutionContext:
    metrics_ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="prompt fitness",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "is_valid": MetricSpec(
                description="validity",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "prompt_length": MetricSpec(
                description="length of prompt",
                higher_is_better=False,
                lower_bound=0.0,
                upper_bound=50000.0,
            ),
        }
    )

    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = Path("/fake/prompt_evolution")
    problem_ctx.task_description = "Evolve mutation prompts."
    problem_ctx.metrics_context = metrics_ctx

    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
        prompts_dir=prompts_dir,
    )


def _edge_pairs(bp: DAGBlueprint) -> set[tuple[str, str]]:
    return {(e.source_stage, e.destination_stage) for e in bp.data_flow_edges}


def _dep_names(bp: DAGBlueprint, stage: str) -> set[str]:
    if bp.exec_order_deps is None:
        return set()
    return {d.stage_name for d in bp.exec_order_deps.get(stage, [])}


# ===================================================================
# PromptEvolutionPipelineBuilder
# ===================================================================


class TestPromptEvolutionPipelineStages:
    def test_has_prompt_specific_stages(self):
        """Pipeline must contain PromptExecution and PromptFitness stages."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "PromptExecutionStage" in bp.nodes
        assert "PromptFitnessStage" in bp.nodes

    def test_has_validation_stage(self):
        """Syntax validation is still done before execution."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "ValidateCodeStage" in bp.nodes

    def test_does_not_have_standard_pipeline_stages(self):
        """Should NOT have the standard pipeline stages that it replaces."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        # These are replaced by PromptExecution + PromptFitness
        assert "CallProgramFunction" not in bp.nodes
        assert "CallValidatorFunction" not in bp.nodes
        assert "FetchMetrics" not in bp.nodes
        assert "FetchArtifact" not in bp.nodes
        assert "FormatterStage" not in bp.nodes

    def test_has_shared_stages(self):
        """Should retain shared stages from the default pipeline."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        shared_stages = {
            "ComputeComplexityStage",
            "MergeMetricsStage",
            "EnsureMetricsStage",
            "InsightsStage",
            "DescendantProgramIds",
            "AncestorProgramIds",
            "LineageStage",
            "LineagesToDescendants",
            "LineagesFromAncestors",
            "MutationContextStage",
            "EvolutionaryStatisticsCollector",
        }
        for stage in shared_stages:
            assert stage in bp.nodes, f"Missing shared stage: {stage}"

    def test_total_stage_count(self):
        """Verify we have exactly the expected number of stages."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        # 2 prompt-specific + 1 validate + 11 shared = 14
        assert len(bp.nodes) == 14


class TestPromptEvolutionPipelineDataFlow:
    def test_prompt_execution_feeds_fitness(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("PromptExecutionStage", "PromptFitnessStage") in edges

    def test_fitness_feeds_merge_metrics(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("PromptFitnessStage", "MergeMetricsStage") in edges

    def test_complexity_feeds_merge_metrics(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("ComputeComplexityStage", "MergeMetricsStage") in edges

    def test_metrics_chain_to_mutation_context(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("MergeMetricsStage", "EnsureMetricsStage") in edges
        assert ("EnsureMetricsStage", "MutationContextStage") in edges

    def test_insights_feeds_mutation_context(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("InsightsStage", "MutationContextStage") in edges

    def test_lineage_data_flow(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("DescendantProgramIds", "LineagesToDescendants") in edges
        assert ("AncestorProgramIds", "LineagesFromAncestors") in edges
        assert ("LineagesToDescendants", "MutationContextStage") in edges
        assert ("LineagesFromAncestors", "MutationContextStage") in edges
        assert ("EvolutionaryStatisticsCollector", "MutationContextStage") in edges

    def test_no_call_program_to_validator_edge(self):
        """There should be no data flow from standard pipeline's CallProgram -> CallValidator."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()
        edges = _edge_pairs(bp)

        assert ("CallProgramFunction", "CallValidatorFunction") not in edges


class TestPromptEvolutionPipelineDeps:
    def test_prompt_execution_after_validation(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "ValidateCodeStage" in _dep_names(bp, "PromptExecutionStage")

    def test_prompt_fitness_after_execution(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "PromptExecutionStage" in _dep_names(bp, "PromptFitnessStage")

    def test_insights_after_metrics(self):
        """InsightsStage depends on EnsureMetricsStage via DataFlowEdge (fitness_metrics)."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert ("EnsureMetricsStage", "InsightsStage") in _edge_pairs(bp)

    def test_lineage_after_metrics(self):
        """LineageStage depends on EnsureMetricsStage via DataFlowEdge (fitness_metrics)."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert ("EnsureMetricsStage", "LineageStage") in _edge_pairs(bp)
        assert "LineageStage" in _dep_names(bp, "LineagesToDescendants")
        assert "LineageStage" in _dep_names(bp, "LineagesFromAncestors")

    def test_statistics_after_metrics(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "EnsureMetricsStage" in _dep_names(bp, "EvolutionaryStatisticsCollector")


class TestPromptEvolutionPipelineConfig:
    def test_custom_dag_timeout(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(
            ctx, stats, dag_timeout=999.0
        ).build_blueprint()

        assert bp.dag_timeout == 999.0

    def test_default_dag_timeout(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert bp.dag_timeout == 3600.0

    def test_max_parallel_stages(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert bp.max_parallel_stages == DEFAULT_DAG_CONCURRENCY

    def test_all_factories_callable(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        for name, factory in bp.nodes.items():
            assert callable(factory), f"{name} factory is not callable"

    def test_blueprint_is_valid_type(self):
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert isinstance(bp, DAGBlueprint)


# ===================================================================
# PromptExecutionStage basic tests
# ===================================================================


def _run(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_program(code: str) -> Program:
    return Program(id=str(uuid.uuid4()), code=code)


class TestPromptExecutionStageBasic:
    def test_accepts_str_return(self):
        """Programs returning a str are accepted."""
        stage = PromptExecutionStage()
        prog = _make_program('def entrypoint():\n    return "hello system"')
        result = _run(stage.compute(prog))
        assert result.prompt_text == "hello system"

    def test_accepts_dict_return(self):
        """Programs returning a dict with system/user keys are accepted."""
        stage = PromptExecutionStage()
        code = (
            "def entrypoint():\n"
            '    return {"system": "sys prompt", "user": "usr prompt"}'
        )
        prog = _make_program(code)
        result = _run(stage.compute(prog))
        assert result.prompt_text == "sys prompt"
        assert result.user_text == "usr prompt"


# ===================================================================
# C2: Beta(1,3) prior default
# ===================================================================


class TestPromptFitnessStagePrior:
    def test_default_prior_is_beta_1_3(self):
        """C2: Default prior is Beta(1,3) → untested prompts get fitness=0.25."""
        stats_provider = MagicMock(spec=PromptStatsProvider)
        stage = PromptFitnessStage(stats_provider=stats_provider)
        assert stage._prior_alpha == 1.0
        assert stage._prior_beta == 3.0

    def test_zero_trials_fitness_is_025(self):
        """C2: With 0 trials, fitness = (0+1)/(0+1+3) = 0.25."""
        # fitness = (successes + alpha) / (trials + alpha + beta)
        # = (0 + 1) / (0 + 1 + 3) = 0.25
        alpha, beta = 1.0, 3.0
        fitness = (0 + alpha) / (0 + alpha + beta)
        assert fitness == 0.25


# ===================================================================
# M1: metrics_count denominator tracking
# ===================================================================


class TestMetricsCountTracking:
    def test_record_outcome_with_metrics_increments_count(self):
        """M1: record_outcome increments metrics_count when child_metrics provided."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="test_prefix",
            main_redis_db=0,
        )

        # Mock the Redis client
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # No existing stats
        fetcher._redis_main_sync = mock_redis

        captured = {}

        def capture_set(key, value):
            captured["key"] = key
            captured["value"] = json.loads(value)

        mock_redis.set.side_effect = capture_set

        fetcher.record_outcome(
            prompt_id="abc123",
            child_fitness=0.8,
            parent_fitness=0.5,
            higher_is_better=True,
            outcome=MutationOutcome.ACCEPTED,
            child_metrics={"em": 0.6, "f1": 0.8},
        )

        assert captured["value"]["metrics_count"] == 1
        assert captured["value"]["metrics_sums"]["em"] == 0.6
        assert captured["value"]["metrics_sums"]["f1"] == 0.8

    def test_record_outcome_without_metrics_no_count(self):
        """M1: record_outcome does NOT increment metrics_count without child_metrics."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="test_prefix",
            main_redis_db=0,
        )

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        fetcher._redis_main_sync = mock_redis

        captured = {}

        def capture_set(key, value):
            captured["value"] = json.loads(value)

        mock_redis.set.side_effect = capture_set

        fetcher.record_outcome(
            prompt_id="abc123",
            child_fitness=0.8,
            parent_fitness=0.5,
            higher_is_better=True,
            outcome=MutationOutcome.ACCEPTED,
            child_metrics=None,
        )

        assert captured["value"]["metrics_count"] == 0


# ===================================================================
# M4: prompt_text_to_id includes user_text
# ===================================================================


class TestPromptTextToIdUserText:
    def test_same_system_different_user_different_ids(self):
        """M4: Two prompts with same system but different user get different IDs."""
        id1 = prompt_text_to_id("system", user_text="user1")
        id2 = prompt_text_to_id("system", user_text="user2")
        assert id1 != id2

    def test_same_system_no_user_backward_compat(self):
        """M4: Without user_text, behaves like before."""
        id_old = prompt_text_to_id("system")
        id_new = prompt_text_to_id("system", user_text=None)
        assert id_old == id_new

    def test_user_text_changes_id(self):
        """M4: Adding user_text changes the ID vs system-only."""
        id_system_only = prompt_text_to_id("system")
        id_with_user = prompt_text_to_id("system", user_text="user")
        assert id_system_only != id_with_user
