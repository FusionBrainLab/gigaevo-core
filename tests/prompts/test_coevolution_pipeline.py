"""Tests for gigaevo.prompts.coevolution.pipeline — PromptEvolutionPipelineBuilder."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.constants import DEFAULT_DAG_CONCURRENCY
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.stages.cache_handler import InputHashCache
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.prompts.coevolution.pipeline import PromptEvolutionPipelineBuilder
from gigaevo.prompts.coevolution.stages import (
    FitnessMetricsInput,
    PromptExecutionStage,
    PromptFitnessStage,
    PromptInsightsStage,
    PromptLineageStage,
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


# ===================================================================
# Amendment #11: PromptInsightsStage / PromptLineageStage
# ===================================================================


class TestPromptInsightsStageInputModel:
    """Verify FitnessMetricsInput schema and cache key behavior."""

    def test_input_model_is_fitness_metrics_input(self):
        """PromptInsightsStage uses FitnessMetricsInput, not VoidInput."""
        assert PromptInsightsStage.InputsModel is FitnessMetricsInput

    def test_input_model_is_fitness_metrics_input_lineage(self):
        """PromptLineageStage uses FitnessMetricsInput, not VoidInput."""
        assert PromptLineageStage.InputsModel is FitnessMetricsInput

    def test_fitness_metrics_is_optional(self):
        """fitness_metrics field is optional (defaults to None)."""
        assert "fitness_metrics" in PromptInsightsStage.optional_fields()
        assert "fitness_metrics" not in PromptInsightsStage.required_fields()

    def test_fitness_metrics_is_optional_lineage(self):
        assert "fitness_metrics" in PromptLineageStage.optional_fields()
        assert "fitness_metrics" not in PromptLineageStage.required_fields()

    def test_uses_input_hash_cache(self):
        """PromptInsightsStage uses InputHashCache (default), not NO_CACHE."""
        assert isinstance(PromptInsightsStage.cache_handler, InputHashCache)

    def test_uses_input_hash_cache_lineage(self):
        assert isinstance(PromptLineageStage.cache_handler, InputHashCache)


class TestFitnessMetricsCacheKey:
    """Verify that different fitness values produce different cache keys."""

    def test_different_trials_different_hash(self):
        """Cache key changes when trials changes."""
        metrics_0 = FloatDictContainer(data={"fitness": 0.25, "trials": 0.0})
        metrics_5 = FloatDictContainer(data={"fitness": 0.40, "trials": 5.0})

        input_0 = FitnessMetricsInput(fitness_metrics=metrics_0)
        input_5 = FitnessMetricsInput(fitness_metrics=metrics_5)

        assert input_0.content_hash != input_5.content_hash

    def test_same_trials_same_hash(self):
        """Cache key is stable when metrics don't change."""
        metrics_a = FloatDictContainer(data={"fitness": 0.40, "trials": 5.0})
        metrics_b = FloatDictContainer(data={"fitness": 0.40, "trials": 5.0})

        input_a = FitnessMetricsInput(fitness_metrics=metrics_a)
        input_b = FitnessMetricsInput(fitness_metrics=metrics_b)

        assert input_a.content_hash == input_b.content_hash

    def test_none_metrics_stable_hash(self):
        """Cache key is stable when fitness_metrics is None."""
        input_a = FitnessMetricsInput(fitness_metrics=None)
        input_b = FitnessMetricsInput(fitness_metrics=None)

        assert input_a.content_hash == input_b.content_hash

    def test_none_vs_zero_trials_different_hash(self):
        """None (no edge provided) differs from trials=0 metrics."""
        metrics_0 = FloatDictContainer(data={"fitness": 0.25, "trials": 0.0})
        input_none = FitnessMetricsInput(fitness_metrics=None)
        input_0 = FitnessMetricsInput(fitness_metrics=metrics_0)

        assert input_none.content_hash != input_0.content_hash


class TestPromptInsightsStageSkipLogic:
    """Test that PromptInsightsStage skips when trials=0."""

    def _make_stage(self):
        """Create a PromptInsightsStage with mocked LLM."""
        llm = MagicMock()
        metrics_ctx = MetricsContext(
            specs={
                "fitness": MetricSpec(
                    description="fitness",
                    is_primary=True,
                    higher_is_better=True,
                ),
            }
        )
        stage = PromptInsightsStage(
            llm=llm,
            task_description="test task",
            metrics_context=metrics_ctx,
            timeout=10.0,
        )
        return stage

    def test_skips_when_trials_zero(self):
        """Stage skips when trials=0 (Beta prior fitness)."""
        stage = self._make_stage()
        metrics = FloatDictContainer(data={"fitness": 0.25, "trials": 0.0})
        stage.attach_inputs({"fitness_metrics": metrics})

        prog = _make_program("def entrypoint(): return 'test'")
        result = _run(stage.compute(prog))

        assert isinstance(result, ProgramStageResult)
        assert result.status == StageState.SKIPPED
        assert "Beta prior" in result.error.message

    def test_runs_when_trials_positive(self):
        """Stage delegates to parent when trials > 0."""
        stage = self._make_stage()
        metrics = FloatDictContainer(data={"fitness": 0.40, "trials": 5.0})
        stage.attach_inputs({"fitness_metrics": metrics})

        prog = _make_program("def entrypoint(): return 'test'")
        # Mock the parent's compute to avoid needing a real LLM
        with patch.object(
            stage.__class__.__bases__[0],
            "compute",
            new_callable=AsyncMock,
        ) as mock_compute:
            from gigaevo.llm.agents.insights import ProgramInsights
            from gigaevo.programs.stages.insights import InsightsOutput

            mock_insights = ProgramInsights(insights=[])
            mock_compute.return_value = InsightsOutput(insights=mock_insights)

            result = _run(stage.compute(prog))
            assert isinstance(result, InsightsOutput)
            mock_compute.assert_called_once()

    def test_skips_when_trials_float_zero(self):
        """Handles trials as float 0.0 (as stored in metrics dict)."""
        stage = self._make_stage()
        metrics = FloatDictContainer(
            data={"fitness": 0.25, "trials": 0.0, "successes": 0.0}
        )
        stage.attach_inputs({"fitness_metrics": metrics})

        prog = _make_program("def entrypoint(): return 'test'")
        result = _run(stage.compute(prog))

        assert isinstance(result, ProgramStageResult)
        assert result.status == StageState.SKIPPED

    def test_runs_when_no_metrics_provided(self):
        """Stage runs normally when fitness_metrics is None (no DAG edge)."""
        stage = self._make_stage()
        stage.attach_inputs({"fitness_metrics": None})

        prog = _make_program("def entrypoint(): return 'test'")
        with patch.object(
            stage.__class__.__bases__[0],
            "compute",
            new_callable=AsyncMock,
        ) as mock_compute:
            from gigaevo.llm.agents.insights import ProgramInsights
            from gigaevo.programs.stages.insights import InsightsOutput

            mock_insights = ProgramInsights(insights=[])
            mock_compute.return_value = InsightsOutput(insights=mock_insights)

            result = _run(stage.compute(prog))
            assert isinstance(result, InsightsOutput)


class TestPromptLineageStageSkipLogic:
    """Test that PromptLineageStage skips when trials=0."""

    def _make_stage(self):
        llm = MagicMock()
        storage = MagicMock(spec=ProgramStorage)
        metrics_ctx = MetricsContext(
            specs={
                "fitness": MetricSpec(
                    description="fitness",
                    is_primary=True,
                    higher_is_better=True,
                ),
            }
        )
        stage = PromptLineageStage(
            llm=llm,
            task_description="test task",
            metrics_context=metrics_ctx,
            storage=storage,
            timeout=10.0,
        )
        return stage

    def test_skips_when_trials_zero(self):
        """Stage skips when trials=0 (Beta prior fitness)."""
        stage = self._make_stage()
        metrics = FloatDictContainer(data={"fitness": 0.25, "trials": 0.0})
        stage.attach_inputs({"fitness_metrics": metrics})

        prog = _make_program("def entrypoint(): return 'test'")
        result = _run(stage.compute(prog))

        assert isinstance(result, ProgramStageResult)
        assert result.status == StageState.SKIPPED
        assert "Beta prior" in result.error.message

    def test_runs_when_trials_positive(self):
        """Stage delegates to parent when trials > 0."""
        stage = self._make_stage()
        metrics = FloatDictContainer(data={"fitness": 0.60, "trials": 10.0})
        stage.attach_inputs({"fitness_metrics": metrics})

        prog = _make_program("def entrypoint(): return 'test'")
        prog.lineage.parents = []

        with patch.object(
            stage.__class__.__bases__[0],
            "compute",
            new_callable=AsyncMock,
        ) as mock_compute:
            from gigaevo.programs.stages.insights_lineage import LineageAnalysesOutput

            mock_compute.return_value = LineageAnalysesOutput(analyses=[])

            result = _run(stage.compute(prog))
            assert isinstance(result, LineageAnalysesOutput)
            mock_compute.assert_called_once()


class TestPipelineDataFlowEdges:
    """Verify the fitness_metrics DataFlowEdges are correctly wired."""

    def test_insights_gets_fitness_metrics_edge(self):
        """InsightsStage receives fitness_metrics from EnsureMetricsStage."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        fitness_edges = [
            e
            for e in bp.data_flow_edges
            if e.destination_stage == "InsightsStage"
            and e.input_name == "fitness_metrics"
        ]
        assert len(fitness_edges) == 1
        assert fitness_edges[0].source_stage == "EnsureMetricsStage"

    def test_lineage_gets_fitness_metrics_edge(self):
        """LineageStage receives fitness_metrics from EnsureMetricsStage."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        fitness_edges = [
            e
            for e in bp.data_flow_edges
            if e.destination_stage == "LineageStage"
            and e.input_name == "fitness_metrics"
        ]
        assert len(fitness_edges) == 1
        assert fitness_edges[0].source_stage == "EnsureMetricsStage"

    def test_insights_node_is_prompt_subclass(self):
        """InsightsStage node factory creates PromptInsightsStage."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        stage = bp.nodes["InsightsStage"]()
        assert isinstance(stage, PromptInsightsStage)

    def test_lineage_node_is_prompt_subclass(self):
        """LineageStage node factory creates PromptLineageStage."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        stage = bp.nodes["LineageStage"]()
        assert isinstance(stage, PromptLineageStage)

    def test_no_exec_order_dep_for_insights_on_metrics(self):
        """InsightsStage should NOT have explicit exec_order_dep on EnsureMetricsStage
        (ordering is now implicit via DataFlowEdge)."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "InsightsStage" not in (bp.exec_order_deps or {})

    def test_no_exec_order_dep_for_lineage_on_metrics(self):
        """LineageStage should NOT have explicit exec_order_dep on EnsureMetricsStage."""
        ctx = _make_ctx()
        stats = MagicMock(spec=PromptStatsProvider)
        bp = PromptEvolutionPipelineBuilder(ctx, stats).build_blueprint()

        assert "LineageStage" not in (bp.exec_order_deps or {})
