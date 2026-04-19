"""Tests for gigaevo.entrypoint.default_pipelines pipeline builders."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.constants import (
    DEFAULT_DAG_CONCURRENCY,
    DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION,
)
from gigaevo.entrypoint.default_pipelines import (
    AlgoTuneSpeedPipelineBuilder,
    CMAOptPipelineBuilder,
    ContextPipelineBuilder,
    CustomPipelineBuilder,
    DefaultPipelineBuilder,
    OptunaOptPipelineBuilder,
    PipelineBuilder,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.runner.dag_blueprint import DAGBlueprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics_context(*, primary_key: str = "fitness") -> MetricsContext:
    return MetricsContext(
        specs={
            primary_key: MetricSpec(
                description="main metric",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "is_valid": MetricSpec(
                description="validity flag",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _make_ctx(
    *,
    is_contextual: bool = False,
    problem_dir: Path | None = None,
    prompts_dir: str | None = None,
    primary_key: str = "fitness",
) -> EvolutionContext:
    """Build a mock EvolutionContext suitable for pipeline builder tests."""
    p_dir = problem_dir or Path("/fake/problem")

    metrics_ctx = _make_metrics_context(primary_key=primary_key)

    # Use spec= to make mocks pass isinstance checks
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = p_dir
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = is_contextual

    storage = MagicMock(spec=ProgramStorage)
    llm_wrapper = MagicMock(spec=MultiModelRouter)

    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=llm_wrapper,
        storage=storage,
        prompts_dir=prompts_dir,
    )


def _edge_pairs(blueprint: DAGBlueprint) -> set[tuple[str, str]]:
    """Extract (source, destination) pairs from a blueprint's data flow edges."""
    return {(e.source_stage, e.destination_stage) for e in blueprint.data_flow_edges}


def _dep_names(blueprint: DAGBlueprint, stage: str) -> set[str]:
    """Get the set of dependency stage names for a given stage."""
    if blueprint.exec_order_deps is None:
        return set()
    return {d.stage_name for d in blueprint.exec_order_deps.get(stage, [])}


# ===================================================================
# PipelineBuilder (base)
# ===================================================================


class TestPipelineBuilder:
    def test_empty_builder_produces_empty_blueprint(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert isinstance(bp, DAGBlueprint)
        assert len(bp.nodes) == 0
        assert len(bp.data_flow_edges) == 0
        assert bp.exec_order_deps is None
        assert bp.max_parallel_stages == DEFAULT_DAG_CONCURRENCY

    def test_add_stage(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        factory = MagicMock()
        builder.add_stage("MyStage", factory)
        bp = builder.build_blueprint()

        assert "MyStage" in bp.nodes
        assert bp.nodes["MyStage"] is factory

    def test_replace_stage(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        factory1 = MagicMock(name="factory1")
        factory2 = MagicMock(name="factory2")
        builder.add_stage("S", factory1)
        builder.replace_stage("S", factory2)
        bp = builder.build_blueprint()

        assert bp.nodes["S"] is factory2

    def test_remove_stage_cleans_edges_and_deps(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_stage("A", MagicMock())
        builder.add_stage("B", MagicMock())
        builder.add_stage("C", MagicMock())
        builder.add_data_flow_edge("A", "B", "input1")
        builder.add_data_flow_edge("B", "C", "input2")
        builder.add_exec_dep("C", ExecutionOrderDependency.on_success("B"))

        builder.remove_stage("B")
        bp = builder.build_blueprint()

        assert "B" not in bp.nodes
        assert _edge_pairs(bp) == set()
        assert _dep_names(bp, "C") == set()

    def test_add_and_remove_data_flow_edge(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_stage("X", MagicMock())
        builder.add_stage("Y", MagicMock())
        builder.add_data_flow_edge("X", "Y", "data")

        bp = builder.build_blueprint()
        assert ("X", "Y") in _edge_pairs(bp)

        builder.remove_data_flow_edge("X", "Y")
        bp = builder.build_blueprint()
        assert ("X", "Y") not in _edge_pairs(bp)

    def test_add_and_remove_exec_dep(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        dep = ExecutionOrderDependency.on_success("A")
        builder.add_exec_dep("B", dep)

        bp = builder.build_blueprint()
        assert "A" in _dep_names(bp, "B")

        builder.remove_exec_dep("B", dep)
        bp = builder.build_blueprint()
        assert "A" not in _dep_names(bp, "B")

    def test_set_limits(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.set_limits(dag_timeout=999.0, max_parallel=4)
        bp = builder.build_blueprint()

        assert bp.dag_timeout == 999.0
        assert bp.max_parallel_stages == 4

    def test_set_limits_partial_none(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx, dag_timeout=500.0)
        builder.set_limits(dag_timeout=None, max_parallel=2)
        bp = builder.build_blueprint()

        assert bp.dag_timeout == 500.0
        assert bp.max_parallel_stages == 2

    def test_fluent_chaining(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        result = (
            builder.add_stage("A", MagicMock())
            .add_stage("B", MagicMock())
            .add_data_flow_edge("A", "B", "x")
            .set_limits(dag_timeout=100.0, max_parallel=2)
        )
        assert result is builder


# ===================================================================
# DefaultPipelineBuilder
# ===================================================================


class TestDefaultPipelineBuilder:
    def test_has_expected_core_stages(self):
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        expected_stages = {
            "ValidateCodeStage",
            "CallProgramFunction",
            "CallValidatorFunction",
            "FetchMetrics",
            "FetchArtifact",
            "FormatterStage",
            "InsightsStage",
            "DescendantProgramIds",
            "AncestorProgramIds",
            "LineageStage",
            "LineagesToDescendants",
            "LineagesFromAncestors",
            "MemoryContextStage",
            "MutationContextStage",
            "ComputeComplexityStage",
            "MergeMetricsStage",
            "EnsureMetricsStage",
            "EvolutionaryStatisticsCollector",
        }
        assert set(bp.nodes.keys()) == expected_stages

    def test_does_not_include_context_stage(self):
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        assert "AddContext" not in bp.nodes

    def test_critical_data_flow_edges(self):
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("CallProgramFunction", "CallValidatorFunction") in edges
        assert ("CallValidatorFunction", "FetchMetrics") in edges
        assert ("CallValidatorFunction", "FetchArtifact") in edges
        assert ("FetchMetrics", "MergeMetricsStage") in edges
        assert ("ComputeComplexityStage", "MergeMetricsStage") in edges
        assert ("MergeMetricsStage", "EnsureMetricsStage") in edges
        assert ("EnsureMetricsStage", "MutationContextStage") in edges
        assert ("InsightsStage", "MutationContextStage") in edges

    def test_lineage_data_flow(self):
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("DescendantProgramIds", "LineagesToDescendants") in edges
        assert ("AncestorProgramIds", "LineagesFromAncestors") in edges
        assert ("LineagesToDescendants", "MutationContextStage") in edges
        assert ("LineagesFromAncestors", "MutationContextStage") in edges

    def test_execution_order_deps(self):
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "ValidateCodeStage" in _dep_names(bp, "CallProgramFunction")
        assert "CallValidatorFunction" in _dep_names(bp, "FetchMetrics")
        assert "CallValidatorFunction" in _dep_names(bp, "FetchArtifact")
        assert "FetchArtifact" in _dep_names(bp, "FormatterStage")
        assert "EnsureMetricsStage" in _dep_names(bp, "InsightsStage")
        assert "EnsureMetricsStage" in _dep_names(bp, "LineageStage")
        assert "LineageStage" in _dep_names(bp, "LineagesToDescendants")
        assert "LineageStage" in _dep_names(bp, "LineagesFromAncestors")
        assert "EnsureMetricsStage" in _dep_names(bp, "EvolutionaryStatisticsCollector")

    def test_custom_dag_timeout(self):
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx, dag_timeout=1234.0)
        bp = builder.build_blueprint()
        assert bp.dag_timeout == 1234.0

    def test_all_stages_are_callable_factories(self):
        """Verify all stage nodes are callable factories (not instantiated)."""
        ctx = _make_ctx()
        builder = DefaultPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        for name, factory in bp.nodes.items():
            assert callable(factory), f"{name} factory is not callable"


# ===================================================================
# ContextPipelineBuilder
# ===================================================================


class TestContextPipelineBuilder:
    def test_adds_context_stage(self):
        ctx = _make_ctx()
        builder = ContextPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "AddContext" in bp.nodes

    def test_context_wired_to_program_and_validator(self):
        ctx = _make_ctx()
        builder = ContextPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("AddContext", "CallProgramFunction") in edges
        assert ("AddContext", "CallValidatorFunction") in edges

    def test_inherits_all_default_stages(self):
        ctx = _make_ctx()
        default_bp = DefaultPipelineBuilder(ctx).build_blueprint()
        context_bp = ContextPipelineBuilder(ctx).build_blueprint()

        default_stages = set(default_bp.nodes.keys())
        context_stages = set(context_bp.nodes.keys())
        assert default_stages.issubset(context_stages)
        assert context_stages - default_stages == {"AddContext"}


# ===================================================================
# CMAOptPipelineBuilder
# ===================================================================


class TestCMAOptPipelineBuilder:
    def test_adds_cma_stage_without_context(self):
        ctx = _make_ctx(is_contextual=False)
        builder = CMAOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "CMAOptStage" in bp.nodes
        assert "AddContext" not in bp.nodes

    def test_adds_cma_and_context_when_contextual(self):
        ctx = _make_ctx(is_contextual=True)
        builder = CMAOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "CMAOptStage" in bp.nodes
        assert "AddContext" in bp.nodes

    def test_cma_exec_deps_without_context(self):
        ctx = _make_ctx(is_contextual=False)
        builder = CMAOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "ValidateCodeStage" in _dep_names(bp, "CMAOptStage")
        assert "CMAOptStage" in _dep_names(bp, "CallProgramFunction")

    def test_cma_exec_deps_with_context(self):
        ctx = _make_ctx(is_contextual=True)
        builder = CMAOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert "AddContext" in _dep_names(bp, "CMAOptStage")
        assert ("AddContext", "CMAOptStage") in edges

    def test_context_wired_to_program_and_validator(self):
        ctx = _make_ctx(is_contextual=True)
        builder = CMAOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("AddContext", "CallProgramFunction") in edges
        assert ("AddContext", "CallValidatorFunction") in edges

    def test_custom_optimization_budget(self):
        ctx = _make_ctx()
        builder = CMAOptPipelineBuilder(ctx, optimization_time_budget=100.0)
        bp = builder.build_blueprint()
        assert "CMAOptStage" in bp.nodes

    def test_default_optimization_budget_from_dag_timeout(self):
        ctx = _make_ctx()
        builder = CMAOptPipelineBuilder(ctx, dag_timeout=2000.0)
        expected_budget = 2000.0 * DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION
        assert builder._optimization_time_budget == expected_budget

    def test_cma_stage_factory_is_callable(self):
        ctx = _make_ctx()
        builder = CMAOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        assert callable(bp.nodes["CMAOptStage"])

    def test_inherits_all_default_stages(self):
        ctx = _make_ctx(is_contextual=False)
        default_bp = DefaultPipelineBuilder(ctx).build_blueprint()
        cma_bp = CMAOptPipelineBuilder(ctx).build_blueprint()

        default_stages = set(default_bp.nodes.keys())
        assert default_stages.issubset(set(cma_bp.nodes.keys()))


# ===================================================================
# OptunaOptPipelineBuilder
# ===================================================================


class TestOptunaOptPipelineBuilder:
    def test_adds_optuna_stages_without_context(self):
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "OptunaOptStage" in bp.nodes
        assert "OptunaPayloadBridge" in bp.nodes
        assert "PayloadResolver" in bp.nodes
        assert "AddContext" not in bp.nodes

    def test_adds_optuna_and_context_when_contextual(self):
        ctx = _make_ctx(is_contextual=True)
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "OptunaOptStage" in bp.nodes
        assert "AddContext" in bp.nodes

    def test_optuna_exec_deps(self):
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert "ValidateCodeStage" in _dep_names(bp, "OptunaOptStage")
        # CallProgramFunction runs on Optuna failure (fallback)
        assert "OptunaOptStage" in _dep_names(bp, "CallProgramFunction")

    def test_optuna_bypass_data_flow(self):
        """OptunaPayloadBridge extracts output from Optuna;
        PayloadResolver picks between Optuna and direct program execution."""
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("OptunaOptStage", "OptunaPayloadBridge") in edges
        assert ("OptunaPayloadBridge", "PayloadResolver") in edges
        assert ("CallProgramFunction", "PayloadResolver") in edges
        assert ("PayloadResolver", "CallValidatorFunction") in edges

    def test_default_edge_replaced(self):
        """The direct CallProgramFunction -> CallValidatorFunction edge
        should be replaced by the PayloadResolver path."""
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("CallProgramFunction", "CallValidatorFunction") not in edges

    def test_optuna_with_context_wiring(self):
        ctx = _make_ctx(is_contextual=True)
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        edges = _edge_pairs(bp)

        assert ("AddContext", "OptunaOptStage") in edges
        assert "AddContext" in _dep_names(bp, "OptunaOptStage")

    def test_custom_optimization_budget(self):
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx, optimization_time_budget=200.0)
        assert builder._optimization_time_budget == 200.0

    def test_default_dag_timeout_is_7200(self):
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        assert bp.dag_timeout == 7200.0

    def test_optuna_stage_factory_is_callable(self):
        ctx = _make_ctx()
        builder = OptunaOptPipelineBuilder(ctx)
        bp = builder.build_blueprint()
        assert callable(bp.nodes["OptunaOptStage"])

    def test_inherits_all_default_stages(self):
        ctx = _make_ctx()
        default_bp = DefaultPipelineBuilder(ctx).build_blueprint()
        optuna_bp = OptunaOptPipelineBuilder(ctx).build_blueprint()

        default_stages = set(default_bp.nodes.keys())
        assert default_stages.issubset(set(optuna_bp.nodes.keys()))


# ===================================================================
# CustomPipelineBuilder
# ===================================================================


class TestCustomPipelineBuilder:
    def test_starts_empty(self):
        ctx = _make_ctx()
        builder = CustomPipelineBuilder(ctx)
        bp = builder.build_blueprint()

        assert len(bp.nodes) == 0
        assert len(bp.data_flow_edges) == 0

    def test_compose_manually(self):
        ctx = _make_ctx()
        builder = CustomPipelineBuilder(ctx)
        builder.add_stage("A", MagicMock())
        builder.add_stage("B", MagicMock())
        builder.add_data_flow_edge("A", "B", "input")
        bp = builder.build_blueprint()

        assert set(bp.nodes.keys()) == {"A", "B"}
        assert ("A", "B") in _edge_pairs(bp)


# ===================================================================
# Edge cases / regression tests
# ===================================================================


class TestPipelineBuilderEdgeCases:
    def test_remove_nonexistent_stage_is_noop(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.remove_stage("DoesNotExist")
        bp = builder.build_blueprint()
        assert len(bp.nodes) == 0

    def test_remove_exec_dep_from_unknown_stage(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        dep = ExecutionOrderDependency.on_success("X")
        builder.remove_exec_dep("Unknown", dep)
        bp = builder.build_blueprint()
        assert bp.exec_order_deps is None

    def test_remove_data_flow_edge_nonexistent(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_stage("A", MagicMock())
        builder.add_stage("B", MagicMock())
        builder.remove_data_flow_edge("A", "B")
        bp = builder.build_blueprint()
        assert len(bp.data_flow_edges) == 0

    def test_multiple_edges_between_same_stages(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_data_flow_edge("A", "B", "input1")
        builder.add_data_flow_edge("A", "B", "input2")
        bp = builder.build_blueprint()
        ab_edges = [
            e
            for e in bp.data_flow_edges
            if e.source_stage == "A" and e.destination_stage == "B"
        ]
        assert len(ab_edges) == 2

    def test_remove_edge_removes_all_between_pair(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_data_flow_edge("A", "B", "input1")
        builder.add_data_flow_edge("A", "B", "input2")
        builder.remove_data_flow_edge("A", "B")
        bp = builder.build_blueprint()
        assert ("A", "B") not in _edge_pairs(bp)

    def test_remove_stage_preserves_unrelated_edges(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_stage("A", MagicMock())
        builder.add_stage("B", MagicMock())
        builder.add_stage("C", MagicMock())
        builder.add_data_flow_edge("A", "C", "x")
        builder.add_data_flow_edge("A", "B", "y")

        builder.remove_stage("B")
        bp = builder.build_blueprint()
        assert ("A", "C") in _edge_pairs(bp)
        assert ("A", "B") not in _edge_pairs(bp)

    def test_remove_stage_cleans_deps_referencing_removed_stage(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_stage("A", MagicMock())
        builder.add_stage("B", MagicMock())
        builder.add_stage("C", MagicMock())
        builder.add_exec_dep("A", ExecutionOrderDependency.on_success("B"))
        builder.add_exec_dep("A", ExecutionOrderDependency.on_success("C"))

        builder.remove_stage("B")
        bp = builder.build_blueprint()
        assert "B" not in _dep_names(bp, "A")
        assert "C" in _dep_names(bp, "A")

    def test_empty_deps_dict_becomes_none(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        bp = builder.build_blueprint()
        assert bp.exec_order_deps is None

    def test_nonempty_deps_dict_is_preserved(self):
        ctx = _make_ctx()
        builder = PipelineBuilder(ctx)
        builder.add_exec_dep("A", ExecutionOrderDependency.on_success("B"))
        bp = builder.build_blueprint()
        assert bp.exec_order_deps is not None
        assert "A" in bp.exec_order_deps


# ===================================================================
# Phase -1 Fix A: stage_timeout threading (regression guard)
# ===================================================================
#
# Audit `/tmp/hydra_bypass_audit.md` documented that 6 stage factories in
# default_pipelines.py hardcoded DEFAULT_SIMPLE_STAGE_TIMEOUT, silently
# ignoring the researcher-provided Hydra `stage_timeout` override.
# These tests pin every critical bypass site and enforce that subclasses
# (Context, AlgoTuneSpeed, CMAOpt, OptunaOpt) accept and thread
# ``stage_timeout`` through to the stages they build.


@pytest.fixture
def real_problem_dir(tmp_path: Path) -> Path:
    """Problem directory with stub context.py / validate.py / validate2.py
    so that CallFileFunction, OptunaOptStage, etc. can materialise."""
    prob_dir = tmp_path / "problem"
    prob_dir.mkdir()
    (prob_dir / "context.py").write_text("def build_context(**kw): return {}\n")
    (prob_dir / "validate.py").write_text("def validate(**kw): return {}\n")
    (prob_dir / "validate2.py").write_text("def validate(**kw): return {}\n")
    return prob_dir


_OVERRIDE = 6000  # distinct from DEFAULT_SIMPLE_STAGE_TIMEOUT=2400


class TestStageTimeoutThreading:
    """Every PipelineBuilder subclass must accept ``stage_timeout`` and thread
    it into every stage it constructs. Audit ref: /tmp/hydra_bypass_audit.md
    CRITICAL-1."""

    def test_context_builder_accepts_stage_timeout_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = ContextPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        assert builder._stage_timeout == _OVERRIDE

    def test_context_addcontext_stage_uses_stage_timeout(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = ContextPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        add_ctx = bp.nodes["AddContext"]()
        assert add_ctx.timeout == _OVERRIDE, (
            "AddContext stage ignored stage_timeout override — "
            "default_pipelines.py:437 hardcodes DEFAULT_SIMPLE_STAGE_TIMEOUT"
        )

    def test_algotune_builder_accepts_stage_timeout_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = AlgoTuneSpeedPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        assert builder._stage_timeout == _OVERRIDE

    def test_algotune_runtime_fitness_stage_uses_stage_timeout(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        # metrics.yaml is missing — the builder handles that gracefully.
        builder = AlgoTuneSpeedPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        rt = bp.nodes["RuntimeFitnessStage"]()
        assert rt.timeout == _OVERRIDE, (
            "RuntimeFitnessStage ignored stage_timeout — "
            "default_pipelines.py:481 hardcodes DEFAULT_SIMPLE_STAGE_TIMEOUT"
        )

    def test_cma_builder_accepts_stage_timeout_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = CMAOptPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        assert builder._stage_timeout == _OVERRIDE

    def test_cma_addcontext_stage_uses_stage_timeout(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = CMAOptPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        add_ctx = bp.nodes["AddContext"]()
        assert add_ctx.timeout == _OVERRIDE, (
            "CMA AddContext ignored stage_timeout — "
            "default_pipelines.py:546 hardcodes DEFAULT_SIMPLE_STAGE_TIMEOUT"
        )

    def test_optuna_builder_accepts_stage_timeout_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = OptunaOptPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        assert builder._stage_timeout == _OVERRIDE

    def test_optuna_addcontext_stage_uses_stage_timeout(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = OptunaOptPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        add_ctx = bp.nodes["AddContext"]()
        assert add_ctx.timeout == _OVERRIDE, (
            "Optuna AddContext ignored stage_timeout — "
            "default_pipelines.py:672 hardcodes DEFAULT_SIMPLE_STAGE_TIMEOUT"
        )

    def test_optuna_payload_bridge_uses_stage_timeout(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = OptunaOptPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        bridge = bp.nodes["OptunaPayloadBridge"]()
        assert bridge.timeout == _OVERRIDE, (
            "OptunaPayloadBridge ignored stage_timeout — "
            "default_pipelines.py:757 hardcodes DEFAULT_SIMPLE_STAGE_TIMEOUT"
        )

    def test_optuna_payload_resolver_uses_stage_timeout(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = OptunaOptPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        resolver = bp.nodes["PayloadResolver"]()
        assert resolver.timeout == _OVERRIDE, (
            "PayloadResolver ignored stage_timeout — "
            "default_pipelines.py:761 hardcodes DEFAULT_SIMPLE_STAGE_TIMEOUT"
        )

    def test_default_pipeline_stages_already_thread_stage_timeout(
        self, real_problem_dir
    ):
        """Regression guard for the stages that were ALREADY correct in
        DefaultPipelineBuilder (used local stage_timeout binding at
        line 180). Catches any future regression that reintroduces the
        literal constant at these sites."""
        ctx = _make_ctx(problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, stage_timeout=_OVERRIDE)
        bp = builder.build_blueprint()
        for name in [
            "ValidateCodeStage",
            "CallProgramFunction",
            "CallValidatorFunction",
            "FetchMetrics",
            "FetchArtifact",
            "FormatterStage",
            "InsightsStage",
            "MutationContextStage",
            "ComputeComplexityStage",
        ]:
            stage = bp.nodes[name]()
            assert stage.timeout == _OVERRIDE, (
                f"{name} lost stage_timeout threading — check for "
                "DEFAULT_SIMPLE_STAGE_TIMEOUT regression"
            )


_DAG_CONC_OVERRIDE = 32  # distinct from both 8 (old py) and 16 (hydra)


class TestDagConcurrencyThreading:
    """PipelineBuilder must accept a ``max_parallel`` override and propagate it
    to ``DAGBlueprint.max_parallel_stages``.  Also guards against the Python
    constant drifting from the Hydra default.
    Audit ref: /tmp/hydra_bypass_audit.md CRITICAL-2."""

    def test_python_and_hydra_defaults_agree(self):
        """Regression guard: DEFAULT_DAG_CONCURRENCY must match
        config/constants/pipeline.yaml's dag_concurrency."""
        from omegaconf import OmegaConf

        repo_root = Path(__file__).resolve().parents[2]
        cfg = OmegaConf.load(repo_root / "config/constants/pipeline.yaml")
        assert cfg.dag_concurrency == DEFAULT_DAG_CONCURRENCY, (
            f"Hydra default dag_concurrency={cfg.dag_concurrency} != "
            f"Python DEFAULT_DAG_CONCURRENCY={DEFAULT_DAG_CONCURRENCY} — "
            "reconcile the two; one is silently losing."
        )

    def test_default_builder_accepts_max_parallel_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, max_parallel=_DAG_CONC_OVERRIDE)
        assert builder._max_parallel == _DAG_CONC_OVERRIDE

    def test_default_builder_threads_max_parallel_into_blueprint(
        self, real_problem_dir
    ):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, max_parallel=_DAG_CONC_OVERRIDE)
        bp = builder.build_blueprint()
        assert bp.max_parallel_stages == _DAG_CONC_OVERRIDE

    def test_context_builder_accepts_max_parallel_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = ContextPipelineBuilder(ctx, max_parallel=_DAG_CONC_OVERRIDE)
        assert builder._max_parallel == _DAG_CONC_OVERRIDE

    def test_cma_builder_accepts_max_parallel_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = CMAOptPipelineBuilder(ctx, max_parallel=_DAG_CONC_OVERRIDE)
        assert builder._max_parallel == _DAG_CONC_OVERRIDE

    def test_optuna_builder_accepts_max_parallel_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = OptunaOptPipelineBuilder(ctx, max_parallel=_DAG_CONC_OVERRIDE)
        assert builder._max_parallel == _DAG_CONC_OVERRIDE

    def test_algotune_builder_accepts_max_parallel_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = AlgoTuneSpeedPipelineBuilder(ctx, max_parallel=_DAG_CONC_OVERRIDE)
        assert builder._max_parallel == _DAG_CONC_OVERRIDE

    def test_default_max_parallel_uses_default_dag_concurrency(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx)
        assert builder._max_parallel == DEFAULT_DAG_CONCURRENCY


_MAX_INSIGHTS_OVERRIDE = 21  # distinct from DEFAULT_MAX_INSIGHTS=8


class TestMaxInsightsThreading:
    """PipelineBuilder must accept ``max_insights`` and forward it to
    InsightsStage. Audit ref: /tmp/hydra_bypass_audit.md CRITICAL-3."""

    def test_default_builder_accepts_max_insights_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, max_insights=_MAX_INSIGHTS_OVERRIDE)
        assert builder._max_insights == _MAX_INSIGHTS_OVERRIDE

    def test_default_insights_stage_uses_max_insights(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, max_insights=_MAX_INSIGHTS_OVERRIDE)
        bp = builder.build_blueprint()
        stage = bp.nodes["InsightsStage"]()
        assert stage._max_insights == _MAX_INSIGHTS_OVERRIDE, (
            "InsightsStage ignored max_insights override — check for "
            "DEFAULT_MAX_INSIGHTS literal regression"
        )

    def test_context_builder_accepts_max_insights_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = ContextPipelineBuilder(ctx, max_insights=_MAX_INSIGHTS_OVERRIDE)
        assert builder._max_insights == _MAX_INSIGHTS_OVERRIDE


_MAX_CODE_LENGTH_OVERRIDE = 12345  # distinct from MAX_CODE_LENGTH=30000


class TestMaxCodeLengthThreading:
    """PipelineBuilder must accept ``max_code_length`` and forward it to
    ValidateCodeStage. Audit ref: /tmp/hydra_bypass_audit.md CRITICAL-4."""

    def test_default_builder_accepts_max_code_length_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, max_code_length=_MAX_CODE_LENGTH_OVERRIDE)
        assert builder._max_code_length == _MAX_CODE_LENGTH_OVERRIDE

    def test_default_validate_stage_uses_max_code_length(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=False, problem_dir=real_problem_dir)
        builder = DefaultPipelineBuilder(ctx, max_code_length=_MAX_CODE_LENGTH_OVERRIDE)
        bp = builder.build_blueprint()
        stage = bp.nodes["ValidateCodeStage"]()
        assert stage.max_code_length == _MAX_CODE_LENGTH_OVERRIDE, (
            "ValidateCodeStage ignored max_code_length override — check for "
            "MAX_CODE_LENGTH literal regression"
        )

    def test_context_builder_accepts_max_code_length_kwarg(self, real_problem_dir):
        ctx = _make_ctx(is_contextual=True, problem_dir=real_problem_dir)
        builder = ContextPipelineBuilder(ctx, max_code_length=_MAX_CODE_LENGTH_OVERRIDE)
        assert builder._max_code_length == _MAX_CODE_LENGTH_OVERRIDE
