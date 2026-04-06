"""Tests for gigaevo.adversarial.pipeline.AdversarialPipelineBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.adversarial.pipeline import AdversarialPipelineBuilder
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.runner.dag_blueprint import DAGBlueprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(OpponentArchiveProvider):
    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        return []


def _make_metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
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


def _make_ctx(problem_dir: Path | None = None) -> EvolutionContext:
    p_dir = problem_dir or Path("/fake/problem")
    metrics_ctx = _make_metrics_context()

    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = p_dir
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False

    storage = MagicMock(spec=ProgramStorage)
    llm_wrapper = MagicMock(spec=MultiModelRouter)

    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=llm_wrapper,
        storage=storage,
    )


def _edge_pairs(blueprint: DAGBlueprint) -> set[tuple[str, str]]:
    return {(e.source_stage, e.destination_stage) for e in blueprint.data_flow_edges}


def _dep_names(blueprint: DAGBlueprint, stage: str) -> set[str]:
    if blueprint.exec_order_deps is None:
        return set()
    return {d.stage_name for d in blueprint.exec_order_deps.get(stage, [])}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdversarialPipelineBuilder:
    def test_inherits_all_default_stages(self):
        ctx = _make_ctx()
        default_bp = DefaultPipelineBuilder(ctx).build_blueprint()
        adversarial_bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()

        default_stages = set(default_bp.nodes.keys())
        adversarial_stages = set(adversarial_bp.nodes.keys())
        assert default_stages.issubset(adversarial_stages)

    def test_adds_fetch_opponent_results_stage(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        assert "FetchOpponentResultsStage" in bp.nodes

    def test_opponent_results_wired_as_context_to_validator(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        edges = _edge_pairs(bp)
        assert ("FetchOpponentResultsStage", "CallValidatorFunction") in edges

    def test_opponent_context_input_name(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        context_edges = [
            e
            for e in bp.data_flow_edges
            if e.source_stage == "FetchOpponentResultsStage"
            and e.destination_stage == "CallValidatorFunction"
        ]
        assert len(context_edges) == 1
        assert context_edges[0].input_name == "context"

    def test_fetch_opponents_depends_on_validate_code(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        assert "ValidateCodeStage" in _dep_names(bp, "FetchOpponentResultsStage")

    def test_program_function_still_wired_as_payload(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        edges = _edge_pairs(bp)
        assert ("CallProgramFunction", "CallValidatorFunction") in edges

    def test_standard_metrics_pipeline_intact(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        edges = _edge_pairs(bp)
        assert ("CallValidatorFunction", "FetchMetrics") in edges
        assert ("FetchMetrics", "MergeMetricsStage") in edges
        assert ("MergeMetricsStage", "EnsureMetricsStage") in edges
        assert ("EnsureMetricsStage", "MutationContextStage") in edges

    def test_only_one_new_stage_vs_default(self):
        ctx = _make_ctx()
        default_bp = DefaultPipelineBuilder(ctx).build_blueprint()
        adversarial_bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()

        new_stages = set(adversarial_bp.nodes.keys()) - set(default_bp.nodes.keys())
        assert new_stages == {"FetchOpponentResultsStage"}

    def test_all_factories_are_callable(self):
        ctx = _make_ctx()
        bp = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider()
        ).build_blueprint()
        for name, factory in bp.nodes.items():
            assert callable(factory), f"{name} factory is not callable"

    def test_custom_n_opponents(self):
        ctx = _make_ctx()
        builder = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider(), n_opponents=10
        )
        bp = builder.build_blueprint()
        assert "FetchOpponentResultsStage" in bp.nodes

    def test_custom_dag_timeout(self):
        ctx = _make_ctx()
        builder = AdversarialPipelineBuilder(
            ctx, opponent_provider=FakeProvider(), dag_timeout=2000.0
        )
        bp = builder.build_blueprint()
        assert bp.dag_timeout == 2000.0

    def test_fallback_codes_loaded_from_directory(self, tmp_path: Path):
        """If fallback dir exists with .py files, they are loaded."""
        problem_dir = tmp_path / "problem"
        problem_dir.mkdir()
        fallback_dir = problem_dir / "fallback"
        fallback_dir.mkdir()
        (fallback_dir / "a.py").write_text("def entrypoint(): return 1")
        (fallback_dir / "b.py").write_text("def entrypoint(): return 2")
        # Also create evaluate.py (needed by CallValidatorFunction)
        (problem_dir / "evaluate.py").write_text(
            "def evaluate(ctx, payload): return {'fitness': 0.0}"
        )

        ctx = _make_ctx(problem_dir=problem_dir)
        builder = AdversarialPipelineBuilder(ctx, opponent_provider=FakeProvider())
        bp = builder.build_blueprint()
        assert "FetchOpponentResultsStage" in bp.nodes

    def test_no_fallback_dir_is_ok(self, tmp_path: Path):
        """If no fallback dir exists, builder proceeds with empty fallback."""
        problem_dir = tmp_path / "problem"
        problem_dir.mkdir()
        (problem_dir / "evaluate.py").write_text(
            "def evaluate(ctx, payload): return {'fitness': 0.0}"
        )

        ctx = _make_ctx(problem_dir=problem_dir)
        builder = AdversarialPipelineBuilder(ctx, opponent_provider=FakeProvider())
        bp = builder.build_blueprint()
        assert "FetchOpponentResultsStage" in bp.nodes


# ---------------------------------------------------------------------------
# Tests: evaluate.py cross-play
# ---------------------------------------------------------------------------

_PROJ = Path(__file__).resolve().parent.parent.parent


def _load_evaluate(pop: str):
    """Import evaluate.py from a problem directory without package __init__.py."""
    import importlib.util

    path = _PROJ / f"problems/adversarial/optimizer_v2/{pop}/evaluate.py"
    spec = importlib.util.spec_from_file_location(f"evaluate_{pop}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate


class TestPopAEvaluate:
    """Unit tests for the Pop A evaluate.py logic (optimizer side)."""

    def test_non_callable_returns_zero(self):
        evaluate = _load_evaluate("pop_a")
        result = evaluate([], "not a callable")
        assert result["fitness"] == 0.0
        assert result["is_valid"] == 0.0

    def test_no_opponents_returns_zero(self):
        evaluate = _load_evaluate("pop_a")
        result = evaluate([], lambda f, bounds, budget: [0.0] * 5)
        assert result["fitness"] == 0.0
        assert result["n_opponents"] == 0.0

    def test_perfect_optimizer_on_quadratic(self):
        evaluate = _load_evaluate("pop_a")

        def perfect_optimizer(f, bounds, budget):
            return [0.0] * len(bounds)

        landscape = (
            lambda x: sum(xi**2 for xi in x),  # quadratic, min at origin
            [(-5, 5)] * 3,
            [0.0, 0.0, 0.0],
            100,
        )
        result = evaluate([landscape], perfect_optimizer)
        assert result["fitness"] == 1.0
        assert result["is_valid"] == 1.0
        assert result["n_opponents"] == 1.0

    def test_bad_optimizer_gets_low_score(self):
        evaluate = _load_evaluate("pop_a")

        def bad_optimizer(f, bounds, budget):
            return [5.0] * len(bounds)  # far from optimum

        landscape = (
            lambda x: sum(xi**2 for xi in x),
            [(-5, 5)] * 3,
            [0.0, 0.0, 0.0],
            100,
        )
        result = evaluate([landscape], bad_optimizer)
        assert result["fitness"] < 0.6


class TestPopBEvaluate:
    """Unit tests for the Pop B evaluate.py logic (landscape side)."""

    def test_invalid_output_returns_zero(self):
        evaluate = _load_evaluate("pop_b")
        result = evaluate([], "not a tuple")
        assert result["fitness"] == 0.0
        assert result["is_valid"] == 0.0

    def test_no_opponents_returns_zero(self):
        evaluate = _load_evaluate("pop_b")
        landscape = (
            lambda x: sum(xi**2 for xi in x),
            [(-5, 5)] * 3,
            [0.0, 0.0, 0.0],
            100,
        )
        result = evaluate([], landscape)
        assert result["fitness"] == 0.0
        assert result["n_opponents"] == 0.0

    def test_deceptive_landscape_scores_well_against_naive(self):
        evaluate = _load_evaluate("pop_b")

        def naive_optimizer(f, bounds, budget):
            # Always returns the center — will be far from hidden optimum
            return [0.0] * len(bounds)

        dim = 3
        optimum = [4.0, -3.0, 2.0]

        def landscape(x):
            return sum((xi - oi) ** 2 for xi, oi in zip(x, optimum))

        result = evaluate(
            [naive_optimizer],
            (landscape, [(-5, 5)] * dim, optimum, 100),
        )
        assert result["is_valid"] == 1.0
        assert result["fitness"] > 0.0  # optimizer is far from optimum = deceptive

    def test_invalid_optimum_rejected(self):
        """If claimed optimum is not actually the minimum, is_valid=0."""
        import random

        evaluate = _load_evaluate("pop_b")
        random.seed(42)

        def landscape(x):
            # min at [0,0,0], but we claim it's at [5,5,5]
            return sum(xi**2 for xi in x)

        result = evaluate(
            [],
            (landscape, [(-10, 10)] * 3, [5.0, 5.0, 5.0], 100),
        )
        assert result["is_valid"] == 0.0
