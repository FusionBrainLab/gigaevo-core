"""Tests for gigaevo/evolution/mutation/context.py — all MutationContext subclasses."""

from __future__ import annotations

from gigaevo.evolution.mutation.context import (
    ArtifactMutationContext,
    CompositeMutationContext,
    EvolutionaryStatisticsMutationContext,
    FamilyTreeMutationContext,
    InsightsMutationContext,
    MetricsMutationContext,
    PreformattedMutationContext,
    ensure_models_rebuilt,
)
from gigaevo.llm.agents.insights import ProgramInsight, ProgramInsights
from gigaevo.llm.agents.lineage import (
    TransitionAnalysis,
    TransitionInsight,
    TransitionInsights,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.stages.collector import (
    EvolutionaryStatistics,
    GenerationMetrics,
)

ensure_models_rebuilt()


def _make_ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="Score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
            "cost": MetricSpec(
                description="Cost",
                higher_is_better=False,
            ),
        }
    )


def _make_formatter() -> MetricsFormatter:
    return MetricsFormatter(_make_ctx())


# ---------------------------------------------------------------------------
# MetricsMutationContext
# ---------------------------------------------------------------------------


class TestMetricsMutationContext:
    def test_format_includes_header(self) -> None:
        ctx = MetricsMutationContext(
            metrics={"score": 80.0, "cost": 5.0},
            metrics_formatter=_make_formatter(),
        )
        result = ctx.format()
        assert "## Program Metrics" in result
        assert "score" in result


# ---------------------------------------------------------------------------
# InsightsMutationContext
# ---------------------------------------------------------------------------


class TestInsightsMutationContext:
    def test_format_with_insights(self) -> None:
        insights = ProgramInsights(
            insights=[
                ProgramInsight(
                    type="perf",
                    insight="Loop is slow",
                    tag="optimization",
                    severity="medium",
                )
            ]
        )
        ctx = InsightsMutationContext(insights=insights)
        result = ctx.format()
        assert "## Program Insights" in result
        assert "Loop is slow" in result

    def test_format_empty_insights(self) -> None:
        insights = ProgramInsights(insights=[])
        ctx = InsightsMutationContext(insights=insights)
        result = ctx.format()
        assert "No insights available" in result


# ---------------------------------------------------------------------------
# FamilyTreeMutationContext
# ---------------------------------------------------------------------------


def _make_transition(
    from_id: str, to_id: str, p_score: float, c_score: float
) -> TransitionAnalysis:
    return TransitionAnalysis(
        **{
            "from": from_id,
            "to": to_id,
            "parent_metrics": {"score": p_score, "cost": 10.0},
            "child_metrics": {"score": c_score, "cost": 8.0},
            "diff_blocks": ["- old\n+ new"],
            "insights": TransitionInsights(
                insights=[
                    TransitionInsight(
                        strategy="imitation", description="Copied approach"
                    ),
                    TransitionInsight(
                        strategy="exploration", description="Tried new idea"
                    ),
                    TransitionInsight(
                        strategy="generalization", description="Generalized pattern"
                    ),
                ]
            ),
        }
    )


class TestFamilyTreeMutationContext:
    def test_format_with_ancestors_and_descendants(self) -> None:
        ctx = FamilyTreeMutationContext(
            ancestors=[_make_transition("aaaa1111", "bbbb2222", 50.0, 60.0)],
            descendants=[_make_transition("bbbb2222", "cccc3333", 60.0, 55.0)],
            metrics_formatter=_make_formatter(),
        )
        result = ctx.format()
        assert "### Parents" in result
        assert "### Children" in result
        assert "aaaa1111" in result

    def test_format_empty_lists(self) -> None:
        ctx = FamilyTreeMutationContext(
            ancestors=[],
            descendants=[],
            metrics_formatter=_make_formatter(),
        )
        result = ctx.format()
        # With no ancestors or descendants, result should be empty
        assert result == ""


# ---------------------------------------------------------------------------
# EvolutionaryStatisticsMutationContext
# ---------------------------------------------------------------------------


def _make_evo_stats() -> EvolutionaryStatistics:
    gen_metrics = GenerationMetrics(
        best=90.0,
        worst=10.0,
        average=50.0,
        valid_rate=0.8,
        avg_num_children=2.5,
        max_num_children=5,
        program_count=20,
    )
    return EvolutionaryStatistics(
        generation=3,
        iteration=10,
        current_program_metrics={"score": 75.0, "cost": 5.0},
        best_fitness={"score": 95.0},
        worst_fitness={"score": 5.0},
        average_fitness={"score": 50.0},
        valid_rate=0.75,
        total_program_count=100,
        avg_num_children=2.0,
        max_num_children=8,
        best_fitness_in_generation={"score": 90.0},
        worst_fitness_in_generation={"score": 20.0},
        average_fitness_in_generation={"score": 55.0},
        valid_rate_in_generation=0.8,
        ancestor_count=2,
        best_fitness_in_ancestors={"score": 80.0},
        worst_fitness_in_ancestors={"score": 60.0},
        average_fitness_in_ancestors={"score": 70.0},
        valid_rate_in_ancestors=1.0,
        descendant_count=3,
        best_fitness_in_descendants={"score": 85.0},
        worst_fitness_in_descendants={"score": 40.0},
        average_fitness_in_descendants={"score": 65.0},
        valid_rate_in_descendants=0.67,
        generation_history={3: gen_metrics},
    )


class TestEvolutionaryStatisticsMutationContext:
    def test_format_with_history(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "## Evolutionary Statistics" in result
        assert "Generation" in result
        assert "100" in result  # total_program_count

    def test_format_empty_history(self) -> None:
        stats = _make_evo_stats()
        stats.generation_history = {}
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=stats,
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "No generation history" in result


# ---------------------------------------------------------------------------
# ArtifactMutationContext
# ---------------------------------------------------------------------------


class TestArtifactMutationContext:
    def test_format_string_artifact(self) -> None:
        ctx = ArtifactMutationContext(artifact="hello world")
        result = ctx.format()
        assert "## Execution Artifact" in result
        assert "hello world" in result

    def test_format_none_artifact(self) -> None:
        ctx = ArtifactMutationContext(artifact=None)
        result = ctx.format()
        assert "<no artifact>" in result

    def test_format_non_string_uses_repr(self) -> None:
        ctx = ArtifactMutationContext(artifact=[1, 2, 3])
        result = ctx.format()
        assert "[1, 2, 3]" in result


# ---------------------------------------------------------------------------
# PreformattedMutationContext
# ---------------------------------------------------------------------------


class TestPreformattedMutationContext:
    def test_format_returns_content(self) -> None:
        ctx = PreformattedMutationContext(content="## Custom Block\nSome text")
        assert ctx.format() == "## Custom Block\nSome text"


# ---------------------------------------------------------------------------
# CompositeMutationContext
# ---------------------------------------------------------------------------


class TestCompositeMutationContext:
    def test_joins_parts(self) -> None:
        ctx = CompositeMutationContext(
            contexts=[
                PreformattedMutationContext(content="Part A"),
                PreformattedMutationContext(content="Part B"),
            ]
        )
        result = ctx.format()
        assert "Part A" in result
        assert "Part B" in result
        assert "---" in result  # separator

    def test_all_empty_returns_fallback(self) -> None:
        ctx = CompositeMutationContext(
            contexts=[
                PreformattedMutationContext(content=""),
                PreformattedMutationContext(content="   "),
            ]
        )
        result = ctx.format()
        assert result == "No context available."
