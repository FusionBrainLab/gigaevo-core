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
)
from gigaevo.llm.agents.insights import ProgramInsight, ProgramInsights
from gigaevo.llm.agents.lineage import (
    TransitionAnalysis,
    TransitionInsight,
    TransitionInsights,
)
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.stages.collector import EvolutionaryStatistics


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


def _make_evo_stats(**overrides) -> EvolutionaryStatistics:
    base = dict(
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
        iter_window_lo=0,
        iter_window_hi=20,
        iter_window_programs=15,
        iter_window_valid=12,
        iter_window_best_fitness=88.0,
        iter_window_best_iter=7,
        iter_window_rank=4,
        iter_window_median_before=60.0,
        iter_window_median_after=72.0,
        iter_window_trend="rising",
        iter_window_trend_thirds=(50.0, 65.0, 80.0),
        iter_window_invalid_streak_max=2,
        iter_window_invalid_count=3,
        iters_since_last_new_best=3,
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
    )
    base.update(overrides)
    return EvolutionaryStatistics(**base)


class TestEvolutionaryStatisticsMutationContext:
    def test_format_includes_header_and_focal(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "## Evolutionary Statistics" in result
        assert "iter=10" in result
        assert "gen=3" in result
        assert "rank 4/12 in window" in result

    def test_format_window_summary(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Window: iters [0..20]" in result
        assert "programs=15" in result
        assert "valid=12" in result

    def test_format_trend_and_thirds(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Trend (window): rising" in result
        assert "medians of thirds" in result

    def test_format_best_and_plateau(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Best in window:" in result
        assert "at iter 7" in result
        assert "Iters since last new best (global): 3" in result

    def test_format_median_after_omitted_when_none(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(iter_window_median_after=None),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Median fitness, 10 iters before this program" in result
        assert "Median fitness, 10 iters after this program" not in result

    def test_format_failure_rate_line(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Recent failure rate: 3/15 invalid" in result
        assert "Invalid streak (max consecutive in window): 2" in result

    def test_format_invalid_focal_shows_INVALID(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"is_valid": 0.0},
                iter_window_rank=None,
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "fitness=INVALID" in result
        assert "rank" not in result

    def test_format_head_of_run_no_window(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                iter_window_lo=None,
                iter_window_hi=None,
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "## Evolutionary Statistics" in result
        assert "window unavailable" in result


class TestArchivePercentileV3:
    """R7+R8 v3 — archive-percentile annotation, no Regime literal."""

    def _archive(self, *values: float) -> tuple[float, ...]:
        return tuple(sorted(values))

    def test_archive_line_omitted_when_archive_lt_4(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 25.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(10.0, 30.0, 50.0),  # n=3
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Archive:" not in result
        assert "archive-percentile" not in result

    def test_regime_literal_never_rendered_in_v3(self) -> None:
        # v3 drops the categorical Regime: tag entirely — the prose gate
        # reads the percentile + target gap directly.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 25.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Regime:" not in result
        assert "archive-quartile" not in result

    def test_focal_at_worst_end_renders_low_percentile(self) -> None:
        # higher_is_better=True; focal=5.0 is the worst in the archive.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 5.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Archive: N=8" in result
        # focal=5.0 ≤ 1 of 8 archive entries (only itself) → p=round(1/8*100)=12
        assert "archive-percentile p12 of N=8" in result

    def test_focal_at_best_end_renders_p100(self) -> None:
        # focal=75.0 dominates the archive → p100.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 75.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "archive-percentile p100 of N=8" in result

    def test_focal_in_middle_renders_mid_percentile(self) -> None:
        # focal=35.0 → 4 of 8 archive entries are ≤ 35.0 → p=50.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 35.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "archive-percentile p50 of N=8" in result

    def test_target_line_never_rendered_when_upper_bound_declared(self) -> None:
        # v3.1: Target: line is ALWAYS omitted regardless of upper_bound. The
        # LLM reads the target from the task description and judges qualitatively
        # against the rendered Archive distribution. _make_ctx has
        # upper_bound=100.0 for "score" — we assert nothing leaks here.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 70.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Target:" not in result
        assert "focal_gap" not in result

    def test_target_line_never_rendered_when_upper_bound_none(self) -> None:
        ctx_no_target = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="Score",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
            }
        )
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 30.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=ctx_no_target,
        )
        result = ctx.format()
        assert "Target:" not in result
        assert "focal_gap" not in result
        # archive-percentile is independent of upper_bound and still renders.
        assert "archive-percentile" in result

    def test_higher_is_better_false_focal_at_best_end_renders_p100(self) -> None:
        # Loss-style metric: lower = better. focal=0.05 is BEST in archive → p100.
        ctx_lower = MetricsContext(
            specs={
                "loss": MetricSpec(
                    description="Loss",
                    is_primary=True,
                    higher_is_better=False,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
            }
        )
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"loss": 0.05},
                archive_valid_fitnesses=self._archive(
                    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70
                ),
            ),
            metrics_context=ctx_lower,
        )
        result = ctx.format()
        # 8 archive entries ≥ 0.05 → p100.
        assert "archive-percentile p100 of N=8" in result

    def test_higher_is_better_false_focal_at_worst_end_renders_low_pct(self) -> None:
        ctx_lower = MetricsContext(
            specs={
                "loss": MetricSpec(
                    description="Loss",
                    is_primary=True,
                    higher_is_better=False,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
            }
        )
        # focal=0.70 is WORST (lower=better) → only 1 entry ≥ 0.70 (itself) → p12.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"loss": 0.70},
                archive_valid_fitnesses=self._archive(
                    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70
                ),
            ),
            metrics_context=ctx_lower,
        )
        result = ctx.format()
        assert "archive-percentile p12 of N=8" in result

    def test_archive_line_includes_worst_higher_is_better_true(self) -> None:
        # v3.1: Archive line renders N, worst, median, best — worst is the
        # minimum value for higher_is_better=True so the LLM can see the
        # spread (compression-detection signal).
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 35.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "Archive: N=8" in result
        # higher_is_better=True → worst is the minimum (5.0).
        assert "worst=5" in result
        # best is the maximum (70.0).
        assert "best=70" in result

    def test_archive_line_worst_inverts_for_higher_is_better_false(self) -> None:
        # Loss-style metric: lower=better → worst is the MAXIMUM value, best
        # is the minimum value. Verifies direction-aware rendering.
        ctx_lower = MetricsContext(
            specs={
                "loss": MetricSpec(
                    description="Loss",
                    is_primary=True,
                    higher_is_better=False,
                    lower_bound=0.0,
                    upper_bound=None,
                ),
            }
        )
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"loss": 0.30},
                archive_valid_fitnesses=self._archive(
                    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70
                ),
            ),
            metrics_context=ctx_lower,
        )
        result = ctx.format()
        assert "Archive: N=8" in result
        # lower=better → worst is the max (0.70).
        assert "worst=0.70" in result
        # best is the min (0.05).
        assert "best=0.05" in result

    def test_invalid_focal_no_percentile_emitted(self) -> None:
        # Focal invalid → no percentile annotation.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"is_valid": 0.0},
                iter_window_rank=None,
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "archive-percentile" not in result
        assert "Archive:" not in result

    def test_compressed_bootstrap_renders_rich_archive_no_target_line(
        self,
    ) -> None:
        # The exact cycle-8 bootstrap-mislead case: focal dominates a tiny
        # compressed archive but is still nowhere near the task's target.
        # v3.1 surfaces archive-percentile p100 alongside the rich Archive
        # line (worst / median / best). No Target: line is rendered — the
        # LLM reads the target from the task description and judges the
        # compression qualitatively against the rendered distribution.
        ctx_compressed_archive = MetricsContext(
            specs={
                "score": MetricSpec(
                    description="Score",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=0.0365,
                ),
            }
        )
        # focal=0.00288 dominates archive but task target ≈ 0.0365 is far away.
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 0.00288, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    1e-5, 3e-5, 8e-5, 3.2e-4, 1.65e-3, 5e-4, 1e-4
                ),  # n=7 compressed near zero
            ),
            metrics_context=ctx_compressed_archive,
        )
        result = ctx.format()
        assert "archive-percentile p100 of N=7" in result
        # v3.1: no rendered Target/focal_gap tokens.
        assert "Target:" not in result
        assert "focal_gap" not in result
        # Rich Archive line must carry worst / median / best so the LLM can
        # see the compression on its own.
        assert "Archive: N=7" in result
        assert "worst=" in result
        assert "median=" in result
        assert "best=" in result

    def test_rank_line_includes_percentile_when_archive_present(self) -> None:
        ctx = EvolutionaryStatisticsMutationContext(
            evolutionary_statistics=_make_evo_stats(
                current_program_metrics={"score": 25.0, "cost": 5.0},
                archive_valid_fitnesses=self._archive(
                    5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
                ),
            ),
            metrics_context=_make_ctx(),
        )
        result = ctx.format()
        assert "rank 4/12 in window, archive-percentile p" in result


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
