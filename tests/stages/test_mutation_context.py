"""Tests for MutationContextStage."""

from __future__ import annotations

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.programs.core_types import StageState
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.common import FloatDictContainer, StringContainer
from gigaevo.programs.stages.insights_lineage import TransitionAnalysisList
from gigaevo.programs.stages.mutation_context import MutationContextStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="main score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
        }
    )


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


def _make_stage() -> MutationContextStage:
    stage = MutationContextStage(metrics_context=_ctx(), timeout=5.0)
    stage.__class__.cache_handler = NO_CACHE
    return stage


def _make_evo_stats() -> EvolutionaryStatistics:
    """Minimal EvolutionaryStatistics for testing."""
    return EvolutionaryStatistics(
        generation=1,
        iteration=None,
        current_program_metrics={"score": 50.0},
        best_fitness={"score": 90.0},
        worst_fitness={"score": 10.0},
        average_fitness={"score": 50.0},
        valid_rate=1.0,
        total_program_count=5,
        avg_num_children=1.0,
        max_num_children=3,
        best_fitness_in_generation={"score": 80.0},
        worst_fitness_in_generation={"score": 20.0},
        average_fitness_in_generation={"score": 50.0},
        valid_rate_in_generation=1.0,
        ancestor_count=1,
        best_fitness_in_ancestors={"score": 70.0},
        worst_fitness_in_ancestors={"score": 70.0},
        average_fitness_in_ancestors={"score": 70.0},
        valid_rate_in_ancestors=1.0,
        descendant_count=0,
        best_fitness_in_descendants={},
        worst_fitness_in_descendants={},
        average_fitness_in_descendants={},
        valid_rate_in_descendants=0.0,
    )


# ---------------------------------------------------------------------------
# TestMutationContextStage
# ---------------------------------------------------------------------------


class TestMutationContextStage:
    async def test_all_inputs_none(self):
        """All inputs None → empty context, metadata still set."""
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": None,
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": None,
                "evolutionary_statistics": None,
                "formatted": None,
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert isinstance(result.output.data, str)
        # Side effect: metadata key is set on program
        assert MUTATION_CONTEXT_METADATA_KEY in prog.metadata

    async def test_only_metrics_provided(self):
        """Only metrics → MetricsMutationContext in composite."""
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": FloatDictContainer(data={"score": 85.0}),
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": None,
                "evolutionary_statistics": None,
                "formatted": None,
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # Output should contain metric info
        assert "85" in result.output.data or "score" in result.output.data

    async def test_only_evolutionary_statistics(self):
        """Only evolutionary_statistics → EvolutionaryStatisticsMutationContext."""
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": None,
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": None,
                "evolutionary_statistics": _make_evo_stats(),
                "formatted": None,
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert len(result.output.data) > 0

    async def test_only_formatted_string(self):
        """Only formatted → PreformattedMutationContext."""
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": None,
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": None,
                "evolutionary_statistics": None,
                "formatted": StringContainer(data="Custom context text"),
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert "Custom context text" in result.output.data

    async def test_only_lineage_ancestors(self):
        """Only lineage_ancestors → FamilyTreeMutationContext with descendants=[]."""
        from gigaevo.llm.agents.lineage import (
            TransitionAnalysis,
            TransitionInsight,
            TransitionInsights,
        )

        insights = TransitionInsights(
            insights=[
                TransitionInsight(
                    strategy="imitation", description="Copied loop structure"
                ),
                TransitionInsight(
                    strategy="avoidance", description="Avoided recursion"
                ),
                TransitionInsight(
                    strategy="exploration", description="Tried new approach"
                ),
            ]
        )
        ancestors = TransitionAnalysisList(
            items=[
                TransitionAnalysis(
                    from_id="parent1",
                    to_id="child1",
                    parent_metrics={"score": 60.0},
                    child_metrics={"score": 80.0},
                    diff_blocks=["+ new line"],
                    insights=insights,
                )
            ]
        )
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": None,
                "insights": None,
                "lineage_ancestors": ancestors,
                "lineage_descendants": None,
                "evolutionary_statistics": None,
                "formatted": None,
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert len(result.output.data) > 0

    async def test_only_lineage_descendants(self):
        """Only lineage_descendants → FamilyTreeMutationContext with ancestors=[]."""
        from gigaevo.llm.agents.lineage import (
            TransitionAnalysis,
            TransitionInsight,
            TransitionInsights,
        )

        insights = TransitionInsights(
            insights=[
                TransitionInsight(strategy="imitation", description="Copied pattern"),
                TransitionInsight(
                    strategy="avoidance", description="Avoided complexity"
                ),
                TransitionInsight(
                    strategy="generalization", description="Made it generic"
                ),
            ]
        )
        descendants = TransitionAnalysisList(
            items=[
                TransitionAnalysis(
                    from_id="parent1",
                    to_id="child1",
                    parent_metrics={"score": 50.0},
                    child_metrics={"score": 70.0},
                    diff_blocks=["- old line", "+ new line"],
                    insights=insights,
                )
            ]
        )
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": None,
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": descendants,
                "evolutionary_statistics": None,
                "formatted": None,
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert len(result.output.data) > 0

    async def test_metadata_side_effect(self):
        """Verify set_metadata side effect on program."""
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": FloatDictContainer(data={"score": 50.0}),
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": None,
                "evolutionary_statistics": None,
                "formatted": None,
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        # The metadata should contain the same string as the output
        assert prog.metadata[MUTATION_CONTEXT_METADATA_KEY] == result.output.data

    async def test_multiple_contexts_combined(self):
        """Multiple inputs → composite context contains all sections."""
        stage = _make_stage()
        stage.attach_inputs(
            {
                "metrics": FloatDictContainer(data={"score": 75.0}),
                "insights": None,
                "lineage_ancestors": None,
                "lineage_descendants": None,
                "evolutionary_statistics": _make_evo_stats(),
                "formatted": StringContainer(data="Extra info"),
            }
        )
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # Should contain elements from all provided contexts
        context = result.output.data
        assert "Extra info" in context
        assert len(context) > 20  # Non-trivial composite
