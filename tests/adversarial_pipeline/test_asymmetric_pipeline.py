"""Tests for AdversarialAsymmetricPipelineBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.adversarial.asymmetric_pipeline import AdversarialAsymmetricPipelineBuilder
from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.adversarial.pipeline import AdversarialPipelineBuilder
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(OpponentArchiveProvider):
    async def get_opponents(self, _n: int = 5) -> list[OpponentProgram]:
        return []

    async def get_top_k(
        self, _k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        return []

    async def get_programs_by_ids(self, _ids: list[str]) -> list[OpponentProgram]:
        return []

    async def get_codes_by_ids(self, _ids: list[str]) -> list[str]:
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


def _edge_pairs(builder: AdversarialAsymmetricPipelineBuilder) -> set[tuple[str, str]]:
    return {(e.source_stage, e.destination_stage) for e in builder._data_flow_edges}


def _stage_names(builder: AdversarialAsymmetricPipelineBuilder) -> set[str]:
    return set(builder._nodes.keys())


# ---------------------------------------------------------------------------
# Tests: D (Improver) runs
# ---------------------------------------------------------------------------


class TestImproverPipeline:
    def test_has_source_injection_stage(self):
        """D runs include SourceCodeInjectionStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
        )
        assert "SourceCodeInjectionStage" in _stage_names(builder)

    def test_source_injection_receives_opponent_ids(self):
        """SourceCodeInjectionStage has data flow edge from FetchOpponentIdsStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
        )
        edges = _edge_pairs(builder)
        assert ("FetchOpponentIdsStage", "SourceCodeInjectionStage") in edges

    def test_source_injection_feeds_mutation_context(self):
        """SourceCodeInjectionStage output goes to MutationContextStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
        )
        edges = _edge_pairs(builder)
        assert ("SourceCodeInjectionStage", "MutationContextStage") in edges

    def test_formatter_edge_removed_for_d(self):
        """FormatterStage → MutationContextStage edge is removed for D runs."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
        )
        edges = _edge_pairs(builder)
        assert ("FormatterStage", "MutationContextStage") not in edges

    def test_still_has_adversarial_stages(self):
        """D runs still have FetchOpponentIdsStage and FetchOpponentResultsStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
        )
        names = _stage_names(builder)
        assert "FetchOpponentIdsStage" in names
        assert "FetchOpponentResultsStage" in names

    def test_cache_on_edges_wired_for_d(self):
        """D runs wire FetchOpponentIdsStage → InsightsStage / LineageStage cache_on."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
        )
        edges = _edge_pairs(builder)
        assert ("FetchOpponentIdsStage", "InsightsStage") in edges
        assert ("FetchOpponentIdsStage", "LineageStage") in edges


# ---------------------------------------------------------------------------
# Tests: G (Constructor) runs
# ---------------------------------------------------------------------------


class TestConstructorPipeline:
    def test_no_source_injection_for_g(self):
        """G runs do NOT have SourceCodeInjectionStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="constructor",
            feedback_mode="composition",
        )
        assert "SourceCodeInjectionStage" not in _stage_names(builder)

    def test_gradient_has_gradient_stage(self):
        """G + gradient_in_prompt → GradientInPromptStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            d_provider=FakeProvider(),
            population_role="constructor",
            feedback_mode="gradient_in_prompt",
        )
        assert "GradientInPromptStage" in _stage_names(builder)

    def test_gradient_feeds_mutation_context(self):
        """GradientInPromptStage output goes to MutationContextStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            d_provider=FakeProvider(),
            population_role="constructor",
            feedback_mode="gradient_in_prompt",
        )
        edges = _edge_pairs(builder)
        assert ("GradientInPromptStage", "MutationContextStage") in edges

    def test_composition_no_gradient_stage(self):
        """G + composition → no GradientInPromptStage."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="constructor",
            feedback_mode="composition",
        )
        assert "GradientInPromptStage" not in _stage_names(builder)

    def test_gradient_without_d_provider_raises(self):
        """gradient_in_prompt on G requires d_provider."""
        with pytest.raises(ValueError, match="d_provider"):
            AdversarialAsymmetricPipelineBuilder(
                ctx=_make_ctx(),
                opponent_provider=FakeProvider(),
                population_role="constructor",
                feedback_mode="gradient_in_prompt",
            )

    def test_inherits_adversarial_pipeline(self):
        """Builder inherits from AdversarialPipelineBuilder."""
        assert issubclass(
            AdversarialAsymmetricPipelineBuilder, AdversarialPipelineBuilder
        )

    def test_g_composition_has_standard_adversarial_stages(self):
        """G composition runs retain all standard adversarial stages."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="constructor",
            feedback_mode="composition",
        )
        names = _stage_names(builder)
        assert "FetchOpponentIdsStage" in names
        assert "FetchOpponentResultsStage" in names
        assert "CallValidatorFunction" in names

    def test_cache_on_edges_wired_for_g(self):
        """G runs wire FetchOpponentIdsStage → InsightsStage / LineageStage cache_on."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=FakeProvider(),
            population_role="constructor",
            feedback_mode="composition",
        )
        edges = _edge_pairs(builder)
        assert ("FetchOpponentIdsStage", "InsightsStage") in edges
        assert ("FetchOpponentIdsStage", "LineageStage") in edges


# ---------------------------------------------------------------------------
# Tests: cache_on edge does NOT bleed into non-adversarial pipelines (C18)
# ---------------------------------------------------------------------------


class TestDefaultPipelineUnchanged:
    def test_default_pipeline_has_no_cache_on_edge(self):
        """DefaultPipelineBuilder must NOT get FetchOpponentIdsStage→InsightsStage."""
        from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder

        builder = DefaultPipelineBuilder(_make_ctx())
        edges = {
            (e.source_stage, e.destination_stage) for e in builder._data_flow_edges
        }
        assert ("FetchOpponentIdsStage", "InsightsStage") not in edges
        assert ("FetchOpponentIdsStage", "LineageStage") not in edges
