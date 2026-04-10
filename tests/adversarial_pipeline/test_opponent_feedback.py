"""Tests for OpponentFeedbackStage and AdversarialFeedbackPipelineBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.adversarial.feedback_pipeline import AdversarialFeedbackPipelineBuilder
from gigaevo.adversarial.feedback_stage import OpponentFeedbackStage
from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import Box
from gigaevo.runner.dag_blueprint import DAGBlueprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_opponent(
    program_id: str = "p1",
    code: str = "def entrypoint(): return 42",
    fitness: float = 0.5,
) -> OpponentProgram:
    return OpponentProgram(program_id=program_id, code=code, fitness=fitness)


def _make_provider(opponents: list[OpponentProgram]) -> OpponentArchiveProvider:
    provider = MagicMock(spec=OpponentArchiveProvider)
    provider.get_top_k = AsyncMock(
        side_effect=lambda k, *, higher_is_better=True: sorted(
            opponents, key=lambda o: o.fitness, reverse=higher_is_better
        )[:k]
    )
    return provider


def _make_program() -> Program:
    return Program(code="def entrypoint(): return 0")


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
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = p_dir
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = _make_metrics_context()
    problem_ctx.is_contextual = False

    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
    )


def _edge_pairs(blueprint: DAGBlueprint) -> set[tuple[str, str]]:
    return {(e.source_stage, e.destination_stage) for e in blueprint.data_flow_edges}


# ---------------------------------------------------------------------------
# OpponentFeedbackStage tests
# ---------------------------------------------------------------------------


class TestOpponentFeedbackStage:
    @pytest.mark.asyncio
    async def test_empty_archive_returns_empty_string(self):
        """When archive is empty, compute() returns StringContainer(data='')."""
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider([]),
            k=3,
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert isinstance(result, Box)
        assert result.data == ""

    @pytest.mark.asyncio
    async def test_constructor_role_contains_attack_report_header(self):
        """Constructor role report includes OPPONENT ATTACK REPORT header."""
        opponents = [_make_opponent(program_id="p1", fitness=0.9)]
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider(opponents),
            k=1,
            role="constructor",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert isinstance(result, Box)
        assert "OPPONENT ATTACK REPORT" in result.data

    @pytest.mark.asyncio
    async def test_constructor_role_contains_opponent_code(self):
        """Constructor role report embeds opponent source code."""
        code = "def entrypoint(): return [0.1, 0.2, 0.3]"
        opponents = [_make_opponent(program_id="p1", code=code, fitness=0.7)]
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider(opponents),
            k=1,
            role="constructor",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert code in result.data

    @pytest.mark.asyncio
    async def test_fewer_opponents_than_k_still_works(self):
        """When only 2 opponents are available and k=3, report contains 2 entries."""
        opponents = [
            _make_opponent(program_id="p1", fitness=0.8),
            _make_opponent(program_id="p2", fitness=0.5),
        ]
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider(opponents),
            k=3,
            role="constructor",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert isinstance(result, Box)
        assert result.data != ""
        # Both opponents appear (indexed as 1 and 2)
        assert "Opponent 1" in result.data
        assert "Opponent 2" in result.data
        assert "Opponent 3" not in result.data

    @pytest.mark.asyncio
    async def test_improver_role_contains_target_analysis_header(self):
        """Improver role report includes TARGET ANALYSIS REPORT header."""
        opponents = [_make_opponent(program_id="p1", fitness=0.6)]
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider(opponents),
            k=1,
            role="improver",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert isinstance(result, Box)
        assert "TARGET ANALYSIS REPORT" in result.data

    @pytest.mark.asyncio
    async def test_improver_role_does_not_contain_attack_report_header(self):
        """Improver role report does NOT include OPPONENT ATTACK REPORT header."""
        opponents = [_make_opponent(program_id="p1", fitness=0.6)]
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider(opponents),
            k=1,
            role="improver",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert "OPPONENT ATTACK REPORT" not in result.data

    @pytest.mark.asyncio
    async def test_selects_top_k_by_fitness(self):
        """With 5 opponents and k=2, the 2 highest-fitness opponents are returned."""
        opponents = [
            _make_opponent(program_id="low1", code="# low1", fitness=0.1),
            _make_opponent(program_id="high1", code="# high1", fitness=0.9),
            _make_opponent(program_id="mid", code="# mid", fitness=0.5),
            _make_opponent(program_id="low2", code="# low2", fitness=0.2),
            _make_opponent(program_id="high2", code="# high2", fitness=0.8),
        ]
        # Provider returns top-2 by fitness (get_top_k called with k=2)
        provider = MagicMock(spec=OpponentArchiveProvider)
        provider.get_top_k = AsyncMock(
            side_effect=lambda k, *, higher_is_better=True: sorted(
                opponents, key=lambda o: o.fitness, reverse=higher_is_better
            )[:k]
        )

        stage = OpponentFeedbackStage(
            opponent_provider=provider,
            k=2,
            role="constructor",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())

        assert "# high1" in result.data
        assert "# high2" in result.data
        assert "# low1" not in result.data
        assert "# low2" not in result.data
        assert "# mid" not in result.data

    def test_uses_no_cache(self):
        """Stage must use NO_CACHE handler."""
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider([]),
            timeout=60.0,
        )
        assert stage.cache_handler is NO_CACHE

    @pytest.mark.asyncio
    async def test_fitness_value_appears_in_report(self):
        """Each opponent block includes its fitness value."""
        opponents = [_make_opponent(program_id="p1", fitness=0.12345)]
        stage = OpponentFeedbackStage(
            opponent_provider=_make_provider(opponents),
            k=1,
            role="constructor",
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert "0.12345" in result.data


# ---------------------------------------------------------------------------
# AdversarialFeedbackPipelineBuilder tests
# ---------------------------------------------------------------------------


class TestAdversarialFeedbackPipelineBuilder:
    def test_removes_formatter_to_mutation_context_edge(self):
        """FormatterStage→MutationContextStage edge is removed."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx, opponent_provider=_make_provider([])
        ).build_blueprint()
        edges = _edge_pairs(bp)
        assert ("FormatterStage", "MutationContextStage") not in edges

    def test_adds_opponent_feedback_to_mutation_context_edge(self):
        """OpponentFeedbackStage→MutationContextStage edge is present."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx, opponent_provider=_make_provider([])
        ).build_blueprint()
        edges = _edge_pairs(bp)
        assert ("OpponentFeedbackStage", "MutationContextStage") in edges

    def test_adds_opponent_feedback_stage(self):
        """OpponentFeedbackStage is present in the blueprint."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx, opponent_provider=_make_provider([])
        ).build_blueprint()
        assert "OpponentFeedbackStage" in bp.nodes

    def test_feedback_stage_factory_is_callable(self):
        """OpponentFeedbackStage factory can be called to produce a stage instance."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx, opponent_provider=_make_provider([])
        ).build_blueprint()
        factory = bp.nodes["OpponentFeedbackStage"]
        assert callable(factory)
        stage = factory()
        assert isinstance(stage, OpponentFeedbackStage)

    def test_feedback_stage_depends_on_validate_code(self):
        """OpponentFeedbackStage has exec dep on ValidateCodeStage."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx, opponent_provider=_make_provider([])
        ).build_blueprint()
        if bp.exec_order_deps is None:
            pytest.fail("exec_order_deps is None — no deps registered")
        dep_names = {
            d.stage_name for d in bp.exec_order_deps.get("OpponentFeedbackStage", [])
        }
        assert "ValidateCodeStage" in dep_names

    def test_inherits_base_adversarial_stages(self):
        """All stages from AdversarialPipelineBuilder are still present."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx, opponent_provider=_make_provider([])
        ).build_blueprint()
        assert "FetchOpponentResultsStage" in bp.nodes
        assert "CallValidatorFunction" in bp.nodes
        assert "MutationContextStage" in bp.nodes

    def test_custom_k_and_role_passed_through(self):
        """k and role are forwarded to OpponentFeedbackStage."""
        ctx = _make_ctx()
        bp = AdversarialFeedbackPipelineBuilder(
            ctx,
            opponent_provider=_make_provider([]),
            opponent_feedback_k=5,
            population_role="improver",
        ).build_blueprint()
        stage = bp.nodes["OpponentFeedbackStage"]()
        assert stage._k == 5
        assert stage._role == "improver"
