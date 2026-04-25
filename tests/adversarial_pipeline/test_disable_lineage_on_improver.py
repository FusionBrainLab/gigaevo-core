"""Tests for disable_lineage_on_improver kwarg on AdversarialAsymmetricPipelineBuilder.

Treatment: heilbron/d-tanh-no-lineage IV 2 (PR #223). When set on improver runs,
removes LineageStage / LineagesToDescendants / LineagesFromAncestors via
PipelineBuilder.remove_stage(). The gate MUST short-circuit BEFORE
_resolve_lineage_filter is called — Volkov M2 ordering invariant. _resolve_lineage_filter
raises ValueError("lineage_filter.aggregator required") on a None/missing aggregator,
which becomes a legal config under the disable flag.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.adversarial.asymmetric_pipeline import (
    AdversarialAsymmetricPipelineBuilder,
    LineageFilterConfig,
)
from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.aggregators import ConfigurableAggregator, ReduceSpec
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec


class _FakeProvider(OpponentArchiveProvider):
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


def _make_ctx() -> EvolutionContext:
    metrics_ctx = _make_metrics_context()
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = Path("/fake/problem")
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False
    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
    )


def _aggregator(ctx: EvolutionContext) -> ConfigurableAggregator:
    return ConfigurableAggregator(
        outputs={"fitness": ReduceSpec(op="mean", field="delta")},
        invalid_defaults={"fitness": 0.0},
        metrics_context=ctx.problem_ctx.metrics_context,
    )


class TestDisableLineageOnImprover:
    """The disable_lineage_on_improver gate must short-circuit safely."""

    def test_disable_flag_removes_three_lineage_stages(self):
        """With disable=True and lineage_filter=None, the three lineage stages are absent."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=_FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
            dg_tracker=MagicMock(spec=DGImprovementTracker),
            disable_lineage_on_improver=True,
            lineage_filter=None,
        )
        nodes = set(builder._nodes.keys())
        assert "LineageStage" not in nodes
        assert "LineagesToDescendants" not in nodes
        assert "LineagesFromAncestors" not in nodes

    def test_disable_flag_does_not_call_resolve_lineage_filter(self):
        """Volkov M2: gate short-circuits BEFORE _resolve_lineage_filter is invoked.

        _resolve_lineage_filter raises ValueError("lineage_filter.aggregator required")
        when lineage_filter is None. If the gate is misordered (resolve called first),
        construction would raise. Successful construction proves correct ordering.
        """
        # No aggregator on the filter — would trip _resolve_lineage_filter's guard.
        AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=_FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
            dg_tracker=MagicMock(spec=DGImprovementTracker),
            disable_lineage_on_improver=True,
            lineage_filter=None,
        )

    def test_default_false_preserves_lineage_wiring_on_improver(self):
        """disable_lineage_on_improver defaults to False; D pipeline keeps LineageStage."""
        ctx = _make_ctx()
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=ctx,
            opponent_provider=_FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
            dg_tracker=MagicMock(spec=DGImprovementTracker),
            lineage_filter=LineageFilterConfig(
                min_shared=1,
                inject_shared_evidence=True,
                aggregator=_aggregator(ctx),
            ),
        )
        assert "LineageStage" in builder._nodes

    def test_constructor_role_unaffected_by_flag(self):
        """Treatment is D-only: G runs keep LineageStage even with flag set."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=_FakeProvider(),
            population_role="constructor",
            feedback_mode="composition",
            disable_lineage_on_improver=True,
        )
        assert "LineageStage" in builder._nodes

    def test_disable_without_dg_tracker_is_noop(self):
        """The gate lives inside `if dg_tracker is not None:` — no-op when tracker is absent."""
        builder = AdversarialAsymmetricPipelineBuilder(
            ctx=_make_ctx(),
            opponent_provider=_FakeProvider(),
            population_role="improver",
            feedback_mode="composition",
            disable_lineage_on_improver=True,
        )
        # Without dg_tracker, the lineage-removal block is never entered. The
        # default pipeline's LineageStage (added by parent builder) survives.
        assert "LineageStage" in builder._nodes
