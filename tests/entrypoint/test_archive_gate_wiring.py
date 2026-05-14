"""DefaultPipelineBuilder wires ArchivePotentialGateStage when enabled.

- archive_gate_enabled=False (default): no gate node, InsightsStage deps unchanged.
- archive_gate_enabled=True: gate node present with:
    * on_success(CallValidatorFunction)
    * always_after(EnsureMetricsStage)
  and InsightsStage gains on_success(ArchivePotentialGateStage) so any
  ``run_insights == False`` skip cascades into Insights and (via existing
  always_after) into MutationContextStage.

The gate uses ``ctx.archive_gate_provider`` (None ⇒ fail open inside the stage).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.default_pipelines import (
    ContextPipelineBuilder,
    DefaultPipelineBuilder,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.stages.archive_gate import (
    ArchiveGateProvider,
    ArchivePotentialGateStage,
)

GATE_NAME = "ArchivePotentialGateStage"


def _make_ctx(*, archive_gate_provider=None) -> EvolutionContext:
    metrics_ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="m",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = Path("/fake/problem")
    problem_ctx.task_description = "x"
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False
    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
        archive_gate_provider=archive_gate_provider,
    )


def _deps_for(blueprint, stage: str) -> set[tuple[str, str]]:
    """Return the set of (dep_stage, condition) tuples for a given stage."""
    if blueprint.exec_order_deps is None:
        return set()
    return {
        (d.stage_name, d.condition) for d in blueprint.exec_order_deps.get(stage, [])
    }


def test_gate_disabled_by_default():
    bp = DefaultPipelineBuilder(_make_ctx()).build_blueprint()
    assert GATE_NAME not in bp.nodes
    insights_deps = _deps_for(bp, "InsightsStage")
    assert (GATE_NAME, "success") not in insights_deps


def test_gate_enabled_adds_node_and_deps():
    bp = DefaultPipelineBuilder(
        _make_ctx(), archive_gate_enabled=True
    ).build_blueprint()
    assert GATE_NAME in bp.nodes

    factory = bp.nodes[GATE_NAME]
    assert isinstance(factory(), ArchivePotentialGateStage)

    gate_deps = _deps_for(bp, GATE_NAME)
    assert ("CallValidatorFunction", "success") in gate_deps
    assert ("EnsureMetricsStage", "always") in gate_deps

    insights_deps = _deps_for(bp, "InsightsStage")
    assert (GATE_NAME, "success") in insights_deps
    # Existing deps preserved
    assert ("CallValidatorFunction", "success") in insights_deps
    assert ("EnsureMetricsStage", "always") in insights_deps


def test_gate_enabled_uses_ctx_provider():
    provider_sentinel = MagicMock(spec=ArchiveGateProvider)
    ctx = _make_ctx(archive_gate_provider=provider_sentinel)
    bp = DefaultPipelineBuilder(ctx, archive_gate_enabled=True).build_blueprint()
    stage = bp.nodes[GATE_NAME]()
    assert stage._provider is provider_sentinel


def test_context_pipeline_builder_propagates_kwarg():
    bp = ContextPipelineBuilder(
        _make_ctx(), archive_gate_enabled=True
    ).build_blueprint()
    assert GATE_NAME in bp.nodes
    assert (GATE_NAME, "success") in _deps_for(bp, "InsightsStage")


def test_subclasses_default_to_disabled():
    """Subclasses must not silently enable the gate."""
    bp = ContextPipelineBuilder(_make_ctx()).build_blueprint()
    assert GATE_NAME not in bp.nodes


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
