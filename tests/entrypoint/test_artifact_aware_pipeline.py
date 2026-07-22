"""Standard pipelines route reserved validator artifact metadata."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    GuidedMutationPipelineBuilder,
    MemoryGuidedMutationPipelineBuilder,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.provider import MemoryProvider
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.stages.validator_metadata import ProgramMetadataValidatorStage


def _make_ctx(problem_dir) -> EvolutionContext:
    (problem_dir / "validate.py").write_text(
        "def validate(payload):\n    return {'fitness': 0.0}\n"
    )
    metrics_ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="main metric",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = problem_dir
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False
    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
        prompts_dir=None,
        memory_provider=MagicMock(spec=MemoryProvider),
    )


@pytest.mark.parametrize(
    "builder_type",
    [GuidedMutationPipelineBuilder, MemoryGuidedMutationPipelineBuilder],
)
def test_standard_builder_uses_artifact_aware_validator(tmp_path, builder_type):
    blueprint = builder_type(_make_ctx(tmp_path)).build_blueprint()

    stage = blueprint.nodes["CallValidatorFunction"]()

    assert isinstance(stage, ProgramMetadataValidatorStage)


def test_archive_gate_keeps_artifact_aware_validator(tmp_path):
    blueprint = MemoryGuidedMutationPipelineBuilder(
        _make_ctx(tmp_path), archive_gate_enabled=True
    ).build_blueprint()

    assert "ArchivePotentialGateStage" in blueprint.nodes
    assert isinstance(
        blueprint.nodes["CallValidatorFunction"](), ProgramMetadataValidatorStage
    )
