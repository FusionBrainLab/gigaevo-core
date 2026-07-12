"""Noise-aware pipeline: per-sample score transport for paired selection."""

from __future__ import annotations

from typing import Any

from gigaevo.entrypoint.constants import MAX_MEMORY_MB, MAX_OUTPUT_SIZE
from gigaevo.entrypoint.default_pipelines import PipelineBuilder, PipelineFeature
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    GuidedMutationPipelineBuilder,
    MemoryGuidedMutationPipelineBuilder,
)
from gigaevo.programs.stages.validator_metadata import ProgramMetadataValidatorStage


class ProgramMetadataValidatorFeature(PipelineFeature):
    """Swap ``CallValidatorFunction`` for the metadata-routing subclass.

    Same node name, same edges, same exec deps — only the stage class changes,
    so the DAG is wire-identical to the parent pipeline. The subclass pops
    ``artifact["_program_metadata"]`` at eval time: per-sample vectors sit on
    ``program.metadata`` before the archive gate / insertion consult the
    selector, and never enter the artifact stream (prompt parity with control).
    """

    name = "program_metadata_validator"
    description = (
        "Route artifact['_program_metadata'] to program.metadata at eval time."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        validator_path = builder.ctx.problem_ctx.problem_dir / "validate.py"
        stage_timeout = builder._stage_timeout
        builder.replace_stage(
            "CallValidatorFunction",
            lambda: ProgramMetadataValidatorStage(
                path=validator_path,
                function_name="validate",
                timeout=stage_timeout,
                max_memory_mb=MAX_MEMORY_MB,
                max_output_size=MAX_OUTPUT_SIZE,
            ),
        )


class NoiseAwareMemoryGuidedPipelineBuilder(MemoryGuidedMutationPipelineBuilder):
    """``pipeline=memory_guided_noise``: memory-guided DAG + metadata validator.

    Identical to :class:`MemoryGuidedMutationPipelineBuilder` except the
    validator stage routes the reserved artifact namespace to
    ``program.metadata``; pair with ``PairedBootstrapArchiveSelector``
    (islands.N.archive_selector) and a ``validate()`` that emits
    ``per_sample_scores`` for noise-aware archive replacement.
    """

    def __init__(self, ctx: EvolutionContext, **kwargs: Any):
        super().__init__(ctx, **kwargs)
        self.apply_feature(ProgramMetadataValidatorFeature())


class NoiseAwareGuidedPipelineBuilder(GuidedMutationPipelineBuilder):
    """``pipeline=guided_noise``: guided (no external memory) DAG + metadata validator.

    The no-memory sibling of :class:`NoiseAwareMemoryGuidedPipelineBuilder`,
    for control arms that need the same per-sample score transport without
    memory-card retrieval.
    """

    def __init__(self, ctx: EvolutionContext, **kwargs: Any):
        super().__init__(ctx, **kwargs)
        self.apply_feature(ProgramMetadataValidatorFeature())
