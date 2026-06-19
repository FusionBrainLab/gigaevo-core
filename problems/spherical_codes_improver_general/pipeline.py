"""Pipeline builder for the general spherical-codes improver.

Reuses the compact artifact formatter from spherical_codes_improver (it renders
artifact["feedback_preview"] verbatim) so the mutator reads the per-dimension
gain summary instead of a repr() of the metrics dict.
"""

from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    IntraExtraMemoryPipelineBuilder,
)
from problems.spherical_codes_improver.formatter import SphericalCodesArtifactFormatter


class SphericalCodesGeneralPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline with FormatterStage replaced by a compact artifact formatter."""

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
    ):
        super().__init__(ctx, dag_timeout=dag_timeout, stage_timeout=stage_timeout)
        self.replace_stage(
            "FormatterStage",
            lambda: SphericalCodesArtifactFormatter(timeout=self._stage_timeout),
        )


class SphericalCodesGeneralMemoryPipelineBuilder(IntraExtraMemoryPipelineBuilder):
    """intra_extra_memory pipeline with FormatterStage replaced by a compact artifact formatter."""

    def __init__(self, ctx: EvolutionContext, **kwargs):
        super().__init__(ctx, **kwargs)
        self.replace_stage(
            "FormatterStage",
            lambda: SphericalCodesArtifactFormatter(timeout=self._stage_timeout),
        )
