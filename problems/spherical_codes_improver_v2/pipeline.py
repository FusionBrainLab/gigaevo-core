"""Pipeline builder for spherical_codes artifact-aware feedback."""

from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from problems.spherical_codes_improver.formatter import SphericalCodesArtifactFormatter


class SphericalCodesFeedbackPipelineBuilder(DefaultPipelineBuilder):
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
