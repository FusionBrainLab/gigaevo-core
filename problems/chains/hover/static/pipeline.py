"""Feedback pipeline builder for HoVer static chain evolution.

Extends DefaultPipelineBuilder by swapping in HoVerFeedbackFormatter so that
per-hop retrieval failure diagnostics from validate.py are injected into the
mutation LLM's context.

Usage (Hydra config):
    pipeline_builder:
      _target_: problems.chains.hover.static.pipeline.HoVerFeedbackPipelineBuilder
"""

from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from problems.chains.hover.static.formatter import HoVerFeedbackFormatter


class HoVerFeedbackPipelineBuilder(DefaultPipelineBuilder):
    """DefaultPipelineBuilder with FormatterStage replaced by HoVerFeedbackFormatter.

    validate.py returns (metrics, failures); FetchArtifact extracts all failures;
    HoVerFeedbackFormatter renders them with per-hop retrieval diagnostics;
    MutationContextStage appends the block to the mutation prompt.
    """

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
            lambda: HoVerFeedbackFormatter(timeout=self._stage_timeout),
        )
