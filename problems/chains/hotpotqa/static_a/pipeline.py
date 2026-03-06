"""ASI pipeline builder for HotpotQA static_a chain evolution.

Extends DefaultPipelineBuilder by swapping in HotpotQAASIFormatter so that
per-sample failure cases with per-hop retrieval diagnostics from static_a/validate.py
are injected into the mutation LLM's context.

Usage (Hydra config):
    pipeline_builder:
      _target_: problems.chains.hotpotqa.static_a.pipeline.ASIPipelineBuilder
"""

from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from problems.chains.hotpotqa.static_a.formatter import HotpotQAASIFormatter


class ASIPipelineBuilder(DefaultPipelineBuilder):
    """DefaultPipelineBuilder with FormatterStage replaced by HotpotQAASIFormatter.

    validate.py returns (metrics, failures[:10]); FetchArtifact extracts failures;
    HotpotQAASIFormatter renders them with per-hop retrieval diagnostics;
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
            lambda: HotpotQAASIFormatter(timeout=self._stage_timeout),
        )
