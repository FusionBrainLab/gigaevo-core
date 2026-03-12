"""Pipeline builder for HotpotQA static_colbert_f1_600 variant.

Uses HotpotQAColBERTFormatter which shows full passage text for missing gold
documents (richer signal than titles-only in HotpotQAASIFormatter).

Usage (Hydra config):
    pipeline_builder:
      _target_: problems.chains.hotpotqa.static_colbert_f1_600.pipeline.ColBERTPipelineBuilder
"""

from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from problems.chains.hotpotqa.static_colbert_f1_600.formatter import HotpotQAColBERTFormatter


class ColBERTPipelineBuilder(DefaultPipelineBuilder):
    """DefaultPipelineBuilder with FormatterStage replaced by HotpotQAColBERTFormatter.

    validate.py returns (metrics, failures); FetchArtifact extracts all failures;
    HotpotQAColBERTFormatter renders them with full passage text for missing gold docs;
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
            lambda: HotpotQAColBERTFormatter(timeout=self._stage_timeout),
        )
