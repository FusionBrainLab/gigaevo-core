"""Reflective pipeline builder for HotpotQA static chain evolution.

Extends DefaultPipelineBuilder by swapping in HotpotQAFailureFormatter
so that per-sample failure cases from validate.py are injected into the
mutation LLM's context.

Usage (Hydra config):
    pipeline_builder:
      _target_: problems.chains.hotpotqa.static.pipeline.ReflectivePipelineBuilder
"""

from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.evolution.engine.context import EvolutionContext
from gigaevo.programs.stages.base import DEFAULT_SIMPLE_STAGE_TIMEOUT

from problems.chains.hotpotqa.static.formatter import HotpotQAFailureFormatter


class ReflectivePipelineBuilder(DefaultPipelineBuilder):
    """DefaultPipelineBuilder with FormatterStage replaced by HotpotQAFailureFormatter.

    validate.py returns (metrics, failures[:10]); FetchArtifact extracts failures;
    HotpotQAFailureFormatter renders them as structured markdown; MutationContextStage
    appends the block to the mutation prompt.
    """

    def __init__(self, ctx: EvolutionContext, *, dag_timeout: float = 3600.0):
        super().__init__(ctx, dag_timeout=dag_timeout)
        self.replace_stage(
            "FormatterStage",
            lambda: HotpotQAFailureFormatter(timeout=DEFAULT_SIMPLE_STAGE_TIMEOUT),
        )
