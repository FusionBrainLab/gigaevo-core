from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from gigaevo.entrypoint.default_pipelines import PipelineBuilder
    from gigaevo.entrypoint.evolution_context import EvolutionContext


class _PipelineBuilderBase(FrozenStrictModel):
    """Common pipeline-builder knobs. ``dag_timeout`` caps the entire
    program-evaluation DAG; per-stage timeouts are owned by individual
    stages or default to the builder's ``stage_timeout`` when the
    runtime builder consults it."""

    dag_timeout: float = Field(default=3600.0, gt=0.0)
    stage_timeout: float = Field(default=300.0, gt=0.0)


class DefaultPipelineBuilderConfig(_PipelineBuilderBase):
    """Vanilla pipeline: validate, call program function, call validator,
    fetch metrics, merge. No problem-context stage. Used by problems
    whose evaluator does not need an injected context dict."""

    kind: Literal["default"] = "default"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder

        return DefaultPipelineBuilder(
            ctx, dag_timeout=self.dag_timeout, stage_timeout=self.stage_timeout
        )


class ContextPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + AddContext stage. Reads context.py from
    the problem directory and feeds the resulting dict into the program
    and validator stages as the ``context`` data-flow input."""

    kind: Literal["context"] = "context"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.default_pipelines import ContextPipelineBuilder

        return ContextPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class AlgoTuneSpeedPipelineBuilderConfig(_PipelineBuilderBase):
    """ContextPipelineBuilder + RuntimeFitnessStage. Used for AlgoTune
    problems where the primary fitness is execution speed measured via
    repeated timing of the candidate program."""

    kind: Literal["algotune_speed"] = "algotune_speed"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.default_pipelines import (
            AlgoTuneSpeedPipelineBuilder,
        )

        return AlgoTuneSpeedPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class CMAOptPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + CMA-ES numerical constant optimisation
    stage inserted between validate and program-call. CMA hyperparameters
    are owned by the runtime subclass ClassVars; tweaking them here
    requires subclassing the runtime builder."""

    kind: Literal["cma_opt"] = "cma_opt"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.default_pipelines import CMAOptPipelineBuilder

        return CMAOptPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class OptunaOptPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + Optuna numerical constant optimisation
    stage. Mirrors the CMA variant but uses Optuna's TPE sampler."""

    kind: Literal["optuna_opt"] = "optuna_opt"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.default_pipelines import OptunaOptPipelineBuilder

        return OptunaOptPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class StructuralMetricsPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + StructuralMetricsStage that emits AST
    structural features (dag_depth, max_dependency_fan_in, etc.) used
    as behavior-space coordinates by the topology_3d algorithm variants."""

    kind: Literal["structural_metrics"] = "structural_metrics"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.experiment_pipelines import (
            StructuralMetricsPipelineBuilder,
        )

        return StructuralMetricsPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class AutoPipelineBuilderConfig(_PipelineBuilderBase):
    """Picks ContextPipelineBuilder when the problem declares
    is_contextual, DefaultPipelineBuilder otherwise. Mirrors the runtime
    select_pipeline_builder dispatch."""

    kind: Literal["auto"] = "auto"

    def build(self, ctx: "EvolutionContext") -> "PipelineBuilder":
        from gigaevo.entrypoint.default_pipelines import (
            ContextPipelineBuilder,
            DefaultPipelineBuilder,
        )

        if ctx.problem_ctx.is_contextual:
            return ContextPipelineBuilder(ctx, dag_timeout=self.dag_timeout)
        return DefaultPipelineBuilder(
            ctx, dag_timeout=self.dag_timeout, stage_timeout=self.stage_timeout
        )


PipelineBuilderConfig = Annotated[
    DefaultPipelineBuilderConfig
    | ContextPipelineBuilderConfig
    | AlgoTuneSpeedPipelineBuilderConfig
    | CMAOptPipelineBuilderConfig
    | OptunaOptPipelineBuilderConfig
    | StructuralMetricsPipelineBuilderConfig
    | AutoPipelineBuilderConfig,
    Field(discriminator="kind"),
]


class PipelineConfig(FrozenStrictModel):
    """Pipeline section of the experiment config. Holds the builder
    selection plus ``prompts_dir`` for prompt-template lookup; the
    EvolutionContext is constructed by ``build_object_graph`` from the
    other resolved subtrees and passed to ``builder.build()``."""

    builder: PipelineBuilderConfig
    prompts_dir: Path | None = None
