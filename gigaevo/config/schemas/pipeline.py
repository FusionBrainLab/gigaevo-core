from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, field_validator

from gigaevo.config.schemas._base import FrozenStrictModel, reject_empty_or_cwd_path

if TYPE_CHECKING:
    from gigaevo.entrypoint.default_pipelines import PipelineBuilder
    from gigaevo.entrypoint.evolution_context import EvolutionContext


class _PipelineBuilderBase(FrozenStrictModel):
    """Common pipeline-builder knob. ``dag_timeout`` caps the entire
    program-evaluation DAG; per-stage timeouts are owned either by
    the individual stages or by the variants whose runtime builder
    accepts a separate ``stage_timeout`` argument.

    The ``dag_timeout`` default agrees with
    :data:`gigaevo.config.defaults.DEFAULT_DAG_TIMEOUT_S` so direct
    schema construction matches the pipeline preset builders."""

    dag_timeout: float = Field(default=7200.0, gt=0.0)


class DefaultPipelineBuilderConfig(_PipelineBuilderBase):
    """Vanilla pipeline: validate, call program function, call validator,
    fetch metrics, merge. No problem-context stage. Used by problems
    whose evaluator does not need an injected context dict.

    ``stage_timeout`` lives on this variant (and on
    :class:`AutoPipelineBuilderConfig`) because the runtime
    ``DefaultPipelineBuilder`` is the only builder whose constructor
    accepts a ``stage_timeout`` argument; the other variants delegate
    to their own per-stage budgets and would silently drop the value
    if it appeared on the base. The default matches
    :data:`gigaevo.config.defaults.DEFAULT_STAGE_TIMEOUT_S`."""

    kind: Literal["default"] = "default"
    stage_timeout: float = Field(default=2400.0, gt=0.0)

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder

        return DefaultPipelineBuilder(
            ctx, dag_timeout=self.dag_timeout, stage_timeout=self.stage_timeout
        )


class ContextPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + AddContext stage. Reads context.py from
    the problem directory and feeds the resulting dict into the program
    and validator stages as the ``context`` data-flow input."""

    kind: Literal["context"] = "context"

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        from gigaevo.entrypoint.default_pipelines import ContextPipelineBuilder

        return ContextPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class AlgoTuneSpeedPipelineBuilderConfig(_PipelineBuilderBase):
    """ContextPipelineBuilder + RuntimeFitnessStage. Used for AlgoTune
    problems where the primary fitness is execution speed measured via
    repeated timing of the candidate program."""

    kind: Literal["algotune_speed"] = "algotune_speed"

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
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

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        from gigaevo.entrypoint.default_pipelines import CMAOptPipelineBuilder

        return CMAOptPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class OptunaOptPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + Optuna numerical constant optimisation
    stage. Mirrors the CMA variant but uses Optuna's TPE sampler."""

    kind: Literal["optuna_opt"] = "optuna_opt"

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        from gigaevo.entrypoint.default_pipelines import OptunaOptPipelineBuilder

        return OptunaOptPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class StructuralMetricsPipelineBuilderConfig(_PipelineBuilderBase):
    """DefaultPipelineBuilder + StructuralMetricsStage that emits AST
    structural features (dag_depth, max_dependency_fan_in, etc.) used
    as behavior-space coordinates by the topology_3d algorithm variants."""

    kind: Literal["structural_metrics"] = "structural_metrics"

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        from gigaevo.entrypoint.experiment_pipelines import (
            StructuralMetricsPipelineBuilder,
        )

        return StructuralMetricsPipelineBuilder(ctx, dag_timeout=self.dag_timeout)


class AutoPipelineBuilderConfig(_PipelineBuilderBase):
    """Picks ContextPipelineBuilder when the problem declares
    is_contextual, DefaultPipelineBuilder otherwise.

    ``stage_timeout`` is consulted only on the Default branch — the
    runtime ContextPipelineBuilder constructor does not accept a
    ``stage_timeout`` argument and reuses the default. The default
    matches :data:`gigaevo.config.defaults.DEFAULT_STAGE_TIMEOUT_S`."""

    kind: Literal["auto"] = "auto"
    stage_timeout: float = Field(default=2400.0, gt=0.0)

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        from gigaevo.entrypoint.default_pipelines import (
            ContextPipelineBuilder,
            DefaultPipelineBuilder,
        )

        if ctx.problem_ctx.is_contextual:
            return ContextPipelineBuilder(ctx, dag_timeout=self.dag_timeout)
        return DefaultPipelineBuilder(
            ctx, dag_timeout=self.dag_timeout, stage_timeout=self.stage_timeout
        )


class ProblemSpecificPipelineBuilderConfig(_PipelineBuilderBase):
    """Builder dispatched by a fully-qualified dotted import path.

    Used when a problem ships its pipeline builder inside its own
    directory rather than under
    :mod:`gigaevo.entrypoint.default_pipelines`. The
    ``builder_path`` is the dotted import path of the builder class;
    :meth:`build` imports the module lazily and invokes the class
    with ``(ctx, dag_timeout=...)``.

    Lazy import keeps the schema layer free of cross-package
    dependencies — the problem-specific module is only loaded when
    the experiment selects this variant."""

    kind: Literal["problem_specific"] = "problem_specific"
    builder_path: str = Field(min_length=1)

    def build(self, ctx: EvolutionContext) -> PipelineBuilder:
        import importlib

        module_name, _, class_name = self.builder_path.rpartition(".")
        if not module_name or not class_name:
            raise ValueError(
                f"builder_path must be a fully-qualified dotted name "
                f"with at least one '.'; got {self.builder_path!r}"
            )
        module = importlib.import_module(module_name)
        builder_class = getattr(module, class_name, None)
        if builder_class is None:
            raise ImportError(
                f"{module_name} does not export {class_name!r}; "
                f"check the builder_path on the pipeline preset"
            )
        return builder_class(ctx, dag_timeout=self.dag_timeout)


PipelineBuilderConfig = Annotated[
    DefaultPipelineBuilderConfig
    | ContextPipelineBuilderConfig
    | AlgoTuneSpeedPipelineBuilderConfig
    | CMAOptPipelineBuilderConfig
    | OptunaOptPipelineBuilderConfig
    | StructuralMetricsPipelineBuilderConfig
    | AutoPipelineBuilderConfig
    | ProblemSpecificPipelineBuilderConfig,
    Field(discriminator="kind"),
]


class PipelineConfig(FrozenStrictModel):
    """Pipeline section of the experiment config. Holds the builder
    selection plus ``prompts_dir`` for prompt-template lookup; the
    EvolutionContext is constructed by ``build_object_graph`` from the
    other resolved subtrees and passed to ``builder.build()``."""

    builder: PipelineBuilderConfig
    prompts_dir: Path | None = None

    @field_validator("prompts_dir")
    @classmethod
    def _prompts_dir_not_empty(cls, value: Path | None) -> Path | None:
        return reject_empty_or_cwd_path("prompts_dir", value)
