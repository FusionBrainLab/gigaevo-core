"""One-liner builders for the shipped pipeline variants.

Each function returns a fully-validated :class:`PipelineConfig`
wrapping one of the discriminated-union variants in
:mod:`gigaevo.config.schemas.pipeline`. Seven cover the
language-level builders; four wrap problem-specific Python builders
shipped under ``problems/chains/{hotpotqa,hover}/``.

``stage_timeout`` is a knob on the Default and Auto variants only;
their runtime ``DefaultPipelineBuilder`` is the single backing that
accepts a separate per-stage budget. Builders that delegate to
context, AlgoTune-speed, CMA, Optuna, structural-metrics, or
problem-specific runtime classes consume only ``dag_timeout``; the
presets reflect that surface.
"""

from __future__ import annotations

from pathlib import Path

from gigaevo.config.defaults import DEFAULT_DAG_TIMEOUT_S, DEFAULT_STAGE_TIMEOUT_S
from gigaevo.config.schemas import (
    AlgoTuneSpeedPipelineBuilderConfig,
    AutoPipelineBuilderConfig,
    CMAOptPipelineBuilderConfig,
    ContextPipelineBuilderConfig,
    DefaultPipelineBuilderConfig,
    OptunaOptPipelineBuilderConfig,
    PipelineConfig,
    StructuralMetricsPipelineBuilderConfig,
)
from gigaevo.config.schemas.pipeline import ProblemSpecificPipelineBuilderConfig


# ---------------------------------------------------------------------------
# Standard pipeline presets
# ---------------------------------------------------------------------------


def build_standard(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """Vanilla validate / call / fetch / merge pipeline."""
    return PipelineConfig(
        builder=DefaultPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_with_context(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """Default pipeline + ``AddContext`` stage. Used when the problem
    ships a ``context.py`` providing ``build_context(problem)``."""
    return PipelineConfig(
        builder=ContextPipelineBuilderConfig(dag_timeout=dag_timeout),
        prompts_dir=prompts_dir,
    )


def build_auto(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """Runtime-dispatch pipeline. Picks ``ContextPipelineBuilder`` when
    the problem declares ``is_contextual``, ``DefaultPipelineBuilder``
    otherwise."""
    return PipelineConfig(
        builder=AutoPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_algotune_speed(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """``ContextPipelineBuilder`` + ``RuntimeFitnessStage``."""
    return PipelineConfig(
        builder=AlgoTuneSpeedPipelineBuilderConfig(dag_timeout=dag_timeout),
        prompts_dir=prompts_dir,
    )


def build_cma_opt(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """``DefaultPipelineBuilder`` + CMA-ES numerical constant
    optimisation stage."""
    return PipelineConfig(
        builder=CMAOptPipelineBuilderConfig(dag_timeout=dag_timeout),
        prompts_dir=prompts_dir,
    )


def build_optuna_opt(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """``DefaultPipelineBuilder`` + Optuna constant optimisation
    stage."""
    return PipelineConfig(
        builder=OptunaOptPipelineBuilderConfig(dag_timeout=dag_timeout),
        prompts_dir=prompts_dir,
    )


def build_structural_metrics(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """``DefaultPipelineBuilder`` + ``StructuralMetricsStage`` emitting
    AST structural features used as behavior-space coordinates by the
    topology-3d algorithm variants."""
    return PipelineConfig(
        builder=StructuralMetricsPipelineBuilderConfig(dag_timeout=dag_timeout),
        prompts_dir=prompts_dir,
    )


# ---------------------------------------------------------------------------
# Problem-specific builders (hotpotqa / hover variants)
# ---------------------------------------------------------------------------


def build_hotpotqa_reflective(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HotpotQA reflective pipeline; lazy-imports the builder under
    :mod:`problems.chains.hotpotqa.static.pipeline`."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hotpotqa.static.pipeline."
                "ReflectivePipelineBuilder"
            ),
            dag_timeout=dag_timeout,
        ),
        prompts_dir=prompts_dir,
    )


def build_hotpotqa_asi(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HotpotQA ASI pipeline; lazy-imports the builder under
    :mod:`problems.chains.hotpotqa.static_a.pipeline`."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hotpotqa.static_a.pipeline."
                "ASIPipelineBuilder"
            ),
            dag_timeout=dag_timeout,
        ),
        prompts_dir=prompts_dir,
    )


def build_hotpotqa_colbert(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HotpotQA ColBERT pipeline; lazy-imports the builder under
    :mod:`problems.chains.hotpotqa.static_colbert_f1_600.pipeline`."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hotpotqa.static_colbert_f1_600.pipeline."
                "ColBERTPipelineBuilder"
            ),
            dag_timeout=dag_timeout,
        ),
        prompts_dir=prompts_dir,
    )


def build_hover_feedback(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HoVer feedback pipeline; lazy-imports the builder under
    :mod:`problems.chains.hover.static.pipeline`."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hover.static.pipeline."
                "HoVerFeedbackPipelineBuilder"
            ),
            dag_timeout=dag_timeout,
        ),
        prompts_dir=prompts_dir,
    )


__all__: list[str] = [
    "ProblemSpecificPipelineBuilderConfig",
    "build_algotune_speed",
    "build_auto",
    "build_cma_opt",
    "build_hotpotqa_asi",
    "build_hotpotqa_colbert",
    "build_hotpotqa_reflective",
    "build_hover_feedback",
    "build_optuna_opt",
    "build_standard",
    "build_structural_metrics",
    "build_with_context",
]
