"""Preset builders for the shipped pipeline YAMLs.

Each builder returns a fully-validated :class:`PipelineConfig`
matching one ``config/pipeline/*.yaml`` shape. Seven builders cover
the standard variants (matching the discriminated union from
hydra-1.6); four cover the problem-specific Python builders that the
hotpotqa / hover YAMLs target.

The declarative ``custom.yaml`` (25 ``_target_`` references
constructing a DAGBlueprint directly) is intentionally not migrated
here — it's a one-off Hydra-only escape hatch and the architectural
direction is to express any new pipeline as a Python builder
function rather than a YAML DAG.
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
# Standard pipeline presets — wrap the seven shipped schema variants
# ---------------------------------------------------------------------------


def build_standard(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """Vanilla validate / call / fetch / merge pipeline matching
    ``config/pipeline/standard.yaml``."""
    return PipelineConfig(
        builder=DefaultPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_with_context(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """Default pipeline + AddContext stage matching
    ``config/pipeline/with_context.yaml`` — used when the problem
    ships a ``context.py`` providing build_context(problem)."""
    return PipelineConfig(
        builder=ContextPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_auto(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """Runtime-dispatch pipeline matching ``config/pipeline/auto.yaml``.
    Picks ContextPipelineBuilder when the problem declares
    is_contextual, DefaultPipelineBuilder otherwise."""
    return PipelineConfig(
        builder=AutoPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_algotune_speed(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """ContextPipelineBuilder + RuntimeFitnessStage matching
    ``config/pipeline/algotune_speed.yaml``."""
    return PipelineConfig(
        builder=AlgoTuneSpeedPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_cma_opt(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """DefaultPipelineBuilder + CMA-ES numerical constant optimisation
    matching ``config/pipeline/cma_opt.yaml``."""
    return PipelineConfig(
        builder=CMAOptPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_optuna_opt(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """DefaultPipelineBuilder + Optuna constant optimisation matching
    ``config/pipeline/optuna_opt.yaml``."""
    return PipelineConfig(
        builder=OptunaOptPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


def build_structural_metrics(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """DefaultPipelineBuilder + StructuralMetricsStage matching
    ``config/pipeline/structural_metrics.yaml``. The stage emits AST
    structural features used as behavior-space coordinates by the
    topology_3d algorithm variants."""
    return PipelineConfig(
        builder=StructuralMetricsPipelineBuilderConfig(
            dag_timeout=dag_timeout, stage_timeout=stage_timeout
        ),
        prompts_dir=prompts_dir,
    )


# ---------------------------------------------------------------------------
# Problem-specific builders (hotpotqa / hover variants)
# ---------------------------------------------------------------------------


def build_hotpotqa_reflective(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HotpotQA reflective pipeline (problem-specific builder under
    ``problems.chains.hotpotqa.static.pipeline``) matching
    ``config/pipeline/hotpotqa_reflective.yaml``."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hotpotqa.static.pipeline."
                "ReflectivePipelineBuilder"
            ),
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
        ),
        prompts_dir=prompts_dir,
    )


def build_hotpotqa_asi(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HotpotQA ASI pipeline matching
    ``config/pipeline/hotpotqa_asi.yaml``."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hotpotqa.static_a.pipeline."
                "ASIPipelineBuilder"
            ),
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
        ),
        prompts_dir=prompts_dir,
    )


def build_hotpotqa_colbert(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HotpotQA ColBERT pipeline matching
    ``config/pipeline/hotpotqa_colbert.yaml``."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hotpotqa.static_colbert_f1_600.pipeline."
                "ColBERTPipelineBuilder"
            ),
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
        ),
        prompts_dir=prompts_dir,
    )


def build_hover_feedback(
    *,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT_S,
    prompts_dir: Path | None = None,
) -> PipelineConfig:
    """HoVer feedback pipeline (problem-specific builder under
    ``problems.chains.hover.static.pipeline``) matching
    ``config/pipeline/hover_feedback.yaml``."""
    return PipelineConfig(
        builder=ProblemSpecificPipelineBuilderConfig(
            builder_path=(
                "problems.chains.hover.static.pipeline."
                "HoVerFeedbackPipelineBuilder"
            ),
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
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
