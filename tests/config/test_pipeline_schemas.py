"""Unit tests for the pipeline-side typed schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    AlgoTuneSpeedPipelineBuilderConfig,
    AutoPipelineBuilderConfig,
    CMAOptPipelineBuilderConfig,
    ContextPipelineBuilderConfig,
    DefaultPipelineBuilderConfig,
    OptunaOptPipelineBuilderConfig,
    PipelineBuilderConfig,
    PipelineConfig,
    ProblemSpecificPipelineBuilderConfig,
    StructuralMetricsPipelineBuilderConfig,
)


class TestPipelineBuilderConfigs:
    def test_default_construct(self) -> None:
        from gigaevo.config.defaults import (
            DEFAULT_DAG_TIMEOUT_S,
            DEFAULT_STAGE_TIMEOUT_S,
        )

        cfg = DefaultPipelineBuilderConfig()
        assert cfg.kind == "default"
        assert cfg.dag_timeout == DEFAULT_DAG_TIMEOUT_S
        assert cfg.stage_timeout == DEFAULT_STAGE_TIMEOUT_S

    def test_context_construct(self) -> None:
        cfg = ContextPipelineBuilderConfig(dag_timeout=1800.0)
        assert cfg.kind == "context"
        assert cfg.dag_timeout == 1800.0

    def test_algotune_speed_construct(self) -> None:
        cfg = AlgoTuneSpeedPipelineBuilderConfig()
        assert cfg.kind == "algotune_speed"

    def test_cma_opt_construct(self) -> None:
        cfg = CMAOptPipelineBuilderConfig()
        assert cfg.kind == "cma_opt"

    def test_optuna_opt_construct(self) -> None:
        cfg = OptunaOptPipelineBuilderConfig()
        assert cfg.kind == "optuna_opt"

    def test_structural_metrics_construct(self) -> None:
        cfg = StructuralMetricsPipelineBuilderConfig()
        assert cfg.kind == "structural_metrics"

    def test_auto_construct(self) -> None:
        cfg = AutoPipelineBuilderConfig()
        assert cfg.kind == "auto"

    def test_dag_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DefaultPipelineBuilderConfig(dag_timeout=0.0)

    def test_stage_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DefaultPipelineBuilderConfig(stage_timeout=0.0)

    def test_stage_timeout_only_on_default_and_auto(self) -> None:
        """The runtime ContextPipelineBuilder / AlgoTune / CMA / Optuna
        / structural-metrics / problem-specific builders do not accept
        a ``stage_timeout`` argument. The schema mirrors that contract:
        ``stage_timeout`` lives on Default and Auto only so a typo
        cannot silently accept a value that the runtime would drop."""
        assert "stage_timeout" in DefaultPipelineBuilderConfig.model_fields
        assert "stage_timeout" in AutoPipelineBuilderConfig.model_fields
        for variant in (
            ContextPipelineBuilderConfig,
            AlgoTuneSpeedPipelineBuilderConfig,
            CMAOptPipelineBuilderConfig,
            OptunaOptPipelineBuilderConfig,
            StructuralMetricsPipelineBuilderConfig,
            ProblemSpecificPipelineBuilderConfig,
        ):
            assert "stage_timeout" not in variant.model_fields, (
                f"{variant.__name__} must not declare stage_timeout; "
                "the runtime backing ignores it"
            )

    def test_stage_timeout_rejected_on_context_variant(self) -> None:
        """Defense-in-depth: ``extra='forbid'`` must reject the field
        even when callers try to pass it explicitly."""
        with pytest.raises(ValidationError):
            ContextPipelineBuilderConfig(stage_timeout=300.0)  # type: ignore[call-arg]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DefaultPipelineBuilderConfig(da_timeout=10.0)  # type: ignore[call-arg]


class TestPipelineBuilderUnion:
    def test_round_trip_each_variant(self) -> None:
        ta: TypeAdapter[PipelineBuilderConfig] = TypeAdapter(PipelineBuilderConfig)
        for cfg in (
            DefaultPipelineBuilderConfig(),
            ContextPipelineBuilderConfig(),
            AlgoTuneSpeedPipelineBuilderConfig(),
            CMAOptPipelineBuilderConfig(),
            OptunaOptPipelineBuilderConfig(),
            StructuralMetricsPipelineBuilderConfig(),
            AutoPipelineBuilderConfig(),
            ProblemSpecificPipelineBuilderConfig(
                builder_path="some.module.Builder"
            ),
        ):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)

    def test_json_round_trip(self) -> None:
        ta: TypeAdapter[PipelineBuilderConfig] = TypeAdapter(PipelineBuilderConfig)
        cfg = ContextPipelineBuilderConfig(dag_timeout=1200.0)
        as_json = ta.dump_json(cfg)
        parsed = ta.validate_json(as_json)
        assert isinstance(parsed, ContextPipelineBuilderConfig)
        assert parsed.dag_timeout == 1200.0

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[PipelineBuilderConfig] = TypeAdapter(PipelineBuilderConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_builder"})


class TestPipelineConfig:
    def test_minimal(self) -> None:
        cfg = PipelineConfig(builder=DefaultPipelineBuilderConfig())
        assert cfg.prompts_dir is None
        assert isinstance(cfg.builder, DefaultPipelineBuilderConfig)

    def test_with_prompts_dir(self) -> None:
        cfg = PipelineConfig(
            builder=AutoPipelineBuilderConfig(),
            prompts_dir=Path("/srv/gigaevo/prompts"),
        )
        assert cfg.prompts_dir == Path("/srv/gigaevo/prompts")

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(
                builder=DefaultPipelineBuilderConfig(),
                buidler=None,  # type: ignore[call-arg]
            )

    def test_frozen(self) -> None:
        cfg = PipelineConfig(builder=DefaultPipelineBuilderConfig())
        with pytest.raises(ValidationError):
            cfg.prompts_dir = Path("/x")  # type: ignore[misc]

    def test_empty_prompts_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="prompts_dir"):
            PipelineConfig(
                builder=DefaultPipelineBuilderConfig(),
                prompts_dir=Path(""),
            )

    def test_cwd_dot_prompts_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="prompts_dir"):
            PipelineConfig(
                builder=DefaultPipelineBuilderConfig(),
                prompts_dir=Path("."),
            )


class TestPipelineBuilderDiscriminator:
    """Defense-in-depth: ensure each variant has a unique kind literal.
    A duplicate-string typo would silently let one variant shadow
    another in the union dispatch — Pydantic would round-trip to the
    first match without warning."""

    def test_each_variant_has_unique_kind(self) -> None:
        variants = (
            DefaultPipelineBuilderConfig,
            ContextPipelineBuilderConfig,
            AlgoTuneSpeedPipelineBuilderConfig,
            CMAOptPipelineBuilderConfig,
            OptunaOptPipelineBuilderConfig,
            StructuralMetricsPipelineBuilderConfig,
            AutoPipelineBuilderConfig,
            ProblemSpecificPipelineBuilderConfig,
        )
        kinds = [v.model_fields["kind"].default for v in variants]
        assert len(set(kinds)) == len(kinds), f"duplicate kind literal in {kinds}"
        assert sorted(kinds) == sorted({
            "default",
            "context",
            "algotune_speed",
            "cma_opt",
            "optuna_opt",
            "structural_metrics",
            "auto",
            "problem_specific",
        })
