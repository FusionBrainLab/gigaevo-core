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
    StructuralMetricsPipelineBuilderConfig,
)


class TestPipelineBuilderConfigs:
    def test_default_construct(self) -> None:
        cfg = DefaultPipelineBuilderConfig()
        assert cfg.kind == "default"
        assert cfg.dag_timeout == 3600.0
        assert cfg.stage_timeout == 300.0

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
