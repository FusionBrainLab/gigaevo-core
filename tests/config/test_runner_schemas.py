"""Unit tests for the DAG runner schema + preset."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gigaevo.config.defaults import (
    DEFAULT_DAG_TIMEOUT_S,
    DEFAULT_MAX_CONCURRENT_DAGS,
    DEFAULT_RUNNER_POLL_INTERVAL_S,
)
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import DAGRunnerConfig


class TestDAGRunnerConfig:
    def test_defaults(self) -> None:
        cfg = DAGRunnerConfig()
        assert cfg.poll_interval == 0.5
        assert cfg.max_concurrent_dags == 8
        assert cfg.prefetch_factor == 8
        assert cfg.metrics_collection_interval == 1.0
        assert cfg.dag_timeout == 3600.0

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(unknown_field=1)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = DAGRunnerConfig()
        with pytest.raises(ValidationError):
            cfg.poll_interval = 1.0  # type: ignore[misc]

    def test_poll_interval_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(poll_interval=0.0)

    def test_poll_interval_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(poll_interval=120.0)

    def test_max_concurrent_dags_positive(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(max_concurrent_dags=0)

    def test_max_concurrent_dags_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(max_concurrent_dags=2000)

    def test_prefetch_factor_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(prefetch_factor=0)
        with pytest.raises(ValidationError):
            DAGRunnerConfig(prefetch_factor=100)

    def test_dag_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DAGRunnerConfig(dag_timeout=0.0)

    def test_build_runtime_round_trip(self) -> None:
        from gigaevo.runner.dag_runner import DagRunnerConfig as RuntimeDagRunnerConfig

        cfg = DAGRunnerConfig(
            poll_interval=2.0,
            max_concurrent_dags=16,
            dag_timeout=1800.0,
            prefetch_factor=4,
            metrics_collection_interval=0.5,
        )
        runtime = cfg.build()
        assert isinstance(runtime, RuntimeDagRunnerConfig)
        assert runtime.poll_interval == 2.0
        assert runtime.max_concurrent_dags == 16
        assert runtime.dag_timeout == 1800.0
        assert runtime.prefetch_factor == 4
        assert runtime.metrics_collection_interval == 0.5


class TestBuildDefaultRunner:
    def test_defaults_match_constants(self) -> None:
        cfg = build_default_runner()
        assert cfg.poll_interval == DEFAULT_RUNNER_POLL_INTERVAL_S
        assert cfg.max_concurrent_dags == DEFAULT_MAX_CONCURRENT_DAGS
        assert cfg.dag_timeout == DEFAULT_DAG_TIMEOUT_S

    def test_yaml_parity(self) -> None:
        """config/runner/default.yaml uses ${runner_poll_interval},
        ${max_concurrent_dags}, ${dag_timeout} which resolve to the
        same DEFAULT_* constants. The preset matches byte-equal."""
        cfg = build_default_runner()
        assert cfg.poll_interval == 5.0  # constants/runner.yaml value
        assert cfg.max_concurrent_dags == 10  # constants/runner.yaml value
        assert cfg.dag_timeout == 7200  # constants/pipeline.yaml value

    def test_custom_overrides_propagate(self) -> None:
        cfg = build_default_runner(
            poll_interval=1.0,
            max_concurrent_dags=4,
            dag_timeout=600.0,
            prefetch_factor=2,
        )
        assert cfg.poll_interval == 1.0
        assert cfg.max_concurrent_dags == 4
        assert cfg.dag_timeout == 600.0
        assert cfg.prefetch_factor == 2


class TestJSONRoundTrip:
    def test_round_trip_preserves_every_field(self) -> None:
        cfg = DAGRunnerConfig(
            poll_interval=1.5,
            max_concurrent_dags=12,
            prefetch_factor=16,
            metrics_collection_interval=2.0,
            dag_timeout=2400.0,
        )
        parsed = DAGRunnerConfig.model_validate_json(cfg.model_dump_json())
        assert parsed == cfg
