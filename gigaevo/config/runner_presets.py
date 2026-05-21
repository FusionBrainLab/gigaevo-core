"""Preset builder for the DAG runner.

Exposes a single :func:`build_default_runner` factory returning a
fully-validated :class:`DAGRunnerConfig`. Experiment files compose it
into the experiment root::

    from gigaevo.config.runner_presets import build_default_runner

    def build() -> ExperimentConfig:
        return ExperimentConfig(..., runner=build_default_runner())

Defaults flow from :mod:`gigaevo.config.defaults`
(``DEFAULT_RUNNER_POLL_INTERVAL_S``, ``DEFAULT_MAX_CONCURRENT_DAGS``,
``DEFAULT_DAG_TIMEOUT_S``); every knob has a keyword override so a
sweep can pin a single parameter without reaching into the schema.
"""

from __future__ import annotations

from gigaevo.config.defaults import (
    DEFAULT_DAG_TIMEOUT_S,
    DEFAULT_MAX_CONCURRENT_DAGS,
    DEFAULT_RUNNER_POLL_INTERVAL_S,
)
from gigaevo.config.schemas import DAGRunnerConfig


def build_default_runner(
    *,
    poll_interval: float = DEFAULT_RUNNER_POLL_INTERVAL_S,
    max_concurrent_dags: int = DEFAULT_MAX_CONCURRENT_DAGS,
    dag_timeout: float = DEFAULT_DAG_TIMEOUT_S,
    prefetch_factor: int = 8,
    metrics_collection_interval: float = 1.0,
) -> DAGRunnerConfig:
    """Default DAG runner."""
    return DAGRunnerConfig(
        poll_interval=poll_interval,
        max_concurrent_dags=max_concurrent_dags,
        dag_timeout=dag_timeout,
        prefetch_factor=prefetch_factor,
        metrics_collection_interval=metrics_collection_interval,
    )


__all__: list[str] = ["DAGRunnerConfig", "build_default_runner"]
