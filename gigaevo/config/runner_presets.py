"""Preset builder for the DAG runner.

A single builder ships; defaults flow from
:mod:`gigaevo.config.defaults` (``DEFAULT_RUNNER_POLL_INTERVAL_S``,
``DEFAULT_MAX_CONCURRENT_DAGS``, ``DEFAULT_DAG_TIMEOUT_S``).
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
