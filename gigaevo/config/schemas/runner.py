from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from gigaevo.runner.dag_runner import DagRunnerConfig as RuntimeDagRunnerConfig


class DAGRunnerConfig(FrozenStrictModel):
    """DAG runner tuning matching the runtime
    ``gigaevo.runner.dag_runner.DagRunnerConfig`` surface.

    ``poll_interval`` controls how often the runner scans the queue
    for programs ready to launch. ``max_concurrent_dags`` caps the
    semaphore guarding actual DAG executions; ``prefetch_factor``
    pre-creates that many batches of tasks waiting on the semaphore
    so the next DAG starts immediately when a slot opens (instead of
    waiting for the next poll tick).

    The poll-interval bounds match the runtime validator: values
    under 0.01s would saturate CPU, and over 60s starve the queue;
    values over 30s emit a runtime warning about responsiveness.

    The ``poll_interval``, ``max_concurrent_dags`` and ``dag_timeout``
    defaults agree with
    :data:`gigaevo.config.defaults.DEFAULT_RUNNER_POLL_INTERVAL_S`,
    :data:`gigaevo.config.defaults.DEFAULT_MAX_CONCURRENT_DAGS` and
    :data:`gigaevo.config.defaults.DEFAULT_DAG_TIMEOUT_S` so direct
    schema construction matches :func:`build_default_runner`."""

    poll_interval: float = Field(default=5.0, ge=0.01, le=60.0)
    max_concurrent_dags: int = Field(default=10, gt=0, le=1000)
    prefetch_factor: int = Field(default=8, ge=1, le=64)
    metrics_collection_interval: float = Field(default=1.0, gt=0.0)
    dag_timeout: float = Field(default=7200.0, gt=0.0)

    def build(self) -> RuntimeDagRunnerConfig:
        from gigaevo.runner.dag_runner import DagRunnerConfig

        return DagRunnerConfig(
            poll_interval=self.poll_interval,
            max_concurrent_dags=self.max_concurrent_dags,
            prefetch_factor=self.prefetch_factor,
            metrics_collection_interval=self.metrics_collection_interval,
            dag_timeout=self.dag_timeout,
        )
