from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
import redis as redis_lib

from gigaevo.monitoring.redis_queries import collect_snapshot
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot

_log = logger.bind(component="experiment_monitor")

# Type alias for the redis factory callable used for dependency injection
RedisFactory = Callable[[int], redis_lib.Redis]


@dataclass
class RunConfig:
    """Configuration for collecting a single run's snapshot."""

    run_spec: RunSpec
    metric_names: list[str] = field(default_factory=lambda: ["fitness"])
    pid: int | None = None


class ExperimentMonitor:
    """Collects RunSnapshots for all runs in an experiment.

    Usage:
        monitor = ExperimentMonitor(redis_host="localhost", redis_port=6379)
        snapshots = monitor.collect(runs=[
            RunConfig(RunSpec.parse("prefix@4:O"), metric_names=["fitness"]),
            RunConfig(RunSpec.parse("prefix@5:R"), metric_names=["fitness", "prompt_length"]),
        ])

    For testing, inject a redis_factory to return fakeredis instances:
        monitor = ExperimentMonitor(redis_factory=lambda db: fakeredis.FakeRedis(server=server, db=db))
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_factory: RedisFactory | None = None,
    ):
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_factory = redis_factory

    def collect(self, runs: list[RunConfig]) -> list[RunSnapshot]:
        """Collect snapshots for all runs.

        Each run is queried independently. If one fails, others still
        succeed -- the failed run gets a snapshot with an error field.

        Args:
            runs: List of RunConfig objects specifying what to collect.

        Returns:
            List of RunSnapshot, one per run, in the same order as input.
        """
        snapshots: list[RunSnapshot] = []
        for run_cfg in runs:
            snapshot = self._collect_one(run_cfg)
            snapshots.append(snapshot)
        return snapshots

    def _collect_one(self, run_cfg: RunConfig) -> RunSnapshot:
        """Collect a single run's snapshot."""
        try:
            if self._redis_factory is not None:
                r = self._redis_factory(run_cfg.run_spec.db)
            else:
                r = redis_lib.Redis(
                    host=self._redis_host,
                    port=self._redis_port,
                    db=run_cfg.run_spec.db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
            try:
                return collect_snapshot(
                    r,
                    run_cfg.run_spec,
                    metric_names=run_cfg.metric_names,
                    pid=run_cfg.pid,
                )
            finally:
                r.close()
        except Exception as exc:
            _log.error(f"Failed to connect to Redis for {run_cfg.run_spec}: {exc}")
            return RunSnapshot(run_spec=run_cfg.run_spec, error=str(exc))
