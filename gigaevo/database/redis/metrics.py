from __future__ import annotations

import asyncio
from typing import Any

from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis.keys import RedisProgramKeys
from gigaevo.exceptions import StorageError
from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.trackers.base import LogWriter


class RedisMetricsCollector:
    """Collects and reports Redis storage metrics."""

    def __init__(
        self,
        connection: RedisConnection,
        keys: RedisProgramKeys,
        writer: LogWriter | None,
        interval: float = 1.0,
    ):
        self._conn = connection
        self._keys = keys
        self._writer = writer.bind(path=["redis_storage"]) if writer else None
        self._interval = interval

        self._task: asyncio.Task | None = None
        self._stop_flag = False

    def start(self) -> None:
        """Start the metrics collection task."""
        if self._writer is None:
            return
        if self._task is not None and not self._task.done():
            return

        self._stop_flag = False
        self._task = start_metrics_collector(
            writer=self._writer,
            collect_fn=self._collect,
            interval=self._interval,
            stop_flag=lambda: self._stop_flag,
            task_name="redis-metrics-collector",
        )

    async def stop(self) -> None:
        """Stop the metrics collection task."""
        self._stop_flag = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _collect(self) -> dict[str, Any]:
        """Collect Redis metrics."""
        if self._conn.is_closing:
            return {}

        try:
            r = await self._conn.get()
        except StorageError:
            return {}

        metrics: dict[str, Any] = {}

        # Count programs
        count = 0
        async for _ in r.scan_iter(match=self._keys.program_pattern(), count=1000):
            count += 1
        metrics["size"] = float(count)

        # Collect Redis INFO sections
        for section in (
            "stats",
            "memory",
            "clients",
            "server",
            "cpu",
            "keyspace",
            "replication",
        ):
            try:
                info = await r.info(section=section)
                for k, v in _flatten_numbers(info, prefix=f"{section}/").items():
                    metrics[k] = v
            except Exception:
                continue

        return metrics


def _flatten_numbers(d: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dict, extracting only numeric values."""
    out: dict[str, float] = {}

    def _walk(x: Any, p: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                key = f"{p}{k}" if not p.endswith("/") else f"{p}{k}"
                if isinstance(v, (int, float)):
                    out[key] = float(v)
                elif isinstance(v, dict):
                    _walk(v, key + "/")

    _walk(d, prefix)
    return out
