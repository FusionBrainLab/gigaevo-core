"""Backend-neutral metric history access for CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import click

from gigaevo.monitoring.run_spec import RunSpec


class MetricHistorySource(Protocol):
    def list_tags(self) -> list[str]: ...

    def get_history(self, tag: str) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_tag(tag: str) -> str:
    return tag.replace("/", ":").replace(" ", "_")


@dataclass
class RedisMetricHistorySource:
    client: Any
    key_prefix: str

    def list_tags(self) -> list[str]:
        tags = {_decode(tag) for tag in self.client.hkeys(f"{self.key_prefix}:latest")}
        history_prefix = f"{self.key_prefix}:history:"
        for raw_key in self.client.scan_iter(match=f"{history_prefix}*", count=1000):
            key = _decode(raw_key)
            if key.startswith(history_prefix):
                tags.add(key.removeprefix(history_prefix).replace(":", "/"))
        return sorted(tags)

    def get_history(self, tag: str) -> list[dict[str, Any]]:
        key = f"{self.key_prefix}:history:{_safe_tag(tag)}"
        entries: list[dict[str, Any]] = []
        for raw in self.client.lrange(key, 0, -1):
            try:
                entry = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def close(self) -> None:
        self.client.close()


@dataclass
class DiskMetricHistorySource:
    metrics_dir: Path

    def _backend(self):
        from gigaevo.utils.trackers.backends.disk import DiskMetricsBackend
        from gigaevo.utils.trackers.configs import DiskMetricsConfig

        return DiskMetricsBackend(DiskMetricsConfig(root_dir=self.metrics_dir))

    def list_tags(self) -> list[str]:
        return self._backend().list_metrics()

    def get_history(self, tag: str) -> list[dict[str, Any]]:
        return self._backend().get_history(tag)

    def close(self) -> None:
        return None


def disk_metrics_dir(spec: RunSpec) -> Path:
    """Return the default metrics directory adjacent to a disk storage root."""
    assert spec.path is not None
    return Path(spec.path).expanduser().parent / "metrics"


def build_metric_history_source(
    spec: RunSpec,
    redis_host: str,
    redis_port: int,
    redis_factory=None,
) -> MetricHistorySource:
    if spec.is_disk:
        metrics_dir = disk_metrics_dir(spec)
        if not metrics_dir.is_dir():
            raise click.UsageError(
                f"Disk metric history not found for {spec.label}: {metrics_dir}. "
                "Expected the default <run-dir>/metrics directory next to storage."
            )
        return DiskMetricHistorySource(metrics_dir)

    if redis_factory is not None:
        client = redis_factory(spec.db)
    else:
        import redis

        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=spec.db,
            decode_responses=True,
        )
    return RedisMetricHistorySource(client, f"{spec.prefix}:metrics")
