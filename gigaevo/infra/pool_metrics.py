"""Per-endpoint usage metrics for LLM load balancer.

Follows the ``TokenTracker`` pattern: writes scalar metrics via ``LogWriter``
so they appear in the same Redis metrics stream as token counts.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from loguru import logger

from gigaevo.utils.trackers.base import LogWriter


def _endpoint_label(url: str) -> str:
    """Convert an endpoint URL to a safe metric label.

    ``http://localhost:8000/v1`` → ``INTERNAL_IP_8777``
    """
    stripped = re.sub(r"^https?://", "", url)
    stripped = re.sub(r"/.*$", "", stripped)
    return stripped.replace(".", "_").replace(":", "_")


class PoolMetricsTracker:
    """Tracks per-endpoint request count, error count, and latency.

    Thread-safe (guarded by a lock, same pattern as ``TokenTracker``).
    Metrics are written under ``pool/{pool_name}/{endpoint_label}/...``.
    """

    def __init__(self, pool_name: str = "default", writer: LogWriter | None = None):
        self.pool_name = pool_name
        self.writer = writer
        self.cumulative: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record(self, endpoint: str, latency_ms: float, success: bool) -> None:
        """Record one request outcome. Thread-safe."""
        label = _endpoint_label(endpoint)
        with self._lock:
            if label not in self.cumulative:
                self.cumulative[label] = {
                    "requests": 0,
                    "errors": 0,
                    "total_latency_ms": 0.0,
                }
            cum = self.cumulative[label]
            cum["requests"] += 1
            if not success:
                cum["errors"] += 1
            cum["total_latency_ms"] += latency_ms

            if self.writer is not None:
                path = ["pool", self.pool_name, label]
                self.writer.scalar("request_count", float(cum["requests"]), path=path)
                self.writer.scalar("error_count", float(cum["errors"]), path=path)
                self.writer.scalar("latency_ms", latency_ms, path=path)
                self.writer.scalar(
                    "avg_latency_ms",
                    cum["total_latency_ms"] / cum["requests"],
                    path=path,
                )

        logger.debug(
            "[PoolMetrics:{}] {} → {:.0f}ms ({}) [total: {} req, {} err]",
            self.pool_name,
            label,
            latency_ms,
            "ok" if success else "FAIL",
            cum["requests"],
            cum["errors"],
        )
