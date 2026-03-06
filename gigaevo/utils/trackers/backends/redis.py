from __future__ import annotations

import json
import threading
import time
from typing import Any

from loguru import logger
import redis

from gigaevo.utils.trackers.configs import RedisMetricsConfig
from gigaevo.utils.trackers.core import LoggerBackend


class RedisMetricsBackend(LoggerBackend):
    """Redis backend that stores metrics for querying.

    Storage structure:
    - {prefix}:latest (hash) - latest value for each metric tag
    - {prefix}:history:{tag} (list) - time series history as JSON entries
    - {prefix}:meta (hash) - metadata like last update time
    """

    def __init__(self, cfg: RedisMetricsConfig):
        self.cfg = cfg
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []

    def _k_latest(self) -> str:
        return f"{self.cfg.key_prefix}:latest"

    def _k_history(self, tag: str) -> str:
        # Sanitize tag for Redis key
        safe_tag = tag.replace("/", ":").replace(" ", "_")
        return f"{self.cfg.key_prefix}:history:{safe_tag}"

    def _k_meta(self) -> str:
        return f"{self.cfg.key_prefix}:meta"

    def open(self) -> None:
        self._pool = redis.ConnectionPool.from_url(
            str(self.cfg.redis_url),
            max_connections=self.cfg.max_connections,
            socket_timeout=self.cfg.socket_timeout,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Test connection
        self._client.ping()
        logger.debug("[RedisMetricsBackend] Connected to {}", self.cfg.redis_url)

    def close(self) -> None:
        self.flush()
        if self._pool:
            self._pool.disconnect()
        self._client = None
        self._pool = None

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        entry = {
            "kind": "scalar",
            "tag": tag,
            "value": value,
            "step": step,
            "wall_time": wall_time,
        }
        with self._lock:
            self._buffer.append(entry)

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        # Store histogram as JSON (simplified - just the values)
        entry = {
            "kind": "hist",
            "tag": tag,
            "values": list(values) if hasattr(values, "__iter__") else values,
            "step": step,
            "wall_time": wall_time,
        }
        with self._lock:
            self._buffer.append(entry)

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        entry = {
            "kind": "text",
            "tag": tag,
            "value": text,
            "step": step,
            "wall_time": wall_time,
        }
        with self._lock:
            self._buffer.append(entry)

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            buf, self._buffer = self._buffer, []

        if not self._client:
            return

        try:
            pipe = self._client.pipeline(transaction=False)

            for entry in buf:
                tag = entry["tag"]
                step = entry["step"]
                wall_time = entry["wall_time"]
                kind = entry["kind"]

                # Update latest value
                if kind == "scalar":
                    pipe.hset(self._k_latest(), tag, entry["value"])
                elif kind == "text":
                    pipe.hset(self._k_latest(), tag, entry["value"])
                # histograms don't update latest (too large)

                # Store history if enabled
                if self.cfg.store_history:
                    history_entry = json.dumps(
                        {
                            "s": step,
                            "t": wall_time,
                            "v": entry.get("value") or entry.get("values"),
                            "k": kind,
                        }
                    )
                    history_key = self._k_history(tag)
                    pipe.rpush(history_key, history_entry)
                    # Trim to max size (FIFO)
                    pipe.ltrim(history_key, -self.cfg.max_history_per_metric, -1)

            # Update metadata
            pipe.hset(self._k_meta(), "last_update", time.time())
            pipe.execute()

        except Exception as e:
            logger.warning("[RedisMetricsBackend] Flush failed: {}", e)

    # --------------------- Query Methods ---------------------

    def get_latest(self, tag: str | None = None) -> dict[str, Any]:
        """Get latest value(s). If tag is None, return all."""
        if not self._client:
            return {}
        try:
            if tag:
                val = self._client.hget(self._k_latest(), tag)
                return {tag: float(val)} if val else {}
            else:
                data = self._client.hgetall(self._k_latest())
                return {k: self._parse_value(v) for k, v in data.items()}
        except Exception as e:
            logger.warning("[RedisMetricsBackend] get_latest failed: {}", e)
            return {}

    def get_history(
        self, tag: str, start: int = 0, end: int = -1
    ) -> list[dict[str, Any]]:
        """Get history for a metric tag."""
        if not self._client:
            return []
        try:
            entries = self._client.lrange(self._k_history(tag), start, end)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.warning("[RedisMetricsBackend] get_history failed: {}", e)
            return []

    def list_metrics(self) -> list[str]:
        """List all metric tags that have been recorded."""
        if not self._client:
            return []
        try:
            return list(self._client.hkeys(self._k_latest()))
        except Exception as e:
            logger.warning("[RedisMetricsBackend] list_metrics failed: {}", e)
            return []

    @staticmethod
    def _parse_value(v: str) -> Any:
        try:
            return float(v)
        except (ValueError, TypeError):
            return v
