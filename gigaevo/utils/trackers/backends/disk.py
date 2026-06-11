from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

from loguru import logger

from gigaevo.utils.trackers.configs import DiskMetricsConfig
from gigaevo.utils.trackers.core import LoggerBackend


class DiskMetricsBackend(LoggerBackend):
    """JSONL disk backend mirroring RedisMetricsBackend's history contract.

    One append-only ``<root_dir>/<safe_tag>.jsonl`` file per tag, where
    ``safe_tag`` uses the same sanitization as the Redis history keys
    (``/`` → ``:``, `` `` → ``_``) and each line is the same entry schema
    ``{"s": step, "t": wall_time, "v": value, "k": kind}``.
    """

    def __init__(self, cfg: DiskMetricsConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []

    @staticmethod
    def _safe_tag(tag: str) -> str:
        # Same sanitization as the Redis backend; tags containing ":" are
        # unsupported (they would collide with a sanitized "/").
        return tag.replace("/", ":").replace(" ", "_")

    def _path(self, tag: str) -> Path:
        return Path(self.cfg.root_dir) / f"{self._safe_tag(tag)}.jsonl"

    def open(self) -> None:
        Path(self.cfg.root_dir).mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.flush()

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        with self._lock:
            self._buffer.append(
                {"tag": tag, "s": step, "t": wall_time, "v": value, "k": "scalar"}
            )

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        vals = list(values) if hasattr(values, "__iter__") else values
        with self._lock:
            self._buffer.append(
                {"tag": tag, "s": step, "t": wall_time, "v": vals, "k": "hist"}
            )

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        with self._lock:
            self._buffer.append(
                {"tag": tag, "s": step, "t": wall_time, "v": text, "k": "text"}
            )

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            buf, self._buffer = self._buffer, []

        by_tag: dict[str, list[str]] = {}
        for entry in buf:
            tag = entry.pop("tag")
            by_tag.setdefault(tag, []).append(json.dumps(entry))

        for tag, lines in by_tag.items():
            try:
                path = self._path(tag)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception as e:
                logger.warning("[DiskMetricsBackend] flush failed for {}: {}", tag, e)

    def clear_series(self, tag: str) -> None:
        with self._lock:
            self._buffer = [e for e in self._buffer if e.get("tag") != tag]
        try:
            self._path(tag).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(
                "[DiskMetricsBackend] clear_series failed for {}: {}", tag, e
            )

    # --------------------- Query Methods ---------------------

    def get_history(
        self, tag: str, start: int = 0, end: int = -1
    ) -> list[dict[str, Any]]:
        """Read a tag's history with LRANGE-style slicing."""
        path = self._path(tag)
        if not path.exists():
            return []
        try:
            entries = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        except Exception as e:
            logger.warning("[DiskMetricsBackend] get_history failed for {}: {}", tag, e)
            return []
        n = len(entries)
        if start < 0:
            start = max(n + start, 0)
        if end < 0:
            end = n + end
        if end < start:
            return []
        return entries[start : end + 1]
