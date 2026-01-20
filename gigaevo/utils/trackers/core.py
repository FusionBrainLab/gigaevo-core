# gigaevo_logger/core.py
from __future__ import annotations

from queue import Empty, Queue
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from gigaevo.utils.trackers.base import LogWriter


def _sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def _render_tag(path: list[str], metric: str) -> str:
    return "/".join(_sanitize(x) for x in [*path, metric] if x)


class _SeriesState(BaseModel):
    last_step: int = Field(default=-1)
    last_value: float | None = Field(default=None)
    last_ts: float = Field(default=0.0)


class LoggerBackend:
    """Minimal adapter every backend must implement."""

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        raise NotImplementedError

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        raise NotImplementedError

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError


class GenericLogger(LogWriter):
    def __init__(
        self, backend: LoggerBackend, *, queue_size: int = 8192, flush_secs: float = 3.0
    ):
        self.backend = backend
        self._series: dict[str, _SeriesState] = {}
        self._q: Queue[dict[str, Any]] = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._closed = False
        self._flush_secs = float(flush_secs)

        self.backend.open()

        self._t = threading.Thread(
            target=self._loop, name="generic-writer", daemon=True
        )
        self._t.start()

        self._last_flush = time.time()

    def bind(self, *, path: list[str] | None = None) -> "BoundGeneric":
        return BoundGeneric(self, path or [])

    def scalar(self, metric: str, value: float, **kw) -> None:
        if self._closed:
            return
        self._offer(
            {
                "k": "scalar",
                "metric": metric,
                "value": float(value),
                "step": kw.get("step"),
                "t": kw.get("wall_time") or time.time(),
                "path": kw.get("path") or [],
                "thr": kw.get("throttle") or {},
            }
        )

    def hist(self, metric: str, values: Any, **kw) -> None:
        if self._closed:
            return
        self._offer(
            {
                "k": "hist",
                "metric": metric,
                "values": values,
                "step": kw.get("step"),
                "t": kw.get("wall_time") or time.time(),
                "path": kw.get("path") or [],
            }
        )

    def text(self, tag: str, text: str, **kw) -> None:
        if self._closed:
            return
        self._offer(
            {
                "k": "text",
                "metric": tag,
                "text": text,
                "step": kw.get("step"),
                "t": kw.get("wall_time") or time.time(),
                "path": kw.get("path") or [],
            }
        )

    def close(self, drain_timeout_s: float = 1.5) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()

        deadline = time.time() + max(0.0, drain_timeout_s)
        while time.time() < deadline:
            try:
                event = self._q.get_nowait()
            except Empty:
                break
            else:
                self._handle(event)

        if self._t.is_alive():
            self._t.join(timeout=2.0)

        try:
            try:
                self.backend.flush()
            except Exception:
                pass
            self.backend.close()
        finally:
            self._closed = True

    def _offer(self, event: dict[str, Any]) -> None:
        try:
            self._q.put_nowait(event)
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._q.get(timeout=0.1)
            except Empty:
                now = time.time()
                if (now - self._last_flush) >= self._flush_secs:
                    try:
                        self.backend.flush()
                    except Exception:
                        pass
                    self._last_flush = now
                continue
            self._handle(event)

            now = time.time()
            if (now - self._last_flush) >= self._flush_secs:
                try:
                    self.backend.flush()
                except Exception:
                    pass
                self._last_flush = now

    def _handle(self, e: dict[str, Any]) -> None:
        tag = _render_tag(e["path"], e["metric"])
        step = self._resolve_step(tag, e.get("step"))
        wall_time = e.get("t", time.time())
        kind = e["k"]

        if kind == "scalar":
            if self._throttle(tag, e["value"], wall_time, e.get("thr") or {}):
                return
            try:
                self.backend.write_scalar(tag, e["value"], step, wall_time)
            except Exception:
                pass
        elif kind == "hist":
            try:
                self.backend.write_hist(tag, e["values"], step, wall_time)
            except Exception:
                pass
        elif kind == "text":
            try:
                self.backend.write_text(tag, e.get("text", ""), step, wall_time)
            except Exception:
                pass

    def _state(self, key: str) -> _SeriesState:
        st = self._series.get(key)
        if st is None:
            st = _SeriesState()
            self._series[key] = st
        return st

    def _resolve_step(self, key: str, step: int | None) -> int:
        st = self._state(key)
        if step is not None:
            st.last_step = int(step)
        else:
            st.last_step += 1
        return st.last_step

    def _throttle(
        self, key: str, value: float, ts: float, thr: dict[str, float]
    ) -> bool:
        st = self._state(key)
        min_interval = float(thr.get("min_interval_s", 0.0))
        min_delta = float(thr.get("min_delta", 0.0))
        if min_interval and (ts - st.last_ts) < min_interval:
            return True
        if (
            min_delta
            and st.last_value is not None
            and abs(value - st.last_value) < min_delta
        ):
            return True
        st.last_value = value
        st.last_ts = ts
        return False


class BoundGeneric(LogWriter):
    def __init__(self, base: GenericLogger, path: list[str]):
        self._base = base
        self._path = list(path)

    def bind(self, *, path: list[str] | None = None) -> "BoundGeneric":
        return BoundGeneric(self._base, [*self._path, *(path or [])])

    def scalar(self, metric: str, value: float, **kw) -> None:
        path = [*self._path, *kw.pop("path", [])]
        self._base.scalar(metric, value, path=path, **kw)

    def hist(self, metric: str, values: Any, **kw) -> None:
        path = [*self._path, *kw.pop("path", [])]
        self._base.hist(metric, values, path=path, **kw)

    def text(self, tag: str, text: str, **kw) -> None:
        path = [*self._path, *kw.pop("path", [])]
        self._base.text(tag, text, path=path, **kw)

    def close(self) -> None:
        self._base.close()
