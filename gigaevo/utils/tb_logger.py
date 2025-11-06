from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any

from pydantic import BaseModel, Field
from tensorboardX import SummaryWriter

from gigaevo.utils.logger import LogWriter


class TBConfig(BaseModel):
    logdir: str | Path
    flush_secs: float = 3.0
    queue_size: int = 8192


def _sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.=," else "_" for ch in str(s))


def _render_tag(path: list[str], metric: str, labels: dict[str, str]) -> str:
    base = "/".join(_sanitize(x) for x in [*path, metric] if x)
    if not labels:
        return base
    return base + (
        "/"
        + ",".join(f"{_sanitize(k)}={_sanitize(v)}" for k, v in sorted(labels.items()))
    )


class _SeriesState(BaseModel):
    last_step: int = Field(default=-1)
    last_value: float | None = Field(default=None)
    last_ts: float = Field(default=0.0)


class TBLogger(LogWriter):
    def __init__(self, cfg: TBConfig):
        self.cfg = cfg
        self._series: dict[str, _SeriesState] = {}
        self._q: Queue[dict[str, Any]] = Queue(maxsize=cfg.queue_size)
        self._stop = threading.Event()
        self._closed = False

        logdir = Path(cfg.logdir).resolve()
        logdir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(logdir), flush_secs=cfg.flush_secs)

        self._t = threading.Thread(target=self._loop, name="tb-writer", daemon=True)
        self._t.start()

    def bind(
        self, *, path: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> "BoundTB":
        return BoundTB(self, path or [], labels or {})

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
                "labels": kw.get("labels") or {},
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
                "labels": kw.get("labels") or {},
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
                "labels": kw.get("labels") or {},
            }
        )

    def close(self, drain_timeout_s: float = 1.5) -> None:
        """Stop thread, drain queue briefly, flush & close writer."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()

        # Drain remaining events for a short grace period
        deadline = time.time() + max(0.0, drain_timeout_s)
        while time.time() < deadline:
            try:
                event = self._q.get_nowait()
            except Empty:
                break
            else:
                self._handle(event)

        # Join the thread
        if self._t.is_alive():
            self._t.join(timeout=2.0)

        # Final flush/close
        try:
            self._writer.flush()
        finally:
            self._writer.close()

    # -------- internal --------

    def _offer(self, event: dict[str, Any]) -> None:
        try:
            self._q.put_nowait(event)
        except Exception:
            # drop on full/any error
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._q.get(timeout=0.1)
            except Empty:
                continue
            self._handle(event)

    def _handle(self, e: dict[str, Any]) -> None:
        tag = _render_tag(e["path"], e["metric"], e["labels"])
        step = self._resolve_step(tag, e.get("step"))
        wall_time = e.get("t", time.time())
        kind = e["k"]

        if kind == "scalar":
            if self._throttle(tag, e["value"], wall_time, e.get("thr") or {}):
                return
            self._writer.add_scalar(
                tag, e["value"], global_step=step, walltime=wall_time
            )
        elif kind == "hist":
            self._writer.add_histogram(
                tag, e["values"], global_step=step, walltime=wall_time
            )
        elif kind == "text":
            self._writer.add_text(tag, e["text"], global_step=step, walltime=wall_time)

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


class BoundTB(LogWriter):
    def __init__(self, base: TBLogger, path: list[str], labels: dict[str, str]):
        self._base = base
        self._path = list(path)
        self._labels = dict(labels)

    def bind(
        self, *, path: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> "BoundTB":
        return BoundTB(
            self._base, [*self._path, *(path or [])], {**self._labels, **(labels or {})}
        )

    def scalar(self, metric: str, value: float, **kw) -> None:
        path = [*self._path, *kw.pop("path", [])]
        labels = {**self._labels, **kw.pop("labels", {})}
        self._base.scalar(metric, value, path=path, labels=labels, **kw)

    def hist(self, metric: str, values: Any, **kw) -> None:
        path = [*self._path, *kw.pop("path", [])]
        labels = {**self._labels, **kw.pop("labels", {})}
        self._base.hist(metric, values, path=path, labels=labels, **kw)

    def text(self, tag: str, text: str, **kw) -> None:
        path = [*self._path, *kw.pop("path", [])]
        labels = {**self._labels, **kw.pop("labels", {})}
        self._base.text(tag, text, path=path, labels=labels, **kw)

    def close(self) -> None:
        self._base.close()


_default: TBLogger | None = None


def init_tb(cfg: TBConfig) -> TBLogger:
    global _default
    if _default is not None:
        return _default
    _default = TBLogger(cfg)
    return _default


def get_tb() -> TBLogger:
    if _default is None:
        raise ValueError("TBLogger not initialized. Call init_tb() first.")
    return _default
