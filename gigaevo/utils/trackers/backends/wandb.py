from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, List, Literal

from pydantic import BaseModel

from gigaevo.utils.trackers.configs import WBConfig
from gigaevo.utils.trackers.core import LoggerBackend
import wandb

EventKind = Literal["scalar", "hist", "text"]


@dataclass
class Event:
    step: int
    tag: str
    kind: EventKind
    value: Any


class WandBBackend(LoggerBackend):
    def __init__(self, cfg: WBConfig):
        self.cfg = cfg
        self._run = None
        self._lock = threading.Lock()
        self._buffer: List[Event] = []

    def _prepare_init_kwargs(self) -> dict[str, Any]:
        base = self.cfg.model_dump(exclude_none=True)
        if isinstance(self.cfg.config, BaseModel):
            base["config"] = self.cfg.config.model_dump()
        return base

    def open(self) -> None:
        init_kwargs = self._prepare_init_kwargs()
        self._run = wandb.init(**init_kwargs)

    def close(self) -> None:
        self.flush()
        if self._run is not None:
            self._run.finish()
        self._run = None

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        ev = Event(step=step, tag=tag, kind="scalar", value=float(value))
        with self._lock:
            self._buffer.append(ev)

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        ev = Event(step=step, tag=tag, kind="hist", value=values)
        with self._lock:
            self._buffer.append(ev)

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        ev = Event(step=step, tag=tag, kind="text", value=text)
        with self._lock:
            self._buffer.append(ev)

    def _convert_event_to_payload(self, ev: Event) -> Any:
        if ev.kind == "scalar":
            return ev.value
        if ev.kind == "hist":
            return wandb.Histogram(ev.value) if ev.value is not None else None
        if ev.kind == "text":
            return wandb.Html(ev.value)
        return ev.value

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            buf, self._buffer = self._buffer, []

        grouped: dict[int, dict[str, Event]] = {}
        for ev in buf:
            d = grouped.setdefault(ev.step, {})
            d[ev.tag] = ev

        any_sent = False
        for step, tag_events in grouped.items():
            payload: dict[str, Any] = {}
            for tag, ev in tag_events.items():
                payload[tag] = self._convert_event_to_payload(ev)
            wandb.log(payload, step=step, commit=False)
            any_sent = True

        if any_sent:
            wandb.log({}, commit=True)
