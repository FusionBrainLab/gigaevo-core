from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tensorboardX import SummaryWriter

from gigaevo.utils.trackers.configs import TBConfig
from gigaevo.utils.trackers.core import LoggerBackend


class TBBackend(LoggerBackend):
    def __init__(self, cfg: TBConfig):
        self.cfg = cfg
        self._writer: Optional[SummaryWriter] = None

    def open(self) -> None:
        logdir = Path(self.cfg.logdir).resolve()
        logdir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(str(logdir), **self.cfg.summary_writer_kwargs)

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.flush()
        finally:
            try:
                self._writer.close()
            except Exception:
                pass

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        assert self._writer is not None
        self._writer.add_scalar(tag, value, global_step=step, walltime=wall_time)

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        assert self._writer is not None
        self._writer.add_histogram(tag, values, global_step=step, walltime=wall_time)

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        assert self._writer is not None
        self._writer.add_text(tag, text, global_step=step, walltime=wall_time)

    def flush(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.flush()
        except Exception:
            pass
