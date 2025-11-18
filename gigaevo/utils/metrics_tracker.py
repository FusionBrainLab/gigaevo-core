from __future__ import annotations

import asyncio
from contextlib import suppress
import math

from loguru import logger
from pydantic import BaseModel

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import INCOMPLETE_STATES
from gigaevo.utils.trackers.base import LogWriter


class _RunningStats(BaseModel):
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta * (1.0 / self.n)
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2

    def mean_value(self) -> float:
        return self.mean if self.n > 0 else 0.0

    def std_value(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


class MetricsTracker:
    """
    Minimal metrics tracker:
      - Runs as a task on the engine's event loop.
      - Polls every `interval` seconds.
      - Processes each program exactly once (by id).
      - Skips running DAGs (FRESH / DAG_PROCESSING_STARTED) or without metrics/validity.
      - Writes:
          * "is_valid" (for all)
          * per-program metrics for valid programs: "valid/program/<metric>"
          * counts: programs/{valid_count, invalid_count, total_count}
          * frontier for valid only: "valid/frontier/<metric>" (step = iteration)
          * NEW (valid only):
              - per-iteration aggregates: "valid/iter/<metric>/{mean,std}" (step = iteration)
              - per-generation aggregates: "valid/gen/<metric>/{mean,std}" (step = generation)
      - Frontier uses MetricsContext.specs[metric].higher_is_better (default True).
    """

    def __init__(
        self,
        *,
        storage: ProgramStorage,
        metrics_context: MetricsContext,
        writer: LogWriter,
        interval: float = 5.0,
    ) -> None:
        self._storage = storage
        self._ctx = metrics_context
        self._writer = writer.bind(path=["program_metrics"])
        self._interval = interval

        self._task: asyncio.Task | None = None
        self._running = False

        # processed ids
        self._seen_ids: set[str] = set()

        # simple counters
        self._valid_count = 0
        self._invalid_count = 0

        # best frontier for VALID programs only: metric -> (best_value, at_iteration)
        self._best_valid: dict[str, tuple[float, int]] = {}

        #   iter -> metric_key -> RunningStats
        self._iter_stats: dict[int, dict[str, _RunningStats]] = {}
        #   generation -> metric_key -> RunningStats
        self._gen_stats: dict[int, dict[str, _RunningStats]] = {}

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule tracker task on the provided loop."""
        if self._task and not self._task.done():
            logger.warning("[MetricsTracker] already running")
            return
        self._running = True
        self._task = loop.create_task(self.run(), name="metrics-tracker")
        logger.info("[MetricsTracker] started (interval={}s)", self._interval)

    async def stop(self) -> None:
        """Cancel tracker task and await it."""
        self._running = False
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.info("[MetricsTracker] stopped")

    # -------- main loop --------

    async def run(self) -> None:
        try:
            while self._running:
                await self._drain_once()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[MetricsTracker] run() error")

    # -------- fetch & process --------

    async def _drain_once(self) -> None:
        all_ids: list[str] = await self._storage.get_all_program_ids()
        new_ids = [pid for pid in all_ids if pid not in self._seen_ids]
        if not new_ids:
            return

        programs: list[Program] = await self._storage.mget(new_ids)
        for prog in programs:
            if not prog:
                continue
            if await self._process_program(prog):
                self._seen_ids.add(prog.id)

    async def _process_program(self, program: Program) -> bool:
        """Process one program; returns True if metrics were written/updated."""
        if program.state in INCOMPLETE_STATES:
            return False

        metrics = program.metrics or {}
        v = metrics.get(VALIDITY_KEY)
        if v is None:
            return False

        is_valid = bool(v >= 0.5)
        iteration = program.metadata["iteration"]  # set during mutation
        generation = program.generation

        # validity flag
        self._writer.scalar(VALIDITY_KEY, 1.0 if is_valid else 0.0)

        # counts
        if is_valid:
            self._valid_count += 1
        else:
            self._invalid_count += 1
        total = self._valid_count + self._invalid_count
        self._writer.scalar("programs/valid_count", float(self._valid_count))
        self._writer.scalar("programs/invalid_count", float(self._invalid_count))
        self._writer.scalar("programs/total_count", float(total))

        if not is_valid:
            return True  # done for invalid programs

        # per-program metrics + frontier + NEW aggregates
        frontier_improved = False
        for key, val in metrics.items():
            if key == VALIDITY_KEY or not isinstance(val, (int, float)):
                continue
            fval = float(val)

            # per-program
            self._writer.scalar(f"valid/program/{key}", fval)

            # frontier
            if self._maybe_update_frontier(key, fval, iteration):
                frontier_improved = True

            istats = self._iter_stats.setdefault(iteration, {})
            rs_i = istats.get(key)
            if rs_i is None:
                rs_i = istats[key] = _RunningStats()
            rs_i.update(fval)
            self._writer.scalar(
                f"valid/iter/{key}/mean", rs_i.mean_value(), step=iteration
            )
            self._writer.scalar(
                f"valid/iter/{key}/std", rs_i.std_value(), step=iteration
            )

            gstats = self._gen_stats.setdefault(generation, {})
            rs_g = gstats.get(key)
            if rs_g is None:
                rs_g = gstats[key] = _RunningStats()
            rs_g.update(fval)
            self._writer.scalar(
                f"valid/gen/{key}/mean", rs_g.mean_value(), step=generation
            )
            self._writer.scalar(
                f"valid/gen/{key}/std", rs_g.std_value(), step=generation
            )

        if frontier_improved:
            self._write_valid_frontier()

        return True

    def _maybe_update_frontier(self, key: str, value: float, iteration: int) -> bool:
        spec = self._ctx.specs.get(key)
        higher_is_better = True if spec is None else bool(spec.higher_is_better)

        best = self._best_valid.get(key)
        if best is None:
            self._best_valid[key] = (value, iteration)
            return True

        best_val, _ = best
        improved = value > best_val if higher_is_better else value < best_val
        if improved:
            self._best_valid[key] = (value, iteration)
        return improved

    def _write_valid_frontier(self) -> None:
        for key, (val, it) in self._best_valid.items():
            self._writer.scalar(f"valid/frontier/{key}", float(val), step=int(it))
