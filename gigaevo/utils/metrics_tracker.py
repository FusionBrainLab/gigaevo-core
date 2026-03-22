from __future__ import annotations

import asyncio
from contextlib import suppress
import math

from loguru import logger
from pydantic import BaseModel

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.metrics.context import (
    DEFAULT_DECIMALS,
    VALIDITY_KEY,
    MetricsContext,
)
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
      - Skips running DAGs (QUEUED / RUNNING) or without metrics/validity.
      - Writes:
          * "is_valid" (for all)
          * per-program metrics for valid programs: "valid/program/<metric>"
          * counts: programs/{valid_count, invalid_count, total_count}
          * frontier for valid only: "valid/frontier/<metric>" (step = iteration)
          * NEW (valid only):
              - per-iteration aggregates: "valid/iter/<metric>/{mean,std}" (step = iteration)
              - per-generation aggregates: "valid/gen/<metric>/{mean,std}" (step = generation)
      - Frontier uses MetricsContext.specs[metric].higher_is_better (default True).
      - Frontier is the single source of truth: when NO_CACHE stages update
        metrics, the full frontier series is recomputed from all valid programs
        and rewritten to the backend.
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
        # last-seen fitness for change detection (handles NO_CACHE stages)
        self._seen_fitness: dict[str, tuple[tuple[str, float], ...]] = {}

        # simple counters
        self._valid_count = 0
        self._invalid_count = 0

        # best frontier for VALID programs only: metric -> (best_value, at_iteration)
        self._best_valid: dict[str, tuple[float, int]] = {}

        # all valid programs: program_id -> (iteration, {metric_key: value})
        # used for full frontier recomputation when NO_CACHE stages change metrics
        self._valid_programs: dict[str, tuple[int, dict[str, float]]] = {}

        #   iter -> metric_key -> RunningStats
        self._iter_stats: dict[int, dict[str, _RunningStats]] = {}
        #   generation -> metric_key -> RunningStats
        self._gen_stats: dict[int, dict[str, _RunningStats]] = {}

    @property
    def metrics_context(self) -> MetricsContext:
        """Expose MetricsContext for formatting (e.g. gen summary logging)."""
        return self._ctx

    def format_best_summary(self) -> str:
        """Return a compact string of best frontier values for log lines.

        Uses MetricsContext for decimal precision. Reads from the in-memory
        ``_best_valid`` cache — no Redis fetch required. Gracefully handles
        metric keys not in the context specs.
        """
        if not self._best_valid:
            return ""
        parts = []
        for key in sorted(self._best_valid):
            best_val, _ = self._best_valid[key]
            spec = self._ctx.specs.get(key)
            decimals = spec.decimals if spec else DEFAULT_DECIMALS
            parts.append(f"{key}={best_val:.{decimals}f}")
        return " best=[" + ", ".join(parts) + "]"

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

        # Process new programs (first time seen)
        new_ids = [pid for pid in all_ids if pid not in self._seen_ids]
        if new_ids:
            programs: list[Program] = await self._storage.mget(new_ids)
            for prog in programs:
                if not prog:
                    continue
                if await self._process_program(prog):
                    self._seen_ids.add(prog.id)
                    metrics = prog.metrics or {}
                    metrics_hash = tuple(
                        (k, v)
                        for k, v in sorted(metrics.items())
                        if isinstance(v, (int, float)) and k != VALIDITY_KEY
                    )
                    self._seen_fitness[prog.id] = metrics_hash

        # Re-check already-seen programs for fitness changes
        # (handles NO_CACHE stages like PromptFitnessStage that update
        # metrics after archive refresh re-runs)
        await self._refresh_changed_fitness()

    async def _refresh_changed_fitness(self) -> None:
        if not self._seen_fitness:
            return

        programs = await self._storage.mget(list(self._seen_fitness.keys()))
        changed_keys: set[str] = set()

        for prog in programs:
            if not prog or prog.state in INCOMPLETE_STATES:
                continue
            metrics = prog.metrics or {}

            # Check if any numeric metric changed since last processing
            metrics_hash = tuple(
                (k, v)
                for k, v in sorted(metrics.items())
                if isinstance(v, (int, float)) and k != VALIDITY_KEY
            )
            old_hash = self._seen_fitness.get(prog.id)
            if old_hash is not None and metrics_hash == old_hash:
                continue

            # Metrics changed — update stored value and valid_programs
            self._seen_fitness[prog.id] = metrics_hash
            iteration = prog.metadata.get("iteration", 0)

            numeric_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if k != VALIDITY_KEY and isinstance(v, (int, float))
            }
            self._valid_programs[prog.id] = (iteration, numeric_metrics)
            changed_keys.update(numeric_metrics.keys())

        # Full frontier recompute for every metric that had any program change
        if changed_keys:
            for key in changed_keys:
                self._recompute_and_write_frontier(key)

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

        # Store program data for frontier recomputation
        numeric_metrics = {
            k: float(val)
            for k, val in metrics.items()
            if k != VALIDITY_KEY and isinstance(val, (int, float))
        }
        self._valid_programs[program.id] = (iteration, numeric_metrics)

        # per-program metrics + frontier + aggregates
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

    # -------- full frontier recomputation --------

    def _recompute_and_write_frontier(self, metric_key: str) -> None:
        """Recompute the entire frontier series for *metric_key* from
        ``_valid_programs`` and rewrite the backend history.

        Called when ``_refresh_changed_fitness`` detects that a NO_CACHE stage
        has changed program metrics, which may invalidate previously-written
        frontier entries (e.g. the program that held the frontier got worse).
        """
        spec = self._ctx.specs.get(metric_key)
        higher_is_better = True if spec is None else bool(spec.higher_is_better)

        # Collect per-iteration best from all valid programs
        iter_best: dict[int, float] = {}
        for _pid, (iteration, metrics) in self._valid_programs.items():
            val = metrics.get(metric_key)
            if val is None:
                continue
            cur = iter_best.get(iteration)
            if cur is None:
                iter_best[iteration] = val
            elif (higher_is_better and val > cur) or (
                not higher_is_better and val < cur
            ):
                iter_best[iteration] = val

        if not iter_best:
            self._best_valid.pop(metric_key, None)
            return

        # Sort by iteration and compute cumulative frontier
        sorted_iters = sorted(iter_best.items())  # (iteration, best_val)
        frontier_series: list[tuple[int, float]] = []
        best_so_far = sorted_iters[0][1]
        for it, val in sorted_iters:
            if higher_is_better:
                best_so_far = max(best_so_far, val)
            else:
                best_so_far = min(best_so_far, val)
            frontier_series.append((it, best_so_far))

        # Update _best_valid to match recomputed frontier
        self._best_valid[metric_key] = (frontier_series[-1][1], frontier_series[-1][0])

        # Clear the old series and write the full recomputed frontier
        tag = f"valid/frontier/{metric_key}"
        self._writer.clear_series(tag)
        for it, val in frontier_series:
            self._writer.scalar(tag, float(val), step=int(it))
