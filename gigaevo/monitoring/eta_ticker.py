"""Live ETA daemon — periodically log estimated time to completion.

Driven by observed throughput (mutants per second) and the stopper's
estimate_remaining contract.

Throughput is measured over a **trailing window**, not over the run's
lifetime. A lifetime average divides a mutant count by a wall-clock span
the count does not cover:

* on resume the engine restores ``total_mutants`` from the snapshot while
  the run clock restarts, so an all-time count would be divided by this
  process's seconds alone;
* the initial-population drain burns wall time before any mutant exists,
  and those seconds never leave a lifetime denominator.

Both inflate or deflate the estimate for the whole run. Differencing two
samples inside the window cancels the restored offset outright and ages
the seed drain out one window after it ends; it also lets the estimate
follow a real throughput change instead of smearing it.

Usage::

    from gigaevo.monitoring.eta_ticker import start_eta_ticker
    start_eta_ticker(evolution_engine, interval_s=60.0)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

from loguru import logger

from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.stopper import (
    EngineThroughput,
    EvolutionStopper,
    StopContext,
)

DEFAULT_WINDOW_SECONDS = 600.0


def _humanize_seconds(s: float) -> str:
    """Format seconds as H:MM:SS (if >= 3600s) or MMmSSs (otherwise)."""
    if s >= 3600:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h}:{m:02d}:{sec:02d}"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}m{sec:02d}s"


@dataclass(frozen=True)
class ThroughputSpan:
    """Mutants created over a trailing stretch of wall clock."""

    seconds: float
    mutants: int


class ThroughputWindow:
    """Trailing-window view of mutant creation.

    Holds ``(elapsed_seconds, total_mutants)`` samples and reports the
    delta across the oldest sample still covering the window. Both fields
    are cumulative and monotonic, so their difference is the work done in
    the span regardless of what either counter started at.
    """

    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._samples: deque[tuple[float, int]] = deque()

    def observe(self, elapsed_seconds: float, total_mutants: int) -> ThroughputSpan:
        """Record a sample and return the span it closes."""
        self._samples.append((elapsed_seconds, total_mutants))
        cutoff = elapsed_seconds - self.window_seconds
        # Keep the newest sample at or before the cutoff as the anchor, so
        # the span still covers the full window. Dropping it instead would
        # shorten the span and spike the rate.
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        anchor_seconds, anchor_mutants = self._samples[0]
        return ThroughputSpan(
            seconds=elapsed_seconds - anchor_seconds,
            mutants=total_mutants - anchor_mutants,
        )


def _tick(
    engine: EvolutionEngine,
    window: ThroughputWindow,
    *,
    warmup_mutants: int = 3,
) -> str | None:
    """Compute one ETA tick, returning the log line or None during warmup."""
    ctx = engine.build_stop_context()
    span = window.observe(ctx.elapsed_seconds, ctx.total_mutants)

    if span.seconds <= 0:
        return None

    # Before the window fills, a couple of mutants is too thin to divide by.
    # Once it is full the estimate is as good as it gets, so always speak —
    # including when the answer is "nothing is being produced".
    if span.seconds < window.window_seconds and span.mutants < warmup_mutants:
        return None

    span_str = _humanize_seconds(span.seconds)
    # A stalled span still gets put to the stopper: a wall-clock bound is
    # throughput-independent, so its ETA stays valid while nothing is being
    # produced. Only the projections that need a rate drop out.
    mutants_per_second = span.mutants / span.seconds if span.mutants > 0 else 0.0
    throughput = EngineThroughput(
        mutants_per_second=mutants_per_second,
        elapsed_seconds=ctx.elapsed_seconds,
    )
    head = (
        f"[eta] elapsed={_humanize_seconds(ctx.elapsed_seconds)} "
        f"| mutants={ctx.total_mutants} ({mutants_per_second * 60:.1f}/min)"
    )

    est_remaining = engine.stopper.estimate_remaining(ctx, throughput)

    if est_remaining is None:
        if span.mutants <= 0:
            return f"{head} | ETA=unknown (no mutants in last {span_str})"
        label = _unbounded_label(engine.stopper, ctx, throughput)
        return f"{head} | ETA=unknown (unbounded: {label})"

    remaining_s, stopper_label = est_remaining
    remaining_str = _humanize_seconds(remaining_s)
    remaining_mutants = int(round(remaining_s * mutants_per_second))
    return f"{head} | remaining={remaining_mutants} | ETA={remaining_str} (bound: {stopper_label}, rate over {span_str})"


def _unbounded_label(
    stopper: EvolutionStopper,
    ctx: StopContext,
    tp: EngineThroughput,
) -> str:
    """Extract label from first unbounded child stopper, or fallback.

    Probes with the live context and throughput: a bounded stopper can
    still decline to estimate under a degenerate throughput, so a synthetic
    zero-rate probe would misreport it as the unbounded one.
    """
    from gigaevo.evolution.engine.stopper import CompositeStopper

    if not isinstance(stopper, CompositeStopper):
        return type(stopper).__name__

    if not stopper.children:
        return "Unknown"

    # Find first unbounded child.
    for child in stopper.children:
        if child.estimate_remaining(ctx, tp) is None:
            return type(child).__name__

    return type(stopper.children[0]).__name__


def _loop(
    engine: EvolutionEngine,
    interval_s: float,
    stop: threading.Event,
    *,
    warmup_mutants: int,
    window_seconds: float,
) -> None:
    """Run-loop: periodically emit ETA line."""
    window = ThroughputWindow(window_seconds)
    while not stop.is_set():
        try:
            line = _tick(engine, window, warmup_mutants=warmup_mutants)
            if line:
                logger.info(line)
        except Exception:
            logger.opt(exception=True).warning(
                "[eta_ticker] tick failed (will retry next interval)"
            )
        if stop.wait(interval_s):
            break


def start_eta_ticker(
    engine: EvolutionEngine,
    *,
    interval_s: float = 60.0,
    warmup_mutants: int = 3,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> threading.Event:
    """Start a daemon thread that periodically logs ETA.

    Parameters:
        engine: the EvolutionEngine being run.
        interval_s: seconds between ETA log lines. Defaults to 60.0.
        warmup_mutants: mutants that must accrue inside the window before
            the first log, while the window is still filling. Defaults to
            3 (early throughput is noisy).
        window_seconds: trailing span the rate is measured over. Defaults
            to 600.0. A longer window is steadier but slower to follow a
            throughput change.

    Returns:
        A :class:`threading.Event` you can ``set()`` to ask the loop to
        exit. The thread is daemonic, so this is optional — process exit
        will kill it anyway.
    """
    stop = threading.Event()
    thread = threading.Thread(
        target=_loop,
        args=(engine, interval_s, stop),
        kwargs=dict(warmup_mutants=warmup_mutants, window_seconds=window_seconds),
        name="eta-ticker",
        daemon=True,
    )
    thread.start()
    return stop
