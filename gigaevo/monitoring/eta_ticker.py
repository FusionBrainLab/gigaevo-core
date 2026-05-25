"""Live ETA daemon — periodically log estimated time to completion.

Driven by observed throughput (mutations per second) and the stopper's
estimate_remaining contract. Skips log while warming up (< 3 mutants)
to avoid noisy early estimates dominated by initial evaluation.

Usage::

    from gigaevo.monitoring.eta_ticker import start_eta_ticker
    start_eta_ticker(evolution_engine, interval_s=60.0)
"""

from __future__ import annotations

import threading

from loguru import logger

from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.stopper import EngineThroughput


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


def _tick(
    engine: EvolutionEngine,
    *,
    warmup_mutants: int = 3,
) -> str | None:
    """Compute one ETA tick, returning the log line or None during warmup."""
    ctx = engine.build_stop_context()

    if ctx.total_mutants < warmup_mutants:
        return None

    if ctx.elapsed_seconds <= 0:
        return None

    throughput = EngineThroughput(
        mutants_per_second=ctx.total_mutants / ctx.elapsed_seconds,
        elapsed_seconds=ctx.elapsed_seconds,
    )

    est_remaining = engine.stopper.estimate_remaining(ctx, throughput)

    elapsed_str = _humanize_seconds(ctx.elapsed_seconds)
    rate = (ctx.total_mutants / ctx.elapsed_seconds) * 60  # per minute

    if est_remaining is None:
        # Unbounded stopper — find a label.
        label = _unbounded_label(engine.stopper)
        return f"[eta] elapsed={elapsed_str} | mutants={ctx.total_mutants} ({rate:.1f}/min) | ETA=unknown (unbounded: {label})"

    remaining_s, stopper_label = est_remaining
    remaining_str = _humanize_seconds(remaining_s)
    return f"[eta] elapsed={elapsed_str} | mutants={ctx.total_mutants} ({rate:.1f}/min) | remaining={int(remaining_s // throughput.mutants_per_second) if throughput.mutants_per_second > 0 else '?'} | ETA={remaining_str} (bound: {stopper_label})"


def _unbounded_label(stopper) -> str:
    """Extract label from first unbounded child stopper, or fallback."""
    from gigaevo.evolution.engine.stopper import CompositeStopper

    if not isinstance(stopper, CompositeStopper):
        return type(stopper).__name__

    if not stopper.children:
        return "Unknown"

    # Find first unbounded child.
    for child in stopper.children:
        ctx_dummy = type(
            "DummyCtx",
            (),
            {
                "total_mutants": 0,
                "elapsed_seconds": 0.0,
                "best_fitness": None,
                "programs_processed": 0,
            },
        )()
        tp_dummy = EngineThroughput(mutants_per_second=0.0, elapsed_seconds=0.0)
        if child.estimate_remaining(ctx_dummy, tp_dummy) is None:
            return type(child).__name__

    return type(stopper.children[0]).__name__


def _loop(
    engine: EvolutionEngine,
    interval_s: float,
    stop: threading.Event,
    *,
    warmup_mutants: int,
) -> None:
    """Run-loop: periodically emit ETA line."""
    while not stop.is_set():
        try:
            line = _tick(engine, warmup_mutants=warmup_mutants)
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
) -> threading.Event:
    """Start a daemon thread that periodically logs ETA.

    Parameters:
        engine: the EvolutionEngine being run.
        interval_s: seconds between ETA log lines. Defaults to 60.0.
        warmup_mutants: number of mutants to reach before first log.
            Defaults to 3 (early throughput is noisy).

    Returns:
        A :class:`threading.Event` you can ``set()`` to ask the loop to
        exit. The thread is daemonic, so this is optional — process exit
        will kill it anyway.
    """
    stop = threading.Event()
    thread = threading.Thread(
        target=_loop,
        args=(engine, interval_s, stop),
        kwargs=dict(warmup_mutants=warmup_mutants),
        name="eta-ticker",
        daemon=True,
    )
    thread.start()
    return stop
