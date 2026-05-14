"""Live frontier-comparison daemon — periodic in-run snapshot of how the
current population compares to the frontier (best-so-far / hall-of-fame).

Sibling of :mod:`gigaevo.monitoring.live_profiler`. While the profiler
re-renders ``profile_live.html`` for visual inspection, this loop emits a
compact text snapshot via loguru (and optionally Telegram) so you can
see — without leaving the terminal — whether the newest mutants are
catching up to, matching, or surpassing the running frontier.

Architecture
------------

The loop runs in a daemon thread, parallel to ``start_live_profiler``.
It reads two series per metric from Redis (written by
:class:`gigaevo.utils.metrics_tracker.MetricsTracker`):

* ``valid/frontier/<metric>`` — best-so-far per iteration (the "HoF").
* ``valid/iter/<metric>/mean`` — per-iteration mean over valid programs.
* ``valid/program/<metric>`` — per-program values, used to compute
  current-iteration best.

The thread is daemonic (no graceful shutdown is required) and every tick
is fully self-contained — a Redis error on one tick never poisons the
next, mirroring the resilience pattern of the live profiler.

Usage::

    from gigaevo.monitoring.live_frontier_compare import (
        start_live_frontier_compare,
    )
    stop = start_live_frontier_compare(
        redis_url="redis://localhost:6379/0",
        key_prefix="heilbron:metrics",
        metrics=["fitness"],
        higher_is_better={"fitness": True},
        interval_s=60.0,
    )

The returned :class:`threading.Event` can be ``set()`` to ask the loop
to exit; this is optional because the thread is daemonic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import threading
import time

from loguru import logger

# ---------------------------------------------------------------------------
# Pure data classes + compute helper (test-friendly, no I/O).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricComparison:
    """One metric's current-vs-frontier snapshot."""

    name: str
    current_best: float
    current_mean: float
    frontier_best: float
    frontier_mean: float
    delta_best: float
    delta_mean: float
    # "+" when the current value is on the *improving* side of the
    # frontier given the metric's higher_is_better orientation, "-"
    # otherwise, "0" for an exact match.
    delta_best_sign: str


@dataclass(frozen=True)
class FrontierCompareSnapshot:
    """One tick's snapshot — per-metric comparisons."""

    metrics: dict[str, MetricComparison] = field(default_factory=dict)


def _select_best(values: Sequence[float], higher_is_better: bool) -> float | None:
    """Pick the better extremum of *values* given the optimization direction."""
    if not values:
        return None
    return max(values) if higher_is_better else min(values)


def _latest_iteration_values(
    points: Sequence[tuple[int, float]],
) -> tuple[int | None, list[float]]:
    """Return ``(iteration, values)`` for the most-recent iteration in *points*.

    *points* is an unsorted ``(iteration, value)`` series; we pick the
    largest iteration index and gather all values at it.
    """
    if not points:
        return None, []
    latest_iter = max(it for it, _ in points)
    return latest_iter, [v for it, v in points if it == latest_iter]


def _improvement_sign(delta: float, higher_is_better: bool) -> str:
    """Sign of an *improvement*, accounting for optimization direction."""
    if delta == 0:
        return "0"
    if higher_is_better:
        return "+" if delta > 0 else "-"
    return "+" if delta < 0 else "-"


def compute_snapshot(
    *,
    metrics: Sequence[str],
    frontier_history: dict[str, list[tuple[int, float]]],
    iter_mean_history: dict[str, list[tuple[int, float]]],
    program_history: dict[str, list[tuple[int, float]]],
    higher_is_better: dict[str, bool],
) -> FrontierCompareSnapshot:
    """Compute a comparison snapshot from raw per-metric histories.

    All three histories are ``(iteration, value)`` lists. A metric is
    skipped if either its frontier or its program/iter history is empty.

    The current "best" is the extremum across the *latest* iteration's
    program values; "frontier best" is the last entry of the frontier
    series. The current "mean" is the latest per-iteration mean;
    "frontier mean" is the mean over the frontier series itself
    (average best-so-far across observed iterations).
    """
    out: dict[str, MetricComparison] = {}
    for name in metrics:
        higher = higher_is_better.get(name, True)
        front = frontier_history.get(name) or []
        if not front:
            continue

        # Frontier best = most-recent frontier value.
        front.sort(key=lambda x: x[0])
        frontier_best = float(front[-1][1])
        frontier_mean = sum(v for _, v in front) / len(front)

        # Current iteration's best, from the per-program series.
        progs = program_history.get(name) or []
        _, latest_vals = _latest_iteration_values(progs)
        current_best = _select_best(latest_vals, higher)

        # Current mean = latest per-iter mean entry, else fall back to
        # the iteration's program-value mean if the iter-mean tag hasn't
        # ticked yet on this iteration.
        iters = iter_mean_history.get(name) or []
        iters_sorted = sorted(iters, key=lambda x: x[0])
        if iters_sorted:
            current_mean = float(iters_sorted[-1][1])
        elif latest_vals:
            current_mean = sum(latest_vals) / len(latest_vals)
        else:
            current_mean = None  # type: ignore[assignment]

        if current_best is None or current_mean is None:
            continue

        delta_best = float(current_best) - frontier_best
        delta_mean = float(current_mean) - frontier_mean
        out[name] = MetricComparison(
            name=name,
            current_best=float(current_best),
            current_mean=float(current_mean),
            frontier_best=float(frontier_best),
            frontier_mean=float(frontier_mean),
            delta_best=delta_best,
            delta_mean=delta_mean,
            delta_best_sign=_improvement_sign(delta_best, higher),
        )
    return FrontierCompareSnapshot(metrics=out)


def format_snapshot(snap: FrontierCompareSnapshot, *, decimals: int = 5) -> str:
    """Render a snapshot as a single compact human-readable line."""
    if not snap.metrics:
        return "[live_frontier_compare] (no frontier data yet)"
    parts = []
    fmt = f".{decimals}f"
    for name in sorted(snap.metrics):
        c = snap.metrics[name]
        parts.append(
            f"{name}: current_best={c.current_best:{fmt}} "
            f"frontier_best={c.frontier_best:{fmt}} "
            f"delta_best={c.delta_best:+{fmt}} ({c.delta_best_sign}) | "
            f"current_mean={c.current_mean:{fmt}} "
            f"frontier_mean={c.frontier_mean:{fmt}} "
            f"delta_mean={c.delta_mean:+{fmt}}"
        )
    return "[live_frontier_compare] " + " || ".join(parts)


# ---------------------------------------------------------------------------
# Redis adapter — fetches the three series the snapshot needs.
# ---------------------------------------------------------------------------


def _history_key(key_prefix: str, tag: str) -> str:
    """Redis key for a tracker tag, matching RedisMetricsBackend._k_history.

    The backend sanitises ``/`` → ``:`` and `` `` → ``_`` when building
    the history list key.
    """
    safe = tag.replace("/", ":").replace(" ", "_")
    return f"{key_prefix}:history:{safe}"


def _parse_series(raw_entries: list) -> list[tuple[int, float]]:
    """Parse a Redis history list (JSON ``{"s": step, "v": value, ...}``)."""
    out: list[tuple[int, float]] = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
            step = entry.get("s")
            value = entry.get("v")
            if step is None or value is None:
                continue
            out.append((int(step), float(value)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def _fetch_histories(
    client,
    key_prefix: str,
    metrics: Sequence[str],
) -> tuple[
    dict[str, list[tuple[int, float]]],
    dict[str, list[tuple[int, float]]],
    dict[str, list[tuple[int, float]]],
]:
    """Pull frontier / iter-mean / per-program series for each metric."""
    frontier: dict[str, list[tuple[int, float]]] = {}
    iter_mean: dict[str, list[tuple[int, float]]] = {}
    program: dict[str, list[tuple[int, float]]] = {}
    for m in metrics:
        # See gigaevo/utils/trackers/core.py _render_tag: per-piece
        # sanitisation strips ``/`` to ``_`` within the metric name, then
        # joins ``path`` + metric with ``/``. Then RedisMetricsBackend
        # converts ``/`` to ``:``. Net key parts:
        #   path = ["program_metrics"], metric = "valid/frontier/<m>"
        #   → tag "program_metrics/valid_frontier_<m>"
        #   → key "{prefix}:history:program_metrics:valid_frontier_<m>"
        frontier_key = _history_key(key_prefix, f"program_metrics/valid_frontier_{m}")
        iter_mean_key = _history_key(key_prefix, f"program_metrics/valid_iter_{m}_mean")
        program_key = _history_key(key_prefix, f"program_metrics/valid_program_{m}")
        frontier[m] = _parse_series(client.lrange(frontier_key, 0, -1))
        iter_mean[m] = _parse_series(client.lrange(iter_mean_key, 0, -1))
        program[m] = _parse_series(client.lrange(program_key, 0, -1))
    return frontier, iter_mean, program


def _emit(
    snap: FrontierCompareSnapshot,
    *,
    emit_log: bool,
    emit_telegram: bool,
    label: str,
) -> None:
    line = format_snapshot(snap)
    if emit_log:
        logger.info("[{}] {}", label, line)
    if emit_telegram and snap.metrics:
        try:
            # Lazy import to avoid a hard dependency on the tools.* package
            # when Telegram emission is disabled.
            from tools.telegram_notify import notify

            notify(line, parse_mode="")
        except Exception:
            logger.opt(exception=True).debug(
                "[live_frontier_compare] telegram emit failed (will retry next tick)"
            )


def _loop(
    *,
    redis_url: str,
    key_prefix: str,
    metrics: Sequence[str],
    higher_is_better: dict[str, bool],
    interval_s: float,
    emit_log: bool,
    emit_telegram: bool,
    label: str,
    stop: threading.Event,
) -> None:
    """Run-loop: open Redis (lazy), tick at ``interval_s`` until stopped."""
    # Lazy import — Redis is a heavy dependency at module import time on
    # constrained CI machines.
    import redis as redis_lib

    client = None
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            if client is None:
                client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
            frontier, iter_mean, program = _fetch_histories(client, key_prefix, metrics)
            snap = compute_snapshot(
                metrics=metrics,
                frontier_history=frontier,
                iter_mean_history=iter_mean,
                program_history=program,
                higher_is_better=higher_is_better,
            )
            _emit(
                snap,
                emit_log=emit_log,
                emit_telegram=emit_telegram,
                label=label,
            )
            logger.debug(
                "[live_frontier_compare] tick in {:.2f}s ({} metrics with data)",
                time.monotonic() - t0,
                len(snap.metrics),
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[live_frontier_compare] tick failed (will retry next tick)"
            )
            # Drop the (possibly broken) Redis client so the next tick
            # re-opens a fresh connection.
            client = None
        if stop.wait(interval_s):
            break


# ---------------------------------------------------------------------------
# Public entry point — mirrors start_live_profiler's surface.
# ---------------------------------------------------------------------------


def start_live_frontier_compare(
    *,
    redis_url: str,
    key_prefix: str,
    metrics: Sequence[str],
    higher_is_better: dict[str, bool],
    interval_s: float = 60.0,
    emit_targets: Sequence[str] = ("log",),
    label: str = "live_frontier_compare",
    enabled: bool = True,
) -> threading.Event:
    """Start a daemon thread emitting periodic frontier-comparison snapshots.

    Parameters:
        redis_url: Redis connection URL (e.g. ``redis://host:6379/0``).
            Must point at the *same* DB as the run's metrics tracker.
        key_prefix: prefix used by the metrics tracker
            (``${problem.name}:metrics`` by default).
        metrics: list of metric names to compare (e.g. ``["fitness"]``).
            Names must match the keys written by
            :class:`gigaevo.utils.metrics_tracker.MetricsTracker`.
        higher_is_better: per-metric optimization direction. Pulled from
            the problem's ``metrics.yaml`` ``MetricSpec`` at wiring time.
        interval_s: seconds between snapshots. 60 s is a reasonable
            default; the read is two ``LRANGE`` calls per metric.
        emit_targets: subset of ``("log", "telegram")``. ``log`` writes
            via loguru at INFO; ``telegram`` calls
            :func:`tools.telegram_notify.notify`.
        label: short identifier used in log lines (and as the thread
            name).
        enabled: when ``False``, returns a set ``Event`` without starting
            a thread. Lets callers gate via Hydra without branching.

    Returns:
        A :class:`threading.Event` that callers can ``set()`` to stop the
        loop. The thread is daemonic, so this is optional.
    """
    stop = threading.Event()
    if not enabled:
        logger.info("[live_frontier_compare] disabled via config")
        return stop

    emit_set = {t.lower() for t in emit_targets}
    emit_log = "log" in emit_set
    emit_telegram = "telegram" in emit_set

    thread = threading.Thread(
        target=_loop,
        kwargs=dict(
            redis_url=redis_url,
            key_prefix=key_prefix,
            metrics=list(metrics),
            higher_is_better=dict(higher_is_better),
            interval_s=float(interval_s),
            emit_log=emit_log,
            emit_telegram=emit_telegram,
            label=label,
            stop=stop,
        ),
        name="live-frontier-compare",
        daemon=True,
    )
    thread.start()
    logger.info(
        "[live_frontier_compare] started "
        "(prefix={}, metrics={}, every {:.0f}s, targets={})",
        key_prefix,
        list(metrics),
        interval_s,
        sorted(emit_set),
    )
    return stop
