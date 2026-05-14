"""Live flow-profiler daemon — periodically re-render the running log to HTML.

Drops a single helper into ``run.py`` so you can watch ``profile_live.html``
in a browser while the experiment is mutating. The thread is daemonic
(no graceful shutdown is required — process exit kills it) and every
render iteration is fully self-contained, so a parsing error on one tick
never poisons the next.

Usage::

    from gigaevo.monitoring.live_profiler import start_live_profiler
    start_live_profiler(log_path, out_dir, interval_s=60)

The HTML is written atomically (``.tmp`` then ``os.replace``) so a
browser that reloads mid-write never sees a half-flushed file.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

from loguru import logger

from gigaevo.monitoring.flow_profiler import (
    compute_saturation,
    compute_utilization,
    parse_log,
    render_full_html,
)


def _render_once(log_path: Path, html_path: Path, label: str) -> tuple[int, int]:
    """One render iteration. Returns ``(n_programs, n_llm_events)``."""
    programs, refreshes, llm_events, backpressure = parse_log(log_path)
    util = compute_utilization(programs, refreshes, llm_events)
    sat = compute_saturation(backpressure)
    html = render_full_html(
        programs,
        refreshes,
        title=f"flow profile · {label} (live)",
        subtitle=str(log_path),
        div_id=f"gigaevo-flow-{label}",
        utilization=util,
        backpressure=backpressure,
        saturation=sat,
    )
    tmp = html_path.with_suffix(html_path.suffix + ".tmp")
    tmp.write_text(html)
    os.replace(tmp, html_path)
    return len(programs), len(llm_events)


def _loop(
    log_path: Path,
    html_path: Path,
    label: str,
    interval_s: float,
    stop: threading.Event,
) -> None:
    """Run-loop: wait until the log file exists, then re-render on tick."""
    # Hold off until the writer has actually created the file. setup_logger
    # opens the sink lazily on first emit, so the path may briefly not exist.
    while not log_path.exists() and not stop.wait(1.0):
        pass
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            n_prog, n_llm = _render_once(log_path, html_path, label)
            logger.debug(
                "[live_profiler] rendered {} ({} programs, {} LLM events) in {:.2f}s",
                html_path,
                n_prog,
                n_llm,
                time.monotonic() - t0,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[live_profiler] render failed (will retry next tick)"
            )
        if stop.wait(interval_s):
            break


def start_live_profiler(
    log_path: str | Path,
    out_dir: str | Path,
    *,
    label: str = "live",
    interval_s: float = 60.0,
    filename: str = "profile_live.html",
) -> threading.Event:
    """Start a daemon thread that periodically re-renders the profiler HTML.

    Parameters:
        log_path: path to the loguru log file the run is writing to.
        out_dir: directory in which to place the rendered HTML (created if
            it does not exist).
        label: short identifier used in the page title and div id.
        interval_s: seconds between re-renders. 60s is a good default —
            parsing a 70MB log takes ~1s and the page is fully usable
            without redraw.
        filename: output filename inside ``out_dir``.

    Returns:
        A :class:`threading.Event` you can ``set()`` to ask the loop to
        exit. The thread is daemonic, so this is optional — process exit
        will kill it anyway.
    """
    log_path = Path(log_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / filename
    stop = threading.Event()
    thread = threading.Thread(
        target=_loop,
        args=(log_path, html_path, label, float(interval_s), stop),
        name="live-profiler",
        daemon=True,
    )
    thread.start()
    logger.info(
        "[live_profiler] watching {} -> {} (every {:.0f}s)",
        log_path,
        html_path,
        interval_s,
    )
    return stop
