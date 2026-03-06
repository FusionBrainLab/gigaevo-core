from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
from typing import Any

from loguru import logger

from gigaevo.utils.trackers.base import LogWriter


async def _maybe_await(x: Any) -> Any:
    """Await x if it's awaitable; otherwise return it."""
    return await x if inspect.isawaitable(x) else x


def start_metrics_collector(
    *,
    writer: LogWriter,
    collect_fn: Callable[[], dict[str, Any] | "asyncio.Future[dict[str, Any]]"],
    interval: float,
    stop_flag: Callable[[], bool],
    task_name: str = "metrics-collector",  # kept for backward compat; used as task name
) -> asyncio.Task:
    """
    Start an async task that periodically collects and writes scalar metrics.

    - Schedules an asyncio.Task on the current running loop.
    - Every `interval` seconds (monotonic), calls `collect_fn()` (sync or async).
    - Writes only numeric/bool values via writer.scalar(key, float(value)).
    - Stops when `stop_flag()` returns True. Exceptions are swallowed to stay minimal.

    Returns:
        asyncio.Task
    """
    loop = asyncio.get_running_loop()

    async def _run() -> None:
        next_tick = loop.time()
        while not stop_flag():
            try:
                metrics = await _maybe_await(collect_fn())
                if metrics:
                    for k, v in metrics.items():
                        if not isinstance(v, (int, float, bool)):
                            continue
                        writer.scalar(k, float(v))
            except Exception as e:
                logger.debug("Error collecting metrics: {}", e)
                await asyncio.sleep(interval)
                continue

            next_tick += interval
            await asyncio.sleep(max(0.0, next_tick - loop.time()))

    return loop.create_task(_run(), name=task_name)
