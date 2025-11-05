import asyncio
import contextlib
import signal
from typing import Awaitable, Iterable


async def serve_until_signal(
    *,
    stop_coros: Iterable[Awaitable] = (),
    on_stop: Iterable[asyncio.Future] = (),
) -> None:
    """
    Wait until SIGINT/SIGTERM, then:
      1) await all stop coroutines (e.g., engine.stop(), dag.stop())
      2) cancel & await any provided task handles
    """
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _set() -> None:
        if not stop_event.is_set():
            stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _set)
    loop.add_signal_handler(signal.SIGTERM, _set)

    try:
        # Block here until a signal arrives
        await stop_event.wait()

        # 1) run component stop coroutines (in parallel)
        if stop_coros:
            await asyncio.gather(*stop_coros, return_exceptions=True)

        # Let any follow-up tasks created by stop() schedule
        await asyncio.sleep(0)

        # 2) cancel & drain provided task handles
        pending: list[asyncio.Future] = []
        for h in on_stop:
            if h is None or h.done():
                continue
            if isinstance(h, asyncio.Task):
                h.cancel()
            pending.append(h)

        if pending:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)

        # Give the loop a final turn for late callbacks created during cancellation
        await asyncio.sleep(0)

    finally:
        # Always remove handlers to avoid leaks
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)
