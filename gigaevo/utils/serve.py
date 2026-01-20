import asyncio
from collections.abc import Awaitable, Iterable
import contextlib
import signal
import sys


def _supports_signal_handlers() -> bool:
    """Check if the platform supports asyncio signal handlers."""
    return sys.platform != "win32"


async def serve_until_signal(
    *,
    stop_coros: Iterable[Awaitable] = (),
    on_stop: Iterable[asyncio.Future] = (),
) -> None:
    """
    Wait until SIGINT/SIGTERM or any task in on_stop completes naturally, then:
      1) await all stop coroutines (e.g., engine.stop(), dag.stop())
      2) cancel & await any provided task handles

    Works on both Unix and Windows platforms.
    """
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _set() -> None:
        if not stop_event.is_set():
            stop_event.set()

    # Store original signal handlers for Windows cleanup
    original_sigint = None
    original_sigterm = None
    use_loop_handlers = _supports_signal_handlers()

    if use_loop_handlers:
        # Unix: use asyncio's signal handlers
        loop.add_signal_handler(signal.SIGINT, _set)
        loop.add_signal_handler(signal.SIGTERM, _set)
    else:
        # Windows: use signal.signal() with thread-safe callback
        def _signal_handler(signum: int, frame: object) -> None:
            # call_soon_threadsafe is safe to call from signal handlers
            loop.call_soon_threadsafe(_set)

        original_sigint = signal.signal(signal.SIGINT, _signal_handler)
        # SIGTERM may not be available on all Windows configurations
        try:
            original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
        except (OSError, ValueError):
            pass  # SIGTERM not supported on this platform

    try:
        # Block here until a signal arrives OR any monitored task completes
        tasks_to_monitor = [t for t in on_stop if t and not t.done()]
        if tasks_to_monitor:
            wait_tasks = [asyncio.create_task(stop_event.wait())] + tasks_to_monitor
            done, pending = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            # If stop_event wasn't the one that completed, set it now to proceed with cleanup
            if not stop_event.is_set():
                stop_event.set()
            # Cancel the stop_event.wait() task if it's still pending
            for task in pending:
                if not task.done():
                    task.cancel()
        else:
            await stop_event.wait()

        # 1) run component stop coroutines (in parallel)
        if stop_coros:
            await asyncio.gather(*stop_coros, return_exceptions=True)

        # Let any follow-up tasks created by stop() schedule
        await asyncio.sleep(0)

        # 2) cancel & drain provided task handles
        pending_futures: list[asyncio.Future] = []
        for h in on_stop:
            if h is None or h.done():
                continue
            if isinstance(h, asyncio.Task):
                h.cancel()
            pending_futures.append(h)

        if pending_futures:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending_futures, return_exceptions=True)

        # Give the loop a final turn for late callbacks created during cancellation
        await asyncio.sleep(0)

    finally:
        # Always remove/restore handlers to avoid leaks
        if use_loop_handlers:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        else:
            # Restore original signal handlers on Windows
            if original_sigint is not None:
                signal.signal(signal.SIGINT, original_sigint)
            if original_sigterm is not None:
                signal.signal(signal.SIGTERM, original_sigterm)
