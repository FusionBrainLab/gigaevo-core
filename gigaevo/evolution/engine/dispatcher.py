"""Long-lived dispatcher loop for the steady-state engine.

Pattern: ``while running: acquire semaphore slot; create_task(run_one_mutant);
loop``. The dispatcher never awaits the per-mutant task it spawned — that
is what makes the engine a continuous stream rather than a sequential
producer. Backpressure is enforced by the semaphore alone.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.evolution.engine.mutant_task import run_one_mutant


async def dispatcher_loop(engine) -> None:
    logger.info("[dispatcher] start")
    active: set[asyncio.Task] = set()
    failures: list[BaseException] = []
    consecutive_empty = 0
    task_id = 0
    completed_normally = False

    def task_finished(task: asyncio.Task) -> None:
        nonlocal consecutive_empty
        if task not in active:
            return
        active.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            failures.append(exc)
            return
        if task.result() is None:
            consecutive_empty += 1
            maximum = engine.config.max_consecutive_mutation_failures
            if consecutive_empty >= maximum:
                failures.append(
                    RuntimeError(
                        "mutation production returned no child "
                        f"{consecutive_empty} consecutive times"
                    )
                )
        else:
            consecutive_empty = 0

    try:
        while engine._running:
            while engine._running and engine._can_dispatch_mutant(reserved=len(active)):
                await engine._producer_sema.acquire()
                # A producer releases its slot immediately before completing. Its
                # done callback may be queued behind this waiter, so inspect the
                # task objects directly before authorizing another mutation.
                for finished in tuple(active):
                    if finished.done():
                        task_finished(finished)
                if failures:
                    engine._producer_sema.release()
                    raise failures[0]
                if not engine._running or not engine._can_dispatch_mutant(
                    reserved=len(active)
                ):
                    # Post-acquire early-stop: hand the slot back so a graceful
                    # restart finds _producer_sema at full capacity.
                    engine._producer_sema.release()
                    break
                t = asyncio.create_task(
                    run_one_mutant(engine, task_id), name=f"mutant-{task_id}"
                )
                task_id += 1
                active.add(t)
                t.add_done_callback(task_finished)

            # Reservations make a hard mutation cap exact, but a producer can
            # still return no child. Settle the final reservations and refill
            # only their shortfall before declaring dispatch complete.
            if active:
                remaining = tuple(active)
                await asyncio.gather(*remaining, return_exceptions=True)
                for finished in remaining:
                    task_finished(finished)
            if failures:
                raise failures[0]
            if not engine._can_dispatch_mutant(reserved=0):
                break
        completed_normally = True
    finally:
        if not completed_normally:
            for t in active:
                t.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
        logger.info("[dispatcher] stop")


__all__ = ["dispatcher_loop"]
