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
from gigaevo.evolution.engine.stopper import StopDecision


async def dispatcher_loop(engine) -> StopDecision:
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
            for finished in tuple(active):
                if finished.done():
                    task_finished(finished)
            if failures:
                raise failures[0]
            if not engine._can_dispatch_mutant(reserved=len(active)):
                if getattr(engine, "_terminal_stop_decision", None) is not None:
                    break
                if active:
                    await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    continue
                break

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
            if not engine._running:
                engine._producer_sema.release()
                break
            if not engine._can_dispatch_mutant(reserved=len(active)):
                engine._producer_sema.release()
                if getattr(engine, "_terminal_stop_decision", None) is not None:
                    break
                continue
            task = asyncio.create_task(
                run_one_mutant(engine, task_id), name=f"mutant-{task_id}"
            )
            task_id += 1
            active.add(task)
            task.add_done_callback(task_finished)

        # Reaching the cap stops new production; it does not cancel producers
        # that already crossed the dispatch boundary. Their persisted children
        # must complete the handoff while the ingestor is still running.
        if active:
            remaining = tuple(active)
            await asyncio.gather(*remaining, return_exceptions=True)
            for finished in remaining:
                task_finished(finished)
        if failures:
            raise failures[0]
        decision = getattr(engine, "_terminal_stop_decision", None) or StopDecision(
            stop=False,
            reason="dispatcher completed without a terminal stopper decision",
        )
        completed_normally = True
        return decision
    finally:
        if not completed_normally:
            for t in active:
                t.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
        logger.info("[dispatcher] stop")


__all__ = ["dispatcher_loop"]
