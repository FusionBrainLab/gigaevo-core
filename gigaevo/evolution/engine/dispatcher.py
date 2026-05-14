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
    task_id = 0
    try:
        while engine._running and not engine._reached_mutant_cap():
            await engine._producer_sema.acquire()
            if not engine._running or engine._reached_mutant_cap():
                # Post-acquire early-stop: hand the slot back so a graceful
                # restart finds _producer_sema at full capacity.
                engine._producer_sema.release()
                break
            t = asyncio.create_task(
                run_one_mutant(engine, task_id), name=f"mutant-{task_id}"
            )
            task_id += 1
            active.add(t)
            t.add_done_callback(active.discard)
    finally:
        for t in active:
            t.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        logger.info("[dispatcher] stop")


__all__ = ["dispatcher_loop"]
