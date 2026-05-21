"""Long-lived ingestion loop for the steady-state engine.

Polls in-flight programs in batch, ingests DONE ones
(accept→archive / reject→DISCARDED), sweeps vanished/DISCARDED ones, and
releases the semaphore slot each owned.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS
from gigaevo.programs.program_state import ProgramState

# Hook timeout / grace are configured on ``EngineConfig`` via
# ``post_step_hook_timeout_s`` and ``post_step_hook_cancel_grace_s``.


async def ingestor_loop(engine) -> None:
    logger.info("[ingestor] start")
    try:
        while engine._running:
            ingested = await poll_and_ingest(engine)
            saturated = len(engine._in_flight) >= engine._ss_config.max_in_flight
            interval = (
                engine.config.loop_interval * 0.25
                if (ingested or saturated)
                else engine.config.loop_interval
            )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise
    finally:
        logger.info("[ingestor] stop")


async def poll_and_ingest(engine) -> int:
    """One pass over ``engine._in_flight``: ingest DONE, sweep leaks, release slots.

    Returns the number of programs handled (ingested + swept).
    """
    async with engine._in_flight_lock:
        if not engine._in_flight:
            return 0
        candidates = list(engine._in_flight)

    programs = await engine.storage.mget(candidates, exclude=EXCLUDE_STAGE_RESULTS)
    found_ids = {p.id for p in programs if p is not None}

    done_ids: list[str] = []
    leaked_ids: list[str] = []
    for prog in programs:
        if prog is None:
            continue
        if prog.state == ProgramState.DONE:
            done_ids.append(prog.id)
        elif prog.state == ProgramState.DISCARDED:
            leaked_ids.append(prog.id)
    for pid in candidates:
        if pid not in found_ids:
            leaked_ids.append(pid)

    added = 0
    handled_ids: list[str] = []
    if done_ids:
        added, handled_ids = await _ingest_batch(engine, done_ids)

    released = set(handled_ids) | set(leaked_ids)
    if released:
        if leaked_ids:
            logger.warning(
                "[ingestor] Sweeping {} leaked in-flight programs", len(leaked_ids)
            )
        # Pop tickets under the lock so the slot/ticket pair is removed
        # atomically with the in_flight set — no torn state where another
        # observer sees the slot freed but the parent locks still held.
        # Defer the actual ticket.release() until OUTSIDE the lock: the
        # release path is pure local mutation on asyncio.Locks, but
        # keeping the lock-held critical section short minimises producer
        # contention on _in_flight_lock during heavy ingest sweeps.
        tickets_to_release = []
        async with engine._in_flight_lock:
            for pid in released:
                if pid in engine._in_flight:
                    engine._in_flight.discard(pid)
                    # Buffer slot transferred to us by the producer under
                    # _in_flight_lock; we release it here so the next
                    # producer blocked at _buffer_sema.acquire() wakes up
                    # and registers ITS already-completed result in
                    # _in_flight on the next event-loop tick. The producer
                    # sema is untouched — the producer task released it in
                    # its own finally the moment it exited.
                    engine._buffer_sema.release()
                    ticket = engine._inflight_tickets.pop(pid, None)
                    if ticket is not None:
                        tickets_to_release.append(ticket)
        for ticket in tickets_to_release:
            ticket.release()

    # Persist programs_processed so external sync hooks see progress immediately.
    if handled_ids or leaked_ids:
        # Without this bump, the population snapshot stays frozen at the seed
        # drain and every collector sees stale data.
        engine.storage.snapshot.bump(incremental=True)
        await engine._write_snapshot(
            programs_processed=engine.metrics.programs_processed
        )

    # Fire post_step_hook once per sweep that landed a program in the
    # archive. Gating on added>0 (rather than every poll tick) avoids
    # hot-spinning hooks like CompositionInjectionHook, which walks the
    # entire G archive on each call. Hook failures are fault-isolated:
    # they must never abort ingestion.
    if added > 0 and engine._post_step_hook is not None:
        await _run_bounded_post_step_hook(engine)

    return len(handled_ids) + len(leaked_ids)


async def _run_bounded_post_step_hook(engine) -> None:
    """Run ``engine._post_step_hook`` bounded by the configured timeout.

    The hook is fault-isolated (a raise inside the hook must NOT abort
    ingestion — the Redis ingest write has already committed) and
    time-bounded (a hook that hangs would otherwise wedge the ingestor:
    no further sweeps fire, no new mutants reach the archive).

    We drive the hook via an explicit ``create_task`` so its lifecycle
    is observable. ``asyncio.wait(..., timeout=...)`` enforces the
    wall-clock budget without depending on the hook honouring
    cancellation — a ``wait_for`` form would block past the timeout if
    the hook suppresses ``CancelledError``. On timeout we cancel +
    grace-wait + log; on outer cancel we cancel + await + re-raise.
    """
    timeout_s = engine.config.post_step_hook_timeout_s
    grace_s = engine.config.post_step_hook_cancel_grace_s
    hook_task = asyncio.create_task(engine._post_step_hook(), name="post-step-hook")
    try:
        _, pending = await asyncio.wait([hook_task], timeout=timeout_s)
    except asyncio.CancelledError:
        # Outer was cancelled. Cancel the hook and wait briefly for it
        # to settle before re-raising — without this, the hook detaches
        # and can race storage.close() during engine teardown. Use
        # ``asyncio.wait`` (not ``await hook_task``) so a hook that
        # ignores ``CancelledError`` cannot wedge teardown; we accept an
        # orphan over an indefinite shutdown stall.
        hook_task.cancel()
        await asyncio.wait([hook_task], timeout=grace_s)
        raise

    if pending:
        # Timeout fired. Cancel and give a brief grace period for the
        # hook to honor cancellation. If it doesn't, log and abandon
        # rather than wedge the ingestor.
        #
        # ``asyncio.wait`` (NOT ``asyncio.wait_for``) is load-bearing
        # here. ``wait_for`` still awaits the cancel to be honoured
        # before raising TimeoutError, so a hook that catches
        # CancelledError and keeps looping would extend our wait past
        # the grace budget. The plain ``wait`` form returns at the
        # deadline regardless of the inner task's state.
        hook_task.cancel()
        _, still_pending = await asyncio.wait([hook_task], timeout=grace_s)
        if still_pending:
            logger.warning(
                "[ingestor] post_step_hook ignored cancel within {}s — "
                "potential orphan coroutine; ingestor proceeding",
                grace_s,
            )
        logger.warning(
            "[ingestor] post_step_hook exceeded {}s budget — cancelled to "
            "keep ingestor responsive (check the hook implementation)",
            timeout_s,
        )
        return

    # Hook finished (success or raise). Surface raises as non-fatal
    # WARN; ingestion has already committed.
    exc = hook_task.exception()
    if exc is not None and not isinstance(exc, asyncio.CancelledError):
        logger.warning(
            "[ingestor] post_step_hook failed: {} "
            "(non-fatal, ingest already committed)",
            exc,
        )


async def _ingest_batch(engine, program_ids: list[str]) -> tuple[int, list[str]]:
    if not program_ids:
        return 0, []

    completed = await engine.storage.mget(program_ids, exclude=EXCLUDE_STAGE_RESULTS)
    completed = [p for p in completed if p.state == ProgramState.DONE]
    if not completed:
        return 0, []

    added = 0
    rej_valid = 0
    rej_strategy = 0
    reject_ids: list[str] = []

    for prog in completed:
        try:
            if not engine.config.program_acceptor.is_accepted(prog):
                logger.info(
                    "[ingestor] {} REJECTED by acceptor (metrics={})",
                    prog.short_id,
                    prog.metrics,
                )
                await engine._notify_hook(prog, MutationOutcome.REJECTED_ACCEPTOR)
                reject_ids.append(prog.id)
                rej_valid += 1
            elif await engine.strategy.add(prog):
                added += 1
                await engine._notify_hook(prog, MutationOutcome.ACCEPTED)
            else:
                await engine._notify_hook(prog, MutationOutcome.REJECTED_STRATEGY)
                reject_ids.append(prog.id)
                rej_strategy += 1
        except Exception as exc:
            logger.error("[ingestor] {} ingestion failed: {}", prog.short_id, exc)
            reject_ids.append(prog.id)

    if reject_ids:
        reject_set = set(reject_ids)
        for prog in completed:
            if prog.id in reject_set:
                prog.state = ProgramState.DISCARDED
        try:
            await engine.storage.batch_transition_by_ids(
                reject_ids,
                ProgramState.DONE.value,
                ProgramState.DISCARDED.value,
            )
        except Exception as exc:
            logger.error(
                "[ingestor] batch discard failed for {} programs: {}",
                len(reject_ids),
                exc,
            )

    engine.metrics.programs_processed += len(completed)
    engine.metrics.record_ingestion_metrics(added, rej_valid, rej_strategy)
    return added, [p.id for p in completed]


__all__ = ["ingestor_loop", "poll_and_ingest"]
