"""Per-mutant async task — the unit of producer work under the steady-state engine.

One task = one mutant. The dispatcher loop spawns these as soon as a
``_producer_sema`` slot opens; the task runs to completion independently
and is never awaited by the dispatcher.

Three ownership-handoff invariants govern every exit path:

1. **Producer-sema slot**: the dispatcher acquires it before spawning;
   the producer task ALWAYS releases it in ``finally``. No transfer
   semantics. This is what guarantees a freshly-freed DAG slot is
   refilled within one event-loop tick — the producer pool is decoupled
   from the per-mutant DAG lifetime.

2. **Buffer-sema slot**: acquired AFTER the LLM call returns and BEFORE
   ``_in_flight.add``. Every exit either (a) adds the new mutant id to
   ``engine._in_flight`` (transferring slot ownership; the ingestor will
   release the slot when the mutant reaches DONE/DISCARDED), or (b)
   releases the slot here. Never both, never neither.

3. **Parent-refresh ticket**: ``refresh_with_ticket`` returns a ticket
   holding the per-parent-id locks. The producer extends lock-hold past
   the refresh through the entire child-DAG by transferring the ticket
   to ``engine._inflight_tickets`` keyed by the new mutant id; the
   ingestor releases the ticket when the child is ingested or swept. If
   the producer fails before the child is registered, the ticket is
   released here. This enforces "no parent refresh while a child of that
   parent is in flight" — without it, a concurrent producer could pick
   the same parents and read this mutant's in-flight (state=RUNNING,
   metrics={}) entry from Redis during its own refresh DAG.

See ``docs/superpowers/specs/2026-05-13-mutation-throughput-two-sema-design.md``.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.evolution.engine.refresh import ParentRefreshTicket


async def run_one_mutant(engine, task_id: int) -> str | None:
    """Produce one mutant. Caller (dispatcher) holds one ``_producer_sema`` slot."""
    slot_transferred = False
    buffer_held = False
    ticket: ParentRefreshTicket | None = None
    new_id: str | None = None
    try:
        parents = await engine._select_parents_for_mutation()
        if not parents:
            # Empty archive — back off so dispatcher does not hot-spin while
            # the population is being seeded or while all programs are being
            # rejected by the acceptor.
            await asyncio.sleep(engine.config.loop_interval)
            return None

        try:
            ticket = await engine._parent_refresher.refresh_with_ticket(parents)
        except (ValueError, TimeoutError) as exc:
            logger.warning(
                "[mutant_task:{}] Parent refresh failed: {} — aborting mutant",
                task_id,
                exc,
            )
            return None
        refreshed = ticket.refreshed

        if refreshed:
            engine.metrics.submitted_for_refresh += len(refreshed)

        # Inline single-mutant primitive — no asyncio.gather to swallow the
        # persisted ID under outer-cancel. If we are cancelled after the
        # program is persisted, generate_one_mutation's except BaseException
        # arm returns the ID and we transfer the slot below before the
        # finally block re-raises.
        # Track LLM occupancy: increment before LLM call, decrement after.
        engine._llm_active += 1
        try:
            new_id = await generate_one_mutation(
                parents=refreshed,
                mutator=engine.mutation_operator,
                storage=engine.storage,
                state_manager=engine.state,
                iteration=engine.metrics.iteration,
                task_id=task_id,
            )
        finally:
            engine._llm_active -= 1

        if new_id is None:
            return None

        # Buffer backpressure: block here when the DAG cannot keep up. The
        # producer slot is still held during this wait — that is the design
        # invariant. The producer pool's job is to keep N LLM calls (or
        # ready-result-held producers) alive; the buffer pool gates
        # registration in _in_flight. See spec § Architecture.
        await engine._buffer_sema.acquire()
        buffer_held = True

        # Transfer both the buffer slot AND the parent-refresh ticket
        # atomically under _in_flight_lock so the ingestor can later pair
        # them by mutant id. Holding _in_flight_lock here is cheap — the
        # critical section is two dict/set ops with no awaits.
        async with engine._in_flight_lock:
            engine._in_flight.add(new_id)
            engine._inflight_tickets[new_id] = ticket
        slot_transferred = True
        # Ticket ownership has transferred to the ingestor; null it locally
        # so the `finally` block does not double-release the same locks.
        ticket = None
        engine.metrics.iteration += 1
        engine.metrics.mutations_created += 1
        # Persist counter so a resume after a crash continues from the
        # correct mutant count rather than 0. Without this,
        # MaxMutantsStopper would run the full budget again on resume.
        # (EngineSnapshot still spells the field ``total_mutants``; rename
        # follow-up tracked under #232.)
        await engine._write_snapshot(total_mutants=engine.metrics.iteration)
        return new_id

    finally:
        # producer_sema: ALWAYS released. No transfer semantics — the
        # dispatcher holds one slot per spawned task and the slot is
        # returned to the pool the moment the producer task exits, win,
        # lose, or cancel. This is what lets a freshly-freed DAG slot get
        # refilled within one event-loop tick from a buffer-held producer.
        engine._producer_sema.release()
        # buffer_sema: released only if we held it AND did not transfer
        # to the ingestor. `slot_transferred=True` means the ingestor
        # owns the release. The (buffer_held, slot_transferred) pair has
        # three reachable states:
        #   (False, False) → never acquired, nothing to release.
        #   (True,  False) → acquired but cancel before _in_flight.add;
        #                    we release here.
        #   (True,  True ) → acquired AND transferred; ingestor releases.
        if buffer_held and not slot_transferred:
            engine._buffer_sema.release()
        # Parent-lock invariant: if the ticket did not transfer to the
        # ingestor (failure path or pre-registration cancel), release it
        # here so the per-parent-id locks are freed for the next producer.
        # ``release()`` is idempotent.
        if ticket is not None:
            ticket.release()


__all__ = ["run_one_mutant"]
