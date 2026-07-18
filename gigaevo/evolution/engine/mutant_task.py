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
import inspect
from uuid import uuid4

from loguru import logger

from gigaevo.evolution.engine.mutation import MutationFailure, generate_one_mutation
from gigaevo.evolution.engine.refresh import ParentRefreshTicket
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.programs.program import Program


def _clear_uncommitted_memory_selection(parents: list[Program]) -> None:
    """Make a gate-skipped refresh an explicit memory-free mutation."""

    for parent in parents:
        parent.set_metadata(MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY, [])
        parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, [])
        parent.set_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY, False)
        parent.set_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY, "")
        parent.set_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY, None)


async def run_one_mutant(engine, task_id: int) -> str | None:
    """Produce one mutant. Caller (dispatcher) holds one ``_producer_sema`` slot."""
    slot_transferred = False
    buffer_held = False
    ticket: ParentRefreshTicket | None = None
    selection_lease = None
    attempt_id: str | None = None
    memory_decision_absent = False
    new_id: str | None = None
    topology_child_id: str | None = None
    mutation_failure: MutationFailure | None = None

    def capture_mutation_failure(failure: MutationFailure) -> None:
        nonlocal mutation_failure
        mutation_failure = failure

    def record_child_topology(
        child_id: str,
        parent_id: str,
        island_id: str,
    ) -> None:
        nonlocal topology_child_id
        topology_child_id = child_id
        engine._link_memory_attempt_child(
            attempt_id=attempt_id,
            child_id=child_id,
            parent_id=parent_id,
            island_id=island_id,
            completion_ordinal=my_iteration,
        )

    try:
        parents = await engine._select_parents_for_mutation()
        if not parents:
            # Empty archive — back off so dispatcher does not hot-spin while
            # the population is being seeded or while all programs are being
            # rejected by the acceptor.
            await asyncio.sleep(engine.config.loop_interval)
            return None

        if engine._selection_leases is not None:
            attempt_id = uuid4().hex
            for parent in parents:
                selection_lease = engine._selection_leases.open_attempt(
                    attempt_id, parent.id
                )

        if engine._ss_config.coalesce_refresh:
            try:
                refresh_scope = (
                    engine._selection_leases.activate_attempt(
                        attempt_id, (parent.id for parent in parents)
                    )
                    if engine._selection_leases is not None and attempt_id is not None
                    else None
                )
                refresh_method = engine._parent_refresher.refresh_if_stale
                refresh_kwargs = (
                    {"refresh_scope": refresh_scope}
                    if "refresh_scope" in inspect.signature(refresh_method).parameters
                    else {}
                )
                result = await refresh_method(parents, **refresh_kwargs)
            except (ValueError, TimeoutError) as exc:
                logger.warning(
                    "[mutant_task:{}] Parent refresh failed: {} — aborting mutant",
                    task_id,
                    exc,
                )
                closed = engine._record_memory_attempt_failure(
                    parents,
                    status="invalid",
                    failure_stage=(
                        "parent_refresh_timeout"
                        if isinstance(exc, TimeoutError)
                        else "parent_refresh"
                    ),
                    completion_ordinal=engine.metrics.iteration,
                    attempt_id=attempt_id,
                )
                memory_decision_absent = (
                    engine._memory_attempt_has_decision(attempt_id) is False
                )
                if closed == 0 and not memory_decision_absent:
                    raise
                return None
            refreshed = result.refreshed
            engine.metrics.submitted_for_refresh += result.stale_count
            if selection_lease is not None:
                for parent in refreshed:
                    selected = list(
                        parent.get_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY)
                        or []
                    )
                    verified = selection_lease.reverify_cards(selected)
                    if tuple(selected) != verified:
                        parent.set_metadata(
                            MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, list(verified)
                        )
                        logger.info(
                            "[Memory][Leases] dropped {} vanished coalesced-fresh "
                            "selection(s) for {}",
                            len(selected) - len(verified),
                            parent.id[:8],
                        )
        else:
            try:
                refresh_scope = (
                    engine._selection_leases.activate_attempt(
                        attempt_id, (parent.id for parent in parents)
                    )
                    if engine._selection_leases is not None and attempt_id is not None
                    else None
                )
                refresh_method = engine._parent_refresher.refresh_with_ticket
                refresh_kwargs = (
                    {"refresh_scope": refresh_scope}
                    if "refresh_scope" in inspect.signature(refresh_method).parameters
                    else {}
                )
                ticket = await refresh_method(parents, **refresh_kwargs)
            except (ValueError, TimeoutError) as exc:
                logger.warning(
                    "[mutant_task:{}] Parent refresh failed: {} — aborting mutant",
                    task_id,
                    exc,
                )
                closed = engine._record_memory_attempt_failure(
                    parents,
                    status="invalid",
                    failure_stage=(
                        "parent_refresh_timeout"
                        if isinstance(exc, TimeoutError)
                        else "parent_refresh"
                    ),
                    completion_ordinal=engine.metrics.iteration,
                    attempt_id=attempt_id,
                )
                memory_decision_absent = (
                    engine._memory_attempt_has_decision(attempt_id) is False
                )
                if closed == 0 and not memory_decision_absent:
                    raise
                return None
            refreshed = ticket.refreshed
            if refreshed:
                engine.metrics.submitted_for_refresh += len(refreshed)

        memory_decision_absent = (
            engine._memory_attempt_has_decision(attempt_id) is False
        )
        if memory_decision_absent:
            _clear_uncommitted_memory_selection(refreshed)
            logger.info(
                "[MemoryV2] attempt {} has no committed decision; "
                "mutating without memory",
                attempt_id,
            )

        # Atomic ordinal reserve — the read+increment pair has no `await` in
        # between, so under asyncio one coroutine cannot interleave another.
        # Without this, N concurrent producers all read the same pre-await
        # value and every child program shares one iteration ordinal,
        # producing the "long vertical lines at iteration 0" plot symptom.
        my_iteration = engine.metrics.iteration
        engine.metrics.iteration += 1

        engine._llm_active += 1
        try:
            try:
                new_id = await generate_one_mutation(
                    parents=refreshed,
                    mutator=engine.mutation_operator,
                    storage=engine.storage,
                    state_manager=engine.state,
                    iteration=my_iteration,
                    task_id=task_id,
                    selection_lease=selection_lease,
                    failure_observer=capture_mutation_failure,
                    child_observer=record_child_topology,
                )
            except asyncio.CancelledError:
                failure = mutation_failure or MutationFailure(
                    status="censored", stage="mutation_cancelled"
                )
                if topology_child_id is not None:
                    engine._record_memory_topology_failure(
                        topology_child_id,
                        status=failure.status,
                        failure_stage=failure.stage,
                    )
                closed = engine._record_memory_attempt_failure(
                    refreshed,
                    status=failure.status,
                    failure_stage=failure.stage,
                    completion_ordinal=my_iteration,
                    attempt_id=attempt_id,
                )
                if closed == 0 and not memory_decision_absent:
                    raise RuntimeError(
                        f"memory attempt {attempt_id!r} has no durable decision"
                    )
                raise
        finally:
            engine._llm_active -= 1

        if new_id is None:
            failure = mutation_failure or MutationFailure(
                status="censored", stage="mutation_unknown"
            )
            if topology_child_id is not None:
                engine._record_memory_topology_failure(
                    topology_child_id,
                    status=failure.status,
                    failure_stage=failure.stage,
                )
            closed = engine._record_memory_attempt_failure(
                refreshed,
                status=failure.status,
                failure_stage=failure.stage,
                completion_ordinal=my_iteration,
                attempt_id=attempt_id,
            )
            if closed == 0 and not memory_decision_absent:
                raise RuntimeError(
                    f"memory attempt {attempt_id!r} has no durable decision"
                )
            return None

        async def handoff_persisted_child() -> None:
            """Finish the durable child-to-ingestor handoff despite cancellation."""

            nonlocal buffer_held, slot_transferred, ticket
            await engine._buffer_sema.acquire()
            buffer_held = True
            try:
                # The slot and parent ticket move together under one lock.
                async with engine._in_flight_lock:
                    engine._in_flight.add(new_id)
                    if ticket is not None:
                        engine._inflight_tickets[new_id] = ticket
                slot_transferred = True
                ticket = None
                engine.metrics.mutations_created += 1
                await engine._write_snapshot(
                    total_mutants=engine.metrics.mutations_created,
                    next_iteration=engine.metrics.iteration,
                )
            except BaseException:
                if not slot_transferred:
                    engine._buffer_sema.release()
                    buffer_held = False
                raise

        # After persistence, cancellation is deferred until the child is visible
        # to the ingestor. The dispatcher waits its producer tasks while the
        # ingestor is still alive, so buffer capacity can continue to drain.
        handoff_task = asyncio.create_task(
            handoff_persisted_child(), name=f"mutant-handoff-{task_id}"
        )
        cancellation: asyncio.CancelledError | None = None
        while not handoff_task.done():
            try:
                await asyncio.shield(handoff_task)
            except asyncio.CancelledError as exc:
                cancellation = exc
        await handoff_task
        if cancellation is not None:
            raise cancellation
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
        if selection_lease is not None:
            selection_lease.release()


__all__ = ["run_one_mutant"]
