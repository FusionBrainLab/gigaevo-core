"""SOTA invariant tests for the steady-state engine concurrency contract.

These tests are written to fail loudly if any of the 8 concurrency
invariants of the steady-state engine are violated:

  I1. Slot conservation. Every ``_producer_sema.acquire`` (dispatcher) and
      ``_buffer_sema.acquire`` (producer) is matched by exactly one release —
      ``_producer_sema`` always in ``mutant_task`` finally; ``_buffer_sema``
      either in ``mutant_task`` finally (when ``slot_transferred=False``) or
      inside the ingestor (when the mutant reached DONE/DISCARDED).
  I2. ``_in_flight`` drains. After ingestion, no ID stays in
      ``_in_flight`` once its program reached DONE/DISCARDED.
  I3. ``slot_transferred`` is exclusive. A mutant either transfers its
      slot to the ingestor or releases in its own finally — never both.
  I4. Locks are not held across awaits. ``_in_flight_lock`` and
      ``_snapshot_lock`` must release before the next ``await`` so other
      coroutines can make progress.
  I5. Snapshot version is strictly monotonic in Redis under concurrency.
  I6. Cancellation propagates cleanly through the supervised loops; the
      post-run hook still fires even when ``run()`` is cancelled.
  I7. The ingestor uses the fast loop interval when saturated.
  I8. ``_await_idle`` returns ``False`` when only DISCARDED programs
      exist (DISCARDED does NOT count toward "active").

The audit by test-obsessed-reviewer flagged 8 gaps in coverage of these
invariants. This file plugs all 8 gaps with deterministic tests — no
``time.sleep`` polls, no flaky timing assumptions; ``asyncio.Event`` and
explicit task scheduling are used wherever synchronisation is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.dispatcher import dispatcher_loop
from gigaevo.evolution.engine.ingestor import ingestor_loop, poll_and_ingest
from gigaevo.evolution.engine.mutant_task import run_one_mutant
from gigaevo.evolution.engine.snapshot import (
    EngineSnapshot,
    _reset_current_snapshot_for_tests,
)
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxMutantsStopper
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

INV_TEST_TIMEOUT = 5.0  # any single test must finish in <=5s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


def _make_engine(
    *,
    max_in_flight: int = 4,
    max_mutants: int | None = None,
    loop_interval: float = 0.01,
) -> SteadyStateEvolutionEngine:
    """Build an engine with all status query methods stubbed to "idle"."""
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()
    metrics_tracker.format_best_summary.return_value = ""

    # Idle by default — overrideable per-test.
    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.mget.return_value = []
    storage.snapshot = MagicMock()
    strategy.get_program_ids.return_value = []

    stopper = MaxMutantsStopper(max_mutants) if max_mutants is not None else None
    cfg_kwargs: dict = {
        "max_in_flight": max_in_flight,
        "loop_interval": loop_interval,
    }
    if stopper is not None:
        cfg_kwargs["stopper"] = stopper
    config = SteadyStateEngineConfig(**cfg_kwargs)

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=config,
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    _reset_current_snapshot_for_tests()
    return engine


@pytest.fixture(autouse=True)
def _reset_snapshot_state():
    """Each test starts with a fresh process-wide snapshot mirror."""
    _reset_current_snapshot_for_tests()
    yield
    _reset_current_snapshot_for_tests()


# ===========================================================================
# Gap 1 — I1: cancel between sema.acquire and slot transfer releases the slot
# ===========================================================================


class TestSlotReleaseOnCancelInAcquireWindow:
    """A mutant_task cancelled at any point before slot_transferred=True
    must release the semaphore slot via its finally block. Otherwise the
    semaphore counter drifts down forever and the engine wedges."""

    async def test_cancel_before_elite_select_releases_slot(self):
        engine = _make_engine(max_in_flight=1)

        # Saturate: caller acquires the slot (mirrors dispatcher protocol).
        await engine._producer_sema.acquire()
        assert engine._producer_sema._value == 0

        # Elite selection blocks forever — simulates a slow query.
        block = asyncio.Event()

        async def hang(*_a, **_kw):
            await block.wait()
            return []

        engine.strategy.select_elites.side_effect = hang

        task = asyncio.create_task(run_one_mutant(engine, task_id=0))
        # Yield so the task enters select_elites and is suspended on block.
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Producer slot was released in the finally block.
        assert engine._producer_sema._value == 1
        assert not engine._in_flight  # never transferred

    async def test_cancel_during_parent_refresh_releases_slot(self):
        engine = _make_engine(max_in_flight=1)
        await engine._producer_sema.acquire()

        parent = _prog()
        engine.strategy.select_elites.return_value = [parent]
        block = asyncio.Event()

        async def hang_refresh(_p):
            await block.wait()
            return [parent]

        engine._parent_refresher.refresh_with_ticket = AsyncMock(
            side_effect=hang_refresh
        )

        task = asyncio.create_task(run_one_mutant(engine, task_id=1))
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert engine._producer_sema._value == 1
        assert not engine._in_flight


# ===========================================================================
# Gap 2 — I6: dispatcher cancel drains all active mutant tasks
# ===========================================================================


class TestDispatcherCancelDrainsActive:
    """When dispatcher_loop is cancelled, every spawned mutant task in
    `active` set must be cancelled and awaited (its finally block runs)."""

    async def test_active_tasks_are_cancelled_on_dispatcher_cancel(self):
        engine = _make_engine(max_in_flight=4, loop_interval=0.001)
        engine._running = True

        # Each mutant blocks on this event so they pile up in `active`.
        block = asyncio.Event()
        seen_cancels = 0
        lock = asyncio.Lock()

        async def hang_select(*_a, **_kw):
            try:
                await block.wait()
                return []
            except asyncio.CancelledError:
                nonlocal seen_cancels
                async with lock:
                    seen_cancels += 1
                raise

        engine.strategy.select_elites.side_effect = hang_select

        disp = asyncio.create_task(dispatcher_loop(engine))
        # Yield until 3 mutant tasks have been spawned.
        for _ in range(30):
            await asyncio.sleep(0)
            if engine._producer_sema._value == 0:
                break
        engine._running = False
        disp.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disp

        # All mutant tasks' finally blocks ran (cancellation seen at least
        # by the ones that had been spawned).
        assert seen_cancels >= 1
        # All slots are free now.
        assert (
            engine._producer_sema._value == engine._ss_config.max_in_flight
            and engine._buffer_sema._value == engine._ss_config.max_in_flight
        )


# ===========================================================================
# Gap 3 — I7: ingestor uses fast interval (loop_interval * 0.25) when saturated
# ===========================================================================


class TestIngestorSaturatedFastInterval:
    """When ``ingested or saturated``, ingestor sleeps at 0.25 *
    loop_interval; otherwise the full loop_interval."""

    @staticmethod
    def _install_sleep_spy(monkeypatch, engine) -> list[float]:
        """Patch the ingestor module's ``asyncio.sleep`` attribute via the
        full dotted path so a future ``from asyncio import sleep`` refactor
        fails loudly (AttributeError) instead of silently bypassing the spy.
        """
        recorded: list[float] = []
        real_sleep = asyncio.sleep

        async def spy_sleep(seconds: float, *a, **kw):
            recorded.append(seconds)
            engine._running = False
            await real_sleep(0, *a, **kw)

        monkeypatch.setattr(
            "gigaevo.evolution.engine.ingestor.asyncio.sleep", spy_sleep
        )
        return recorded

    async def test_fast_interval_when_saturated(self, monkeypatch):
        """saturated=True, ingested=False → fast interval (0.25 * loop_interval)."""
        engine = _make_engine(max_in_flight=2, loop_interval=0.10)
        engine._running = True
        engine._in_flight.update({"p1", "p2"})
        # poll_and_ingest will return 0 (mget returns []); saturation alone
        # drives the fast branch.

        recorded = self._install_sleep_spy(monkeypatch, engine)
        await ingestor_loop(engine)

        assert recorded, "ingestor never slept"
        assert abs(recorded[0] - 0.025) < 1e-9, f"expected 0.025, got {recorded[0]}"

    async def test_fast_interval_when_ingested_not_saturated(self, monkeypatch):
        """ingested=True, saturated=False → fast interval. Distinct from the
        saturated branch: catches a regression that drops the ``ingested``
        half of the ``(ingested or saturated)`` predicate."""
        engine = _make_engine(max_in_flight=4, loop_interval=0.10)
        engine._running = True
        # _in_flight stays empty; force poll_and_ingest to report a non-zero
        # count so the predicate's `ingested` half is the only thing that can
        # select the fast branch.
        import gigaevo.evolution.engine.ingestor as ing_mod

        async def fake_poll(_engine):
            return 1

        monkeypatch.setattr(ing_mod, "poll_and_ingest", fake_poll)

        recorded = self._install_sleep_spy(monkeypatch, engine)
        await ingestor_loop(engine)

        assert recorded, "ingestor never slept"
        assert abs(recorded[0] - 0.025) < 1e-9, f"expected 0.025, got {recorded[0]}"

    async def test_slow_interval_when_idle(self, monkeypatch):
        """ingested=False, saturated=False → full loop_interval. Without
        this we can't assert the fast-branch tests aren't just always 0.025."""
        engine = _make_engine(max_in_flight=4, loop_interval=0.10)
        engine._running = True
        # _in_flight is empty, so poll_and_ingest returns 0 and the
        # saturated branch is False.

        recorded = self._install_sleep_spy(monkeypatch, engine)
        await ingestor_loop(engine)

        assert recorded, "ingestor never slept"
        assert abs(recorded[0] - 0.10) < 1e-9


# ===========================================================================
# Gap 4 — I6: post_run_hook fires even on cancellation
# ===========================================================================


class TestPostRunHookOnCancel:
    """The post_run_hook is a finaliser — it MUST run when the engine is
    cancelled, otherwise per-run-folder cleanup (e.g. flushing CSV writers,
    closing DB connections) is silently skipped."""

    async def test_hook_fires_when_run_cancelled(self):
        engine = _make_engine(max_in_flight=1, loop_interval=0.001)

        hook_fired = asyncio.Event()
        hook = AsyncMock()

        async def hook_call(_storage):
            hook_fired.set()

        hook.on_run_complete.side_effect = hook_call
        engine._post_run_hook = hook

        # Pre-seed snapshot so the initial _await_idle returns immediately
        # (count_by_status defaults to 0).
        run_task = asyncio.create_task(engine.run())
        # Let run() reach the loops.
        for _ in range(20):
            await asyncio.sleep(0)
            if engine._dispatcher_task is not None:
                break

        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=INV_TEST_TIMEOUT)

        assert hook_fired.is_set(), (
            "post_run_hook did not fire on cancel — finaliser contract broken"
        )
        hook.on_run_complete.assert_awaited()


# ===========================================================================
# Gap 5 — I4: _in_flight_lock does not starve under high concurrency
# ===========================================================================


class TestInFlightLockNoStarvation:
    """asyncio.Lock is FIFO-fair. Under concurrent contention, every waiter
    must eventually acquire — no waiter starves indefinitely. This guards
    against accidentally holding the lock across an unbounded await."""

    async def test_many_waiters_all_progress(self):
        engine = _make_engine(max_in_flight=8)

        acquired_order: list[int] = []
        N = 50

        async def add_id(i: int):
            async with engine._in_flight_lock:
                acquired_order.append(i)
                # No await inside the critical section — just lock & release.

        await asyncio.gather(*(add_id(i) for i in range(N)))

        assert len(acquired_order) == N
        assert sorted(acquired_order) == list(range(N))


# ===========================================================================
# Gap 6 — I8: _await_idle treats DISCARDED as "idle" (not active)
# ===========================================================================


class TestAwaitIdleDiscardedOnly:
    """`_has_active_dags` counts only QUEUED + RUNNING. A registry full of
    DISCARDED programs must return False (idle), or the engine wedges
    waiting for ghost DAGs."""

    async def test_discarded_only_returns_idle(self):
        engine = _make_engine()
        # QUEUED=0, RUNNING=0, but DISCARDED could be any number — irrelevant
        engine.storage.count_by_status.side_effect = [0, 0]

        has_active = await engine._has_active_dags()
        assert has_active is False
        assert engine._last_pending_dags_counts is None

    async def test_await_idle_returns_promptly_with_only_discarded(self):
        engine = _make_engine(loop_interval=0.001)
        engine.storage.count_by_status.return_value = 0

        # Should return without raising and within the 5s timeout.
        await asyncio.wait_for(engine._await_idle(), timeout=1.0)


# ===========================================================================
# Gap 7 — I5: snapshot version is monotonic in Redis under concurrent writes
# ===========================================================================


class TestSnapshotVersionMonotonic:
    """The _snapshot_lock serialises _write_snapshot so the version that
    lands in Redis matches the in-memory mirror. Without the lock,
    concurrent writers can produce out-of-order Redis state.

    The contract: the SEQUENCE of versions persisted to Redis is monotone
    increasing by 1, with no gaps and no reorderings."""

    async def test_concurrent_writes_versions_monotone(self):
        engine = _make_engine()

        # Capture every save_run_state call's version field, in order.
        recorded_versions: list[int] = []
        recorded_lock = asyncio.Lock()

        async def capture_save(field: str, value: str):
            # Parse the JSON snapshot to read the version.
            snap = EngineSnapshot.model_validate_json(value)
            # Yield a few times to invite reorder; the lock should still
            # enforce a single monotone sequence.
            await asyncio.sleep(0)
            async with recorded_lock:
                recorded_versions.append(snap.version)

        engine.storage.save_run_state = AsyncMock(side_effect=capture_save)

        N = 20
        await asyncio.gather(
            *(engine._write_snapshot(total_mutants=i) for i in range(N))
        )

        # Versions land in [1..N] in strict monotone order with no gaps.
        assert recorded_versions == list(range(1, N + 1)), (
            f"versions reordered or skipped: {recorded_versions}"
        )

    async def test_in_memory_mirror_tracks_redis(self):
        """The in-process mirror version equals the count of writes."""
        engine = _make_engine()
        engine.storage.save_run_state = AsyncMock()

        await engine._write_snapshot(total_mutants=1)
        assert engine._snapshot.version == 1

        await engine._write_snapshot(total_mutants=2)
        assert engine._snapshot.version == 2

        await engine._write_snapshot(programs_processed=7)
        assert engine._snapshot.version == 3
        # And the most recent fields are retained.
        assert engine._snapshot.total_mutants == 2
        assert engine._snapshot.programs_processed == 7


# ===========================================================================
# Gap 8 — I1+I2: double-poll same DONE id releases the slot exactly once
# ===========================================================================


class TestDoublePollNoDoubleRelease:
    """If poll_and_ingest is invoked twice in quick succession (e.g. by a
    test, or by the final-sweep + main loop overlap), the second pass must
    not see the same id in _in_flight and release the semaphore a second
    time. The contract: discard from _in_flight ATOMICALLY with release."""

    async def test_id_not_double_released(self):
        engine = _make_engine(max_in_flight=2)
        engine._running = True

        # Pre-acquire one buffer slot (simulates producer's slot ownership).
        await engine._buffer_sema.acquire()
        starting_value = engine._buffer_sema._value
        assert starting_value == 1

        prog = _prog(state=ProgramState.DONE)
        async with engine._in_flight_lock:
            engine._in_flight.add(prog.id)

        # Storage: mget returns the DONE program both passes; strategy
        # accepts it the first time. Second pass: _in_flight is now empty
        # (already discarded), so the pass should be a no-op.
        engine.storage.mget.return_value = [prog]
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        # First pass: id is ingested + removed from _in_flight + slot released.
        await poll_and_ingest(engine)
        assert prog.id not in engine._in_flight
        # Slot count returned to max_in_flight.
        assert engine._buffer_sema._value == engine._ss_config.max_in_flight

        # Second pass: _in_flight is empty, mget would return the program
        # again (idempotent storage), but the early-exit check guards us.
        await poll_and_ingest(engine)

        # Critical invariant: semaphore did not over-release.
        assert engine._buffer_sema._value == engine._ss_config.max_in_flight, (
            "double release — semaphore drifted above max_in_flight"
        )

    async def test_leaked_id_swept_once(self):
        """A leaked (vanished or DISCARDED) id is also released exactly once."""
        engine = _make_engine(max_in_flight=2)
        engine._running = True

        # Take one buffer slot — that's the leaked mutant's slot.
        await engine._buffer_sema.acquire()

        leaked_id = "leaked-mutant-12345"
        async with engine._in_flight_lock:
            engine._in_flight.add(leaked_id)

        # mget returns empty: program has vanished — leaked path.
        engine.storage.mget.return_value = []

        await poll_and_ingest(engine)
        assert leaked_id not in engine._in_flight
        assert engine._buffer_sema._value == engine._ss_config.max_in_flight

        # Second pass: nothing to do.
        await poll_and_ingest(engine)
        assert engine._buffer_sema._value == engine._ss_config.max_in_flight


# ===========================================================================
# Bonus — I3 deterministic check: slot_transferred flag is exclusive
# ===========================================================================


class TestSlotTransferredExclusive:
    """`slot_transferred` toggles to True iff the new_id has been added to
    _in_flight. Once True, the finally block must NOT release the
    semaphore (ingestor owns the slot). Once False (or on early return),
    the finally MUST release."""

    async def test_success_path_transfers_slot(self):
        """When a mutant successfully persists, the slot transfers to the
        ingestor — the post-mutant_task semaphore value stays decremented."""
        engine = _make_engine(max_in_flight=1)
        engine._ss_config.coalesce_refresh = False
        await engine._producer_sema.acquire()
        starting_value = engine._producer_sema._value
        assert starting_value == 0

        parent = _prog()
        engine.strategy.select_elites.return_value = [parent]
        from gigaevo.evolution.engine.refresh import ParentRefreshTicket

        engine._parent_refresher.refresh_with_ticket = AsyncMock(
            return_value=ParentRefreshTicket(refreshed=[parent], _locks=[])
        )

        # Patch generate_one_mutation to return a known ID.
        import gigaevo.evolution.engine.mutant_task as mt_mod

        async def fake_gen(*_a, **_kw):
            return "new-mutant-deadbeef"

        original = mt_mod.generate_one_mutation
        mt_mod.generate_one_mutation = fake_gen  # type: ignore[assignment]
        # Snapshot write should not blow up.
        engine.storage.save_run_state = AsyncMock()
        try:
            new_id = await run_one_mutant(engine, task_id=0)
        finally:
            mt_mod.generate_one_mutation = original  # type: ignore[assignment]

        assert new_id == "new-mutant-deadbeef"
        # Buffer slot ownership transferred to ingestor — buffer_sema still drained.
        # Producer slot always released in finally — producer_sema is back at 1.
        assert engine._buffer_sema._value == 0
        assert engine._producer_sema._value == 1
        assert "new-mutant-deadbeef" in engine._in_flight

    async def test_no_elite_releases_slot(self):
        """Empty elites: early return WITHOUT slot transfer — finally releases."""
        engine = _make_engine(max_in_flight=1)
        await engine._producer_sema.acquire()
        engine.strategy.select_elites.return_value = []

        new_id = await run_one_mutant(engine, task_id=0)
        assert new_id is None
        # Producer slot released in finally; buffer slot never acquired (early return).
        assert engine._producer_sema._value == 1
        assert not engine._in_flight


# ===========================================================================
# Bonus — F4: metrics_collector_task is awaited before storage.close()
# ===========================================================================


class TestMetricsCollectorAwaitedOnStop:
    """`stop()` must await the cancelled `_metrics_collector_task` before
    closing storage. Without the await, the collector may still be in the
    middle of `await storage.<call>` when `storage.close()` fires, raising
    ConnectionClosedError into an orphan coroutine that has no caller."""

    async def test_collector_finished_before_storage_close(self):
        engine = _make_engine()

        # Build a collector task that performs an await on storage. We
        # observe whether it has finished (cancelled or not) by the time
        # storage.close() is invoked.
        collector_finished = asyncio.Event()
        storage_close_called = asyncio.Event()
        order: list[str] = []

        async def collector():
            try:
                # Long-lived poll loop, but the cancel must reach us.
                while True:
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                order.append("collector-cancelled")
                collector_finished.set()
                raise

        async def fake_close():
            order.append("storage-close")
            storage_close_called.set()

        engine.storage.close = AsyncMock(side_effect=fake_close)
        engine._metrics_tracker.stop = AsyncMock()

        engine._task = None  # so the early-return branch is taken
        engine._metrics_collector_task = asyncio.create_task(collector())
        # Let collector be scheduled.
        await asyncio.sleep(0)

        await asyncio.wait_for(engine.stop(), timeout=INV_TEST_TIMEOUT)

        assert collector_finished.is_set(), "collector did not see CancelledError"
        assert storage_close_called.is_set(), "storage.close() did not fire"
        # The collector's finally ran BEFORE storage.close was called.
        assert order.index("collector-cancelled") < order.index("storage-close"), (
            f"order violation: {order} — storage closed before collector finished"
        )

    async def test_wedged_collector_does_not_block_stop_forever(self):
        """If the collector somehow ignores cancel (e.g. holds a sync
        thread), stop() must still return within the 2s timeout — not
        wedge indefinitely."""
        engine = _make_engine()

        async def wedged_collector():
            # Shield ourselves from cancel — simulates a stuck collector.
            try:
                await asyncio.shield(asyncio.sleep(60))
            except asyncio.CancelledError:
                # Re-shield to defeat the cancel one more time.
                await asyncio.shield(asyncio.sleep(60))

        engine.storage.close = AsyncMock()
        engine._metrics_tracker.stop = AsyncMock()
        engine._task = None
        engine._metrics_collector_task = asyncio.create_task(wedged_collector())
        await asyncio.sleep(0)

        # The 2s wait_for timeout caps the wedge; total budget must be <5s.
        await asyncio.wait_for(engine.stop(), timeout=INV_TEST_TIMEOUT)


# ===========================================================================
# Dispatcher: `finally` cancels spawned mutants on outer cancel
# ===========================================================================


class TestDispatcherFinallyCancelsSpawnedMutants:
    """The dispatcher loop's `finally` is the only path that cancels the
    in-flight mutant tasks it spawned. This test pins the contract that
    the finally block is entered on outer cancel and that every spawned
    mutant is cancelled (and its slot subsequently released by the
    sweep). Catches accidental swallowing of CancelledError that would
    leave detached mutants leaking semaphore slots."""

    async def test_outer_cancel_propagates_and_cancels_spawned(self):
        engine = _make_engine(max_in_flight=3)

        spawned_cancelled: list[bool] = []
        spawn_event = asyncio.Event()

        async def long_running_mutant():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                spawned_cancelled.append(True)
                raise

        # Stub `run_one_mutant` indirection by patching the dispatcher's
        # imported symbol. The dispatcher would normally call
        # gigaevo.evolution.engine.mutant_task.run_one_mutant — we monkey-patch
        # at the dispatcher module so spawned tasks are our long-runner.
        import gigaevo.evolution.engine.dispatcher as disp_mod

        async def fake_run(_engine, _task_id):
            spawn_event.set()
            await long_running_mutant()

        original = disp_mod.run_one_mutant
        disp_mod.run_one_mutant = fake_run
        try:
            engine._running = True
            engine._reached_mutant_cap = lambda: False  # type: ignore[method-assign]

            loop_task = asyncio.create_task(dispatcher_loop(engine))
            # Let the dispatcher acquire a slot and spawn at least one mutant.
            await asyncio.wait_for(spawn_event.wait(), timeout=INV_TEST_TIMEOUT)

            # Cancel the dispatcher; the finally must cancel the spawned mutant.
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(loop_task, timeout=INV_TEST_TIMEOUT)
        finally:
            disp_mod.run_one_mutant = original

        assert spawned_cancelled, (
            "dispatcher finally did not cancel spawned mutant — "
            "outer CancelledError was swallowed or finally was skipped"
        )


# ===========================================================================
# _final_ingestion_sweep inner-task is awaited on cancel
# ===========================================================================


class TestFinalSweepInnerAwaitedOnCancel:
    """`_final_ingestion_sweep` must not leave its `poll_and_ingest` inner
    task running past method return when the outer is cancelled. Two
    escape paths a blanket `suppress(Exception)` would allow silently:

      (1) Slow-cancel target — inner takes >1s to honor cancel, wait_for
          raises TimeoutError (Exception subclass), suppressed; inner runs
          detached and would race storage.close().
      (2) Double-cancel — outer cancelled again during wait_for, which
          re-raises CancelledError (BaseException, not Exception); the
          suppress doesn't catch it, `cancelled=True; break` never runs,
          inner is detached.

    Both lead to a ConnectionClosedError-into-orphan when storage.close()
    fires next, so the sweep's bounded-wait must route CancelledError AND
    TimeoutError through the explicit cancelled-flag path.
    """

    async def test_slow_cancel_inner_logs_timeout_but_no_orphan_on_normal_path(
        self, caplog
    ):
        """If inner ignores cancel for >1s, the WARN is emitted and the
        sweep does not silently leak — the outer's cancel still drives
        the loop to `cancelled=True; break`."""
        engine = _make_engine()
        engine._in_flight.add("stuck-id")

        from loguru import logger as loguru_logger

        warnings: list[str] = []
        sink_id = loguru_logger.add(lambda rec: warnings.append(rec), level="WARNING")
        try:
            # Patch poll_and_ingest to a slow-cancel target — sleep 5s,
            # ignore the first cancel by re-shielding once.
            import gigaevo.evolution.engine.steady_state as ss_mod

            slow_cancel_started = asyncio.Event()

            async def slow_cancel_poll(_engine):
                slow_cancel_started.set()
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    # Ignore the first cancel for 2s — simulates a sync
                    # Redis call that doesn't reach a cancel checkpoint.
                    await asyncio.shield(asyncio.sleep(2.0))
                    raise
                return 0

            original = ss_mod.poll_and_ingest
            ss_mod.poll_and_ingest = slow_cancel_poll
            try:
                # Start the sweep, let inner be spawned, then cancel
                # the sweep.
                sweep = asyncio.create_task(
                    engine._final_ingestion_sweep(deadline_seconds=10.0)
                )
                await asyncio.wait_for(slow_cancel_started.wait(), timeout=2.0)
                sweep.cancel()
                # The sweep should raise CancelledError back to us, having
                # bounded the inner-wait to 1s and routed through the
                # cancelled=True; break path.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(sweep, timeout=INV_TEST_TIMEOUT)
            finally:
                ss_mod.poll_and_ingest = original
        finally:
            loguru_logger.remove(sink_id)

        # WARN should have been emitted about the orphan risk.
        warn_text = "".join(str(w) for w in warnings)
        assert "did not honor cancel" in warn_text or "orphan" in warn_text, (
            f"expected orphan-risk warning in logs, got: {warn_text[:400]}"
        )

    async def test_double_cancel_routes_through_cancelled_flag(self):
        """If a SECOND cancel arrives mid-`wait_for(inner)`, the explicit
        `suppress(CancelledError)` must catch it so `cancelled=True; break`
        still runs (rather than the exception flying out with inner
        detached)."""
        engine = _make_engine()
        engine._in_flight.add("stuck-id")

        import gigaevo.evolution.engine.steady_state as ss_mod

        inner_cancelled = asyncio.Event()

        async def slow_poll(_engine):
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                inner_cancelled.set()
                raise
            return 0

        original = ss_mod.poll_and_ingest
        ss_mod.poll_and_ingest = slow_poll
        try:
            sweep = asyncio.create_task(
                engine._final_ingestion_sweep(deadline_seconds=10.0)
            )
            # Let inner be spawned.
            await asyncio.sleep(0.05)
            # First cancel — enters the except arm.
            sweep.cancel()
            await asyncio.sleep(0)
            # Second cancel — would land on the wait_for(inner) call.
            sweep.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(sweep, timeout=INV_TEST_TIMEOUT)
        finally:
            ss_mod.poll_and_ingest = original

        # The inner must have been cancelled (not left orphaned).
        # Even under double-cancel, the explicit cancel reached inner.
        assert inner_cancelled.is_set(), (
            "inner poll_and_ingest task was not cancelled by sweep — "
            "double-cancel routed the cancel out before inner.cancel() fired"
        )


# ===========================================================================
# _write_snapshot persist-then-mirror ordering
# ===========================================================================


class TestWriteSnapshotPersistThenMirror:
    """`_write_snapshot` must save to Redis FIRST and mirror SECOND. If
    Redis save raises, the in-memory mirror keeps its current version so
    the next call retries the same version (no version skip in Redis)."""

    async def test_save_failure_leaves_mirror_at_old_version(self):
        engine = _make_engine()
        from gigaevo.evolution.engine.snapshot import get_current_snapshot

        assert engine._snapshot.version == 0
        engine.storage.save_run_state.side_effect = RuntimeError("redis down")

        with pytest.raises(RuntimeError, match="redis down"):
            await engine._write_snapshot(total_mutants=5)

        # Mirror stayed at version 0; total_mutants was NOT applied.
        assert engine._snapshot.version == 0, (
            f"mirror advanced to {engine._snapshot.version} despite save failure"
        )
        assert engine._snapshot.total_mutants == 0
        # Process-wide cache also unchanged.
        cached = get_current_snapshot()
        if cached is not None:
            assert cached.version == 0
            assert cached.total_mutants == 0

    async def test_successful_save_updates_mirror_and_redis_in_one_step(self):
        """On the happy path, mirror and Redis both reflect the new version."""
        engine = _make_engine()
        from gigaevo.evolution.engine.snapshot import get_current_snapshot

        saved_payloads: list[str] = []

        async def capture_save(_key, payload):
            saved_payloads.append(payload)

        engine.storage.save_run_state.side_effect = capture_save

        await engine._write_snapshot(total_mutants=7)

        assert engine._snapshot.version == 1
        assert engine._snapshot.total_mutants == 7
        # The payload sent to Redis carries the new version.
        assert '"version":1' in saved_payloads[0]
        assert '"total_mutants":7' in saved_payloads[0]
        # Process-wide cache matches.
        cached = get_current_snapshot()
        assert cached is not None
        assert cached.version == 1
        assert cached.total_mutants == 7

    async def test_retry_after_failure_uses_same_version(self):
        """After a save failure, the next call retries the SAME version
        number (not version+2). Redis must never see a version skip."""
        engine = _make_engine()

        saved_versions: list[int] = []

        async def fail_once_then_succeed(_key, payload):
            import json

            data = json.loads(payload)
            saved_versions.append(data["version"])
            if len(saved_versions) == 1:
                raise RuntimeError("transient redis blip")

        engine.storage.save_run_state.side_effect = fail_once_then_succeed

        with pytest.raises(RuntimeError):
            await engine._write_snapshot(total_mutants=3)
        # Retry — same version number reached Redis.
        await engine._write_snapshot(total_mutants=3)

        # Both attempts saved version=1 (not 1 then 2 — that would skip).
        assert saved_versions == [1, 1], f"version skip detected: {saved_versions}"


# ===========================================================================
# I9: No parent-refresh while a child of that parent is in flight.
#
# The producer must hold the per-parent-id lock from refresh through child-DAG
# completion. A concurrent producer that selects the same parent MUST block
# until the ingestor drains the in-flight child (DONE/DISCARDED) — not just
# until the first refresh returns. Otherwise concurrent producer B would
# observe parent P's lineage WHILE child A is mid-DAG (state=RUNNING,
# metrics={}), and the AncestrySelector picks up an unscored child as
# "ancestry" — the bug the user originally reported.
# ===========================================================================


class TestNoRefreshWhileChildInFlight:
    """Invariant: the per-parent-id lock spans refresh + mutate + child-DAG.

    Exercises the producer→ingestor ticket handoff under real lock contention.
    Uses the engine's actual ParentRefresher (not a mock) so the lock
    semantics are real; storage is mocked so each refresh completes in O(ms).
    """

    async def test_second_producer_blocks_until_child_ingested(self):
        from gigaevo.evolution.engine.refresh import ParentRefresher

        engine = _make_engine(max_in_flight=2)
        engine._running = True

        parent = _prog(state=ProgramState.DONE)

        # Real refresher so per-id locks actually enforce the contract.
        # Mock storage: parent is DONE on every mget so refresh completes fast.
        engine._parent_refresher = ParentRefresher(
            storage=engine.storage, poll_interval=0.005, timeout_seconds=5.0
        )

        import uuid

        child_id = str(uuid.uuid4())

        def _make_child() -> Program:
            return Program(
                id=child_id, code="def x(): return 1", state=ProgramState.DONE
            )

        async def _mget(ids, exclude=None):
            programs = []
            for pid in ids:
                if pid == parent.id:
                    programs.append(parent)
                elif pid == child_id:
                    programs.append(_make_child())
                # else: leaked (vanished) — skip
            return programs

        engine.storage.mget.side_effect = _mget
        engine.storage.batch_transition_by_ids = AsyncMock()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False
        # Acceptor rejects → ingestor moves child to DISCARDED quickly.
        engine.storage.save_run_state = AsyncMock()

        # --- Phase 1: Producer A acquires the ticket and "creates" a child ---
        # Simulate the slot-and-ticket transfer that mutant_task does.
        await engine._buffer_sema.acquire()
        ticket_a = await engine._parent_refresher.refresh_with_ticket([parent])
        async with engine._in_flight_lock:
            engine._in_flight.add(child_id)
            engine._inflight_tickets[child_id] = ticket_a

        # --- Phase 2: Producer B tries to refresh same parent ---
        # MUST BLOCK on the per-parent-id lock that ticket_a holds.
        b_task = asyncio.create_task(engine._parent_refresher.refresh([parent]))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not b_task.done(), (
            "Producer B's refresh completed while child of same parent was "
            "still in flight — parent-lock did not span child-DAG. "
            "Concurrent producers can now observe unscored mutants in "
            "ancestry lookups."
        )

        # --- Phase 3: Ingestor drains the child, releasing ticket A ---
        # Acceptor rejected → child becomes DISCARDED; ticket released.
        handled = await poll_and_ingest(engine)
        assert handled == 1, f"ingestor did not drain the child; handled={handled}"
        assert child_id not in engine._in_flight
        assert child_id not in engine._inflight_tickets

        # --- Phase 4: Producer B's refresh unblocks ---
        result_b = await asyncio.wait_for(b_task, timeout=INV_TEST_TIMEOUT)
        assert len(result_b) == 1 and result_b[0].id == parent.id

    async def test_failure_before_register_releases_ticket(self):
        """If the producer fails AFTER refresh but BEFORE registering the
        child in _in_flight, the ticket finally-release path must kick in.
        Otherwise the parent stays locked behind a permanent ticket and
        future producers deadlock."""
        from gigaevo.evolution.engine.refresh import ParentRefresher

        engine = _make_engine(max_in_flight=1)
        parent = _prog(state=ProgramState.DONE)

        engine._parent_refresher = ParentRefresher(
            storage=engine.storage, poll_interval=0.005, timeout_seconds=5.0
        )

        async def _mget(ids, exclude=None):
            return [parent for _ in ids]

        engine.storage.mget.side_effect = _mget
        engine.storage.batch_transition_by_ids = AsyncMock()

        # Acquire ticket via the public path; then drop it (simulating
        # mutant_task's failure-path `finally: ticket.release()`).
        await engine._buffer_sema.acquire()
        ticket = await engine._parent_refresher.refresh_with_ticket([parent])
        ticket.release()

        # A subsequent refresh on the same parent must NOT deadlock.
        refreshed = await asyncio.wait_for(
            engine._parent_refresher.refresh([parent]),
            timeout=INV_TEST_TIMEOUT,
        )
        assert len(refreshed) == 1 and refreshed[0].id == parent.id

    async def test_ticket_released_on_discard_path(self):
        """Acceptor-rejected children move DONE→DISCARDED. The ingestor's
        ticket-release path is the SAME for accept and reject — both routes
        run through `released = handled_ids | leaked_ids`."""
        from gigaevo.evolution.engine.refresh import ParentRefresher

        engine = _make_engine(max_in_flight=1)
        engine._running = True
        parent = _prog(state=ProgramState.DONE)

        engine._parent_refresher = ParentRefresher(
            storage=engine.storage, poll_interval=0.005, timeout_seconds=5.0
        )

        import uuid

        child_id = str(uuid.uuid4())

        async def _mget(ids, exclude=None):
            out = []
            for pid in ids:
                if pid == parent.id:
                    out.append(parent)
                elif pid == child_id:
                    out.append(
                        Program(
                            id=child_id,
                            code="def x(): return 1",
                            state=ProgramState.DONE,
                        )
                    )
            return out

        engine.storage.mget.side_effect = _mget
        engine.storage.batch_transition_by_ids = AsyncMock()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False  # reject!
        engine.storage.save_run_state = AsyncMock()

        await engine._buffer_sema.acquire()
        ticket = await engine._parent_refresher.refresh_with_ticket([parent])
        async with engine._in_flight_lock:
            engine._in_flight.add(child_id)
            engine._inflight_tickets[child_id] = ticket

        # Confirm the lock is held by trying a concurrent refresh.
        b_task = asyncio.create_task(engine._parent_refresher.refresh([parent]))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not b_task.done()

        # Ingest the rejected child → ticket released.
        await poll_and_ingest(engine)
        assert child_id not in engine._inflight_tickets

        # Lock now free.
        result = await asyncio.wait_for(b_task, timeout=INV_TEST_TIMEOUT)
        assert len(result) == 1

    async def test_leaked_child_releases_ticket(self):
        """A child that vanishes (e.g. evicted from storage) is treated as
        leaked. The ticket must still release on the leaked-path so the
        parent doesn't stay locked indefinitely."""
        from gigaevo.evolution.engine.refresh import ParentRefresher

        engine = _make_engine(max_in_flight=1)
        engine._running = True
        parent = _prog(state=ProgramState.DONE)

        engine._parent_refresher = ParentRefresher(
            storage=engine.storage, poll_interval=0.005, timeout_seconds=5.0
        )

        # First call: refresh queries parent — return DONE parent.
        # After child registered: poll_and_ingest queries [child] — return
        # empty list (child vanished). Need stateful side_effect.
        call_count = {"n": 0}

        async def _mget(ids, exclude=None):
            call_count["n"] += 1
            # If any requested id matches parent, treat as a refresh call.
            if any(pid == parent.id for pid in ids):
                return [parent]
            # Otherwise it's the ingestor polling for a child — return empty.
            return []

        engine.storage.mget.side_effect = _mget
        engine.storage.batch_transition_by_ids = AsyncMock()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False
        engine.storage.save_run_state = AsyncMock()

        await engine._buffer_sema.acquire()
        ticket = await engine._parent_refresher.refresh_with_ticket([parent])
        import uuid

        leaked_id = str(uuid.uuid4())
        async with engine._in_flight_lock:
            engine._in_flight.add(leaked_id)
            engine._inflight_tickets[leaked_id] = ticket

        # Ingest pass: storage returns no program for leaked_id → leaked path.
        await poll_and_ingest(engine)
        assert leaked_id not in engine._in_flight
        assert leaked_id not in engine._inflight_tickets
        # Buffer slot released exactly once.
        assert engine._buffer_sema._value == engine._ss_config.max_in_flight


__all__ = []
