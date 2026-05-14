"""Post-step hook contract + adjacent observability invariants.

Contract:

  * The hook fires from ``poll_and_ingest`` ONCE per sweep that lands
    at least one program in the archive.
  * It does NOT fire on a poll tick that handles only rejected or
    leaked programs — composition-injection-style hooks walk the
    entire G archive on every call, so firing on every tick would
    rebuild the tracker O(poll_freq) times even with no progress.
  * A hook that raises must NOT abort the surrounding ingestion —
    the ingest write to Redis has already committed by the time the
    hook runs, and aborting it would orphan accepted programs.
  * The hook is wall-clock bounded by
    ``engine.config.post_step_hook_timeout_s``. A hung or
    uncancellable hook must not wedge the ingestor.

Adjacent invariants:

  * ``ParentRefresher.timeout_seconds`` defaults to a *finite* bound
    so a stranded mutant task cannot block forever on a DAG-runner
    crash, leaking its in-flight slot.
  * The steady-state engine's final ingestion sweep logs a WARNING
    with the stuck-id list when its wall-clock deadline elapses
    before ``_in_flight`` drains. Without this, operators have no
    signal that a run shut down with leaked semaphore slots.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.ingestor import poll_and_ingest
from gigaevo.evolution.engine.refresh import ParentRefresher
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import EvolutionStopper, MaxMutantsStopper
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

HOOK_TEST_TIMEOUT = 5.0


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


def _make_engine(
    *,
    post_step_hook=None,
    max_in_flight: int = 4,
    max_mutants: int | None = None,
) -> SteadyStateEvolutionEngine:
    """Build a minimal SteadyStateEvolutionEngine with mocked dependencies.

    Mirrors the ``_make_ss_engine`` helper in ``test_steady_state.py`` but
    accepts a real ``post_step_hook`` callable so we can drive
    ``poll_and_ingest`` end-to-end against it.
    """
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()
    metrics_tracker.format_best_summary.return_value = ""

    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.snapshot = MagicMock()
    strategy.get_program_ids.return_value = []

    stopper = (
        MaxMutantsStopper(max_mutants)
        if max_mutants is not None
        else EvolutionStopper()
    )
    config = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        stopper=stopper,
        loop_interval=0.01,
    )

    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=config,
        writer=writer,
        metrics_tracker=metrics_tracker,
        post_step_hook=post_step_hook,
    )
    # Ensure ProgramStateManager calls in _ingest_batch are no-ops, too.
    engine.state = AsyncMock()
    return engine


def _stage_in_flight(engine, programs: list[Program]) -> list[str]:
    """Register programs in _in_flight and wire storage.mget to return them.

    Uses each Program's auto-generated UUID so pydantic validation in the
    ``Program.id`` field stays happy.
    """
    ids: list[str] = []
    for p in programs:
        engine._in_flight.add(p.id)
        ids.append(p.id)
    engine.storage.mget.return_value = programs
    return ids


# ---------------------------------------------------------------------------
# post_step_hook cadence
# ---------------------------------------------------------------------------


class TestPostStepHookCadence:
    async def test_fires_after_ingest_sweep_that_adds_program(self) -> None:
        """One added program in a sweep → hook fires exactly once."""
        calls: list[None] = []

        async def hook() -> None:
            calls.append(None)

        engine = _make_engine(post_step_hook=hook)
        prog = _prog()
        _stage_in_flight(engine, [prog])
        # Acceptor accepts; strategy accepts → ingestion path increments `added`.
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        await asyncio.wait_for(poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT)

        assert len(calls) == 1, "Hook must fire exactly once per accepted sweep"

    async def test_does_not_fire_when_zero_added(self) -> None:
        """Sweep that only rejects programs MUST NOT fire the hook.

        Composition injection walks the whole G archive on every call;
        firing on every poll tick (including reject-only sweeps) would
        rebuild it O(poll_freq) times even with no archive progress.
        """
        calls: list[None] = []

        async def hook() -> None:
            calls.append(None)

        engine = _make_engine(post_step_hook=hook)
        prog = _prog()
        _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False  # reject

        await asyncio.wait_for(poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT)

        assert calls == [], "Hook must not fire when no program was accepted"

    async def test_does_not_fire_on_empty_in_flight(self) -> None:
        """No in-flight programs → no hook call regardless of mock state."""
        calls: list[None] = []

        async def hook() -> None:
            calls.append(None)

        engine = _make_engine(post_step_hook=hook)
        # _in_flight is empty by default; poll_and_ingest returns 0
        await asyncio.wait_for(poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT)
        assert calls == []

    async def test_hook_failure_isolated_from_ingestion(self) -> None:
        """A buggy hook must not abort the surrounding sweep.

        The Redis ingest write has already committed by the time the
        hook runs, so raising would orphan accepted programs (no path
        back into the archive).
        """

        async def buggy_hook() -> None:
            raise RuntimeError("hook intentionally raises")

        engine = _make_engine(post_step_hook=buggy_hook)
        prog = _prog()
        [new_id] = _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        # poll_and_ingest must return normally — no propagated exception.
        handled = await asyncio.wait_for(
            poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT
        )
        assert handled >= 1, "Ingestion must continue past hook failure"
        # Slot was released — the buggy hook did not strand the in-flight slot.
        assert new_id not in engine._in_flight

    async def test_hook_unset_is_no_op(self) -> None:
        """Engine with no hook configured must not blow up on accepted sweeps."""
        engine = _make_engine(post_step_hook=None)
        prog = _prog()
        _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        handled = await asyncio.wait_for(
            poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT
        )
        assert handled >= 1


# ---------------------------------------------------------------------------
# post_step_hook is wall-clock bounded (no wedged-ingestor risk)
# ---------------------------------------------------------------------------


class TestPostStepHookTimeoutBound:
    """The hook is wrapped in ``asyncio.wait(..., timeout=...)`` and
    driven via an explicit ``create_task`` so the bound holds even
    against a hook that ignores ``CancelledError``. These tests pin:

    * Fast hooks complete normally.
    * A hook that exceeds the budget is cancelled, logged, and the
      ingestor returns successfully (not propagating TimeoutError).
    * A hook that ignores cancellation gets an orphan-warn log
      and the ingestor still returns (does NOT wedge on the grace wait).
    * Outer cancellation (ingestor_loop teardown) propagates to the
      hook task — no detached coroutine outlives ``poll_and_ingest``.
    * The default bound is in the minutes range (sanity).
    """

    async def test_fast_hook_completes_normally(self) -> None:
        """50ms hook with the default 300s budget — runs to completion."""
        calls: list[float] = []

        async def hook() -> None:
            await asyncio.sleep(0.05)
            calls.append(asyncio.get_event_loop().time())

        engine = _make_engine(post_step_hook=hook)
        prog = _prog()
        _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        await asyncio.wait_for(poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT)

        assert len(calls) == 1, "Fast hook must complete inside the budget"

    async def test_hung_hook_cancelled_after_budget(self) -> None:
        """Hook that sleeps forever → cancelled at the budget, ingestor returns."""
        hook_started = asyncio.Event()
        hook_was_cancelled = asyncio.Event()

        async def hung_hook() -> None:
            hook_started.set()
            try:
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                hook_was_cancelled.set()
                raise

        engine = _make_engine(post_step_hook=hung_hook)
        engine.config.post_step_hook_timeout_s = 0.1
        prog = _prog()
        _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        # Capture WARN lines so we can assert the budget message fired.
        from loguru import logger as loguru_logger

        warnings_emitted: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: warnings_emitted.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            handled = await asyncio.wait_for(
                poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT
            )
        finally:
            loguru_logger.remove(sink_id)

        warning_text = "\n".join(warnings_emitted)
        assert hook_started.is_set(), "Hook must have actually started"
        assert hook_was_cancelled.is_set(), "Hook must have been cancelled by the bound"
        assert "exceeded 0.1s budget" in warning_text, warning_text
        assert handled >= 1, "Ingestor must return success past hook timeout"

    async def test_uncooperative_hook_logs_orphan_warn(self) -> None:
        """A hook that catches CancelledError and keeps running must NOT
        wedge the ingestor — after the grace period we log + abandon.

        Load-bearing safety property: even a buggy hook that swallows
        cancel cannot extend our wait past
        ``post_step_hook_timeout_s + post_step_hook_cancel_grace_s``.
        """
        # Bounded-badness: swallow the first cancel (so the orphan-warn
        # path fires) then honour the second cancel so the test's
        # event-loop teardown can reap the task. An unbounded
        # ``while True: except: pass`` is a real orphan and blocks
        # pytest-asyncio's loop close.
        cancels_swallowed = 0

        async def stubborn_hook() -> None:
            nonlocal cancels_swallowed
            while True:
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    cancels_swallowed += 1
                    if cancels_swallowed >= 2:
                        raise

        engine = _make_engine(post_step_hook=stubborn_hook)
        engine.config.post_step_hook_timeout_s = 0.05
        engine.config.post_step_hook_cancel_grace_s = 0.1
        prog = _prog()
        _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        from loguru import logger as loguru_logger

        warnings_emitted: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: warnings_emitted.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        start = asyncio.get_event_loop().time()
        try:
            handled = await asyncio.wait_for(
                poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT
            )
        finally:
            loguru_logger.remove(sink_id)

        elapsed = asyncio.get_event_loop().time() - start
        warning_text = "\n".join(warnings_emitted)

        # The ingestor must NOT wait the full hook-sleep duration; cap
        # is timeout + grace + slack. 1.0s is a generous slack for CI.
        assert elapsed < 1.0, (
            f"Stubborn hook extended ingestor wait to {elapsed:.2f}s — "
            "bound is not load-bearing"
        )
        assert "ignored cancel within 0.1s" in warning_text, warning_text
        assert "exceeded 0.05s budget" in warning_text, warning_text
        assert handled >= 1, "Ingestor must return past uncooperative hook"

    async def test_outer_cancel_propagates_to_hook(self) -> None:
        """Cancel ``poll_and_ingest`` while the hook is running — the
        hook must be cancelled (not detached), and the sweep must
        re-raise ``CancelledError`` so the ingestor_loop sees a true
        cancellation, not a clean return.
        """
        hook_started = asyncio.Event()
        hook_cancelled = asyncio.Event()

        async def slow_hook() -> None:
            hook_started.set()
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                hook_cancelled.set()
                raise

        engine = _make_engine(post_step_hook=slow_hook)
        # Keep the hook bound large so the timeout doesn't race the
        # outer cancel — we want the cancel path, not the timeout path.
        engine.config.post_step_hook_timeout_s = 30.0
        prog = _prog()
        _stage_in_flight(engine, [prog])
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        sweep_task = asyncio.create_task(poll_and_ingest(engine))
        await asyncio.wait_for(hook_started.wait(), timeout=HOOK_TEST_TIMEOUT)
        sweep_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await sweep_task

        # Give the loop one more turn for the hook's `finally` to fire.
        await asyncio.sleep(0)

        assert hook_cancelled.is_set(), (
            "Outer cancellation did not reach the hook — hook task was orphaned"
        )
        assert sweep_task.cancelled(), (
            f"Sweep absorbed cancellation: done={sweep_task.done()}, "
            f"cancelled={sweep_task.cancelled()}"
        )

    def test_default_timeout_is_generous(self) -> None:
        """Sanity: the default bound is in the minutes range so a slow-
        but-legitimate hook (e.g. CompositionInjectionHook walking a
        10k archive) doesn't false-positive into a WARN cycle."""
        defaults = SteadyStateEngineConfig()
        timeout_s = defaults.post_step_hook_timeout_s
        grace_s = defaults.post_step_hook_cancel_grace_s

        assert timeout_s >= 60.0, (
            f"timeout {timeout_s}s too aggressive — would false-positive "
            "on legitimate slow hooks"
        )
        assert timeout_s <= 3600.0, (
            f"timeout {timeout_s}s too lax — a hung hook would wedge the "
            "ingestor for an unacceptable window"
        )
        assert 0.5 <= grace_s <= 30.0, f"grace {grace_s}s outside reasonable range"


# ---------------------------------------------------------------------------
# H3: ParentRefresher finite timeout default
# ---------------------------------------------------------------------------


class TestParentRefresherTimeoutDefault:
    def test_default_is_finite(self) -> None:
        """The default must be a finite number so a stranded mutant
        cannot block its task forever."""
        refresher = ParentRefresher(storage=AsyncMock())
        assert refresher._timeout_seconds is not None
        assert refresher._timeout_seconds > 0
        # Sanity-check the chosen budget is in the "minutes" range — small
        # enough to surface a stuck DAG within an experiment's lifetime,
        # large enough to absorb routine DAG queueing.
        assert refresher._timeout_seconds <= 3600.0
        assert refresher._timeout_seconds >= 60.0

    def test_explicit_none_still_accepted(self) -> None:
        """Passing ``None`` explicitly remains allowed for tests that
        need to disable the bound, but is NOT the default."""
        refresher = ParentRefresher(storage=AsyncMock(), timeout_seconds=None)
        assert refresher._timeout_seconds is None

    def test_explicit_finite_overrides_default(self) -> None:
        refresher = ParentRefresher(storage=AsyncMock(), timeout_seconds=10.0)
        assert refresher._timeout_seconds == 10.0


# ---------------------------------------------------------------------------
# Final-sweep observability
# ---------------------------------------------------------------------------


class TestFinalSweepWarning:
    """The final-sweep WARNING is reached by calling the extracted
    ``_final_ingestion_sweep`` directly. That keeps the test honest
    (it exercises the real loop, including its asyncio.shield path)
    while avoiding the brittle monkey-patching that driving full
    ``run()`` would require.
    """

    async def test_warns_when_deadline_elapses_with_in_flight(self) -> None:
        """If the deadline elapses with mutants still in-flight, the
        engine MUST emit a WARNING containing the count + IDs.

        Without this, an operator has no signal that the run shut down
        with leaked semaphore slots and stranded mutants in Redis.
        """
        engine = _make_engine(max_in_flight=2, max_mutants=1)
        stuck_id = "p-stuck-forever"
        engine._in_flight.add(stuck_id)

        # Patch poll_and_ingest to be a no-op so the sweep cannot drain.
        import gigaevo.evolution.engine.steady_state as ss_mod

        async def _noop_sweep(_engine):
            await asyncio.sleep(0.01)
            return 0

        original = ss_mod.poll_and_ingest
        ss_mod.poll_and_ingest = _noop_sweep

        # Capture loguru WARNING lines via a temporary sink.
        from loguru import logger as loguru_logger

        warnings_emitted: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: warnings_emitted.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            await asyncio.wait_for(
                engine._final_ingestion_sweep(deadline_seconds=0.05),
                timeout=HOOK_TEST_TIMEOUT,
            )
        finally:
            loguru_logger.remove(sink_id)
            ss_mod.poll_and_ingest = original

        warning_text = "\n".join(warnings_emitted)
        assert "final sweep deadline elapsed" in warning_text, warning_text
        assert stuck_id in warning_text, warning_text

    async def test_no_warning_when_in_flight_empty(self) -> None:
        """If ``_in_flight`` is already empty, the sweep exits immediately
        and the WARNING branch is NOT entered."""
        engine = _make_engine(max_in_flight=2, max_mutants=1)
        # _in_flight is empty by default

        from loguru import logger as loguru_logger

        warnings_emitted: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: warnings_emitted.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            await asyncio.wait_for(
                engine._final_ingestion_sweep(deadline_seconds=0.05),
                timeout=HOOK_TEST_TIMEOUT,
            )
        finally:
            loguru_logger.remove(sink_id)

        warning_text = "\n".join(warnings_emitted)
        assert "final sweep deadline elapsed" not in warning_text, warning_text

    async def test_sweep_drains_before_deadline(self) -> None:
        """If poll_and_ingest drains _in_flight quickly, no WARNING fires
        even with a tight deadline."""
        engine = _make_engine(max_in_flight=2, max_mutants=1)
        engine._in_flight.add("p1")

        import gigaevo.evolution.engine.steady_state as ss_mod

        # First call drains the in-flight set; subsequent calls are no-ops.
        async def _draining_sweep(_engine):
            _engine._in_flight.discard("p1")
            return 1

        original = ss_mod.poll_and_ingest
        ss_mod.poll_and_ingest = _draining_sweep

        from loguru import logger as loguru_logger

        warnings_emitted: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: warnings_emitted.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            await asyncio.wait_for(
                engine._final_ingestion_sweep(deadline_seconds=2.0),
                timeout=HOOK_TEST_TIMEOUT,
            )
        finally:
            loguru_logger.remove(sink_id)
            ss_mod.poll_and_ingest = original

        warning_text = "\n".join(warnings_emitted)
        assert "final sweep deadline elapsed" not in warning_text
        assert "p1" not in engine._in_flight


# ---------------------------------------------------------------------------
# Hook cadence across many mutants in a single sweep
# ---------------------------------------------------------------------------


class TestHookCadenceAcrossManyMutants:
    async def test_hook_fires_once_per_added_program(self) -> None:
        """Three accepted programs in a single sweep → one hook call.

        The hook cadence is "per ingest sweep that added ≥1", NOT
        "per added program" — composition_injection.inject_all() is
        idempotent across the (D, G) pair tracker, so per-program
        cadence would only waste archive walks.
        """
        calls: list[None] = []

        async def hook() -> None:
            calls.append(None)

        engine = _make_engine(post_step_hook=hook)
        progs = [_prog(), _prog(), _prog()]
        _stage_in_flight(engine, progs)
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        await asyncio.wait_for(poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT)

        assert calls == [None], "Sweep boundary, not per-program"

    async def test_mixed_accept_reject_in_sweep_fires_once(self) -> None:
        """At least one accepted program is enough to fire."""
        calls: list[None] = []

        async def hook() -> None:
            calls.append(None)

        engine = _make_engine(post_step_hook=hook)
        acc_prog = _prog()
        rej_prog = _prog()
        progs = [acc_prog, rej_prog]
        _stage_in_flight(engine, progs)
        engine.config.program_acceptor = MagicMock()
        # First program accepted, second rejected.
        accept_map = {acc_prog.id: True, rej_prog.id: False}
        engine.config.program_acceptor.is_accepted.side_effect = lambda p: accept_map[
            p.id
        ]
        engine.strategy.add.return_value = True

        await asyncio.wait_for(poll_and_ingest(engine), timeout=HOOK_TEST_TIMEOUT)
        assert calls == [None]


# ---------------------------------------------------------------------------
# Cancellation safety: _final_ingestion_sweep must not leak detached tasks
# ---------------------------------------------------------------------------


class TestFinalSweepCancellationSafety:
    """``asyncio.shield`` does NOT prevent ``CancelledError`` from
    propagating to the awaiter — it only protects the inner from
    cancellation. If the outer is cancelled, the inner ``poll_and_ingest``
    becomes a *detached* task that races ``_post_run_hook.on_run_complete``
    and engine teardown for access to ``storage`` and ``_in_flight``. The
    sweep must cancel-and-await its inner task on cancellation, not let
    it leak.
    """

    async def test_cancellation_does_not_leak_inner_task(self) -> None:
        """Cancel the sweep mid-poll and assert no zombie remains running."""
        engine = _make_engine(max_in_flight=2, max_mutants=1)
        engine._in_flight.add("p-blocker")

        import gigaevo.evolution.engine.steady_state as ss_mod

        # Inner poll_and_ingest that sleeps long enough that the outer
        # cancellation lands while it's mid-await. Without the explicit
        # cancel+wait_for in _final_ingestion_sweep, this task would be
        # detached and still running after the sweep returns.
        inner_started = asyncio.Event()
        inner_done = asyncio.Event()

        async def _slow_sweep(_engine):
            inner_started.set()
            try:
                await asyncio.sleep(5.0)
                return 0
            finally:
                inner_done.set()

        original = ss_mod.poll_and_ingest
        ss_mod.poll_and_ingest = _slow_sweep
        try:
            sweep_task = asyncio.create_task(
                engine._final_ingestion_sweep(deadline_seconds=5.0)
            )
            await asyncio.wait_for(inner_started.wait(), timeout=HOOK_TEST_TIMEOUT)
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep_task

            # The inner task must have been cancelled+awaited, not detached.
            # Give the event loop one more turn to settle.
            await asyncio.sleep(0)
            assert inner_done.is_set(), (
                "Inner poll_and_ingest was orphaned by sweep cancellation"
            )
        finally:
            ss_mod.poll_and_ingest = original

    async def test_cancellation_propagates_to_awaiter(self) -> None:
        """After settling the inner task, the sweep must re-raise the
        originating CancelledError. A blanket
        ``contextlib.suppress(BaseException)`` would make the sweep
        return normally on cancel, so the engine awaiter would see a
        clean shutdown and ``post_run_hook.on_run_complete`` would
        execute in a teardown context the supervisor never authorised.
        """
        engine = _make_engine(max_in_flight=2, max_mutants=1)
        engine._in_flight.add("p-blocker")

        import gigaevo.evolution.engine.steady_state as ss_mod

        inner_started = asyncio.Event()

        async def _slow_sweep(_engine):
            inner_started.set()
            await asyncio.sleep(5.0)
            return 0

        original = ss_mod.poll_and_ingest
        ss_mod.poll_and_ingest = _slow_sweep
        try:
            sweep_task = asyncio.create_task(
                engine._final_ingestion_sweep(deadline_seconds=5.0)
            )
            await asyncio.wait_for(inner_started.wait(), timeout=HOOK_TEST_TIMEOUT)
            sweep_task.cancel()
            with contextlib.suppress(BaseException) as _:
                await sweep_task
            # CancelledError must reach the awaiter as a true cancellation
            # (.cancelled() is True), not as a regular exception. The
            # distinction matters: asyncio.wait(..., return_when=
            # FIRST_COMPLETED) + the dispatcher/ingestor cleanup loop in
            # run() rely on .cancelled() to skip exception-style handling.
            # A laxer "done() and exception() is CancelledError" branch
            # would mask a regression that breaks that contract.
            assert sweep_task.cancelled(), (
                f"Sweep absorbed cancellation: done={sweep_task.done()}, "
                f"cancelled={sweep_task.cancelled()}, "
                f"exc={sweep_task.exception() if sweep_task.done() and not sweep_task.cancelled() else 'n/a'}"
            )
        finally:
            ss_mod.poll_and_ingest = original

    async def test_normal_completion_returns_without_cancellederror(self) -> None:
        """Negative: when not cancelled, the sweep returns cleanly even
        if the in-flight set never drains within the deadline. This pins
        the WARNING-and-return path so a future refactor of the cancel
        plumbing doesn't accidentally raise on the happy/timeout path.
        """
        engine = _make_engine(max_in_flight=2, max_mutants=1)
        engine._in_flight.add("p-stuck")

        import gigaevo.evolution.engine.steady_state as ss_mod

        async def _empty_sweep(_engine):
            await asyncio.sleep(0)
            return 0

        original = ss_mod.poll_and_ingest
        ss_mod.poll_and_ingest = _empty_sweep
        # Bypass the 5s default with a near-instant deadline so the
        # WARNING-and-return path completes inside the test budget.
        engine._ss_config.loop_interval = 0.01
        try:
            await asyncio.wait_for(
                engine._final_ingestion_sweep(deadline_seconds=0.05),
                timeout=HOOK_TEST_TIMEOUT,
            )
        finally:
            ss_mod.poll_and_ingest = original
