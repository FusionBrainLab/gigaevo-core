"""SteadyStateEvolutionEngine — continuous async dispatch + ingest stream.

Composes :func:`dispatcher_loop` and :func:`ingestor_loop`. Archive
programs are re-evaluated only when they are themselves selected as
parents (:class:`gigaevo.evolution.engine.refresh.ParentRefresher`);
there is no global archive refresh.

See ``docs/superpowers/specs/2026-05-12-steady-state-engine-audit-and-redesign.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import cast

from loguru import logger

from gigaevo.evolution.engine.backpressure_sampler import backpressure_sampler_loop
from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.dispatcher import dispatcher_loop
from gigaevo.evolution.engine.ingestor import ingestor_loop, poll_and_ingest
from gigaevo.evolution.engine.refresh import ParentRefresher, ParentRefreshTicket


class SteadyStateEvolutionEngine(EvolutionEngine):
    """Steady-state engine. Composes dispatcher + ingestor + ParentRefresher."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = cast(SteadyStateEngineConfig, self.config)
        if not isinstance(cfg, SteadyStateEngineConfig):
            raise TypeError(
                f"SteadyStateEvolutionEngine requires SteadyStateEngineConfig, "
                f"got {type(self.config).__name__}"
            )
        self._ss_config: SteadyStateEngineConfig = cfg

        self._in_flight: set[str] = set()
        # Parent-refresh tickets transferred from producer (mutant_task) to
        # ingestor. The ticket holds the per-parent-id locks that prevent
        # another producer from refreshing the same parents while THIS
        # mutant's DAG is still in flight. Ownership transfers atomically
        # with ``_in_flight.add(new_id)`` under ``_in_flight_lock`` and is
        # released by the ingestor when the child reaches DONE/DISCARDED.
        # Keyed by mutant id (the in-flight registration key).
        self._inflight_tickets: dict[str, ParentRefreshTicket] = {}
        # Two-sema model: producer pool caps concurrent (refresh + LLM); buffer
        # pool caps produced-but-not-yet-ingested mutants. Both sized from the
        # single ``max_in_flight`` knob; steady-state pipeline depth ~2 × N.
        # See docs/superpowers/specs/2026-05-13-mutation-throughput-two-sema-design.md.
        self._producer_sema = asyncio.Semaphore(self._ss_config.max_in_flight)
        self._buffer_sema = asyncio.Semaphore(self._ss_config.max_in_flight)
        self._in_flight_lock = asyncio.Lock()

        # Counter of tasks actively in LLM inference. Incremented before
        # generate_one_mutation, decremented after. Sampled by backpressure_sampler
        # to break down producer occupancy into LLM vs DAG phases.
        self._llm_active: int = 0

        self._parent_refresher = ParentRefresher(storage=self.storage)

        self._dispatcher_task: asyncio.Task | None = None
        self._ingestor_task: asyncio.Task | None = None
        # Observability sidecar: emits BackpressureSample canonical events at
        # config.backpressure_sample_interval cadence so a runner log carries a time-series
        # of producer/buffer/in_flight held counts. Lifecycle mirrors the
        # dispatcher/ingestor tasks (start in run(), cancel in finally).
        self._sampler_task: asyncio.Task | None = None

    async def run(self) -> None:
        logger.info(
            "[SteadyState] Start | producer_sema={} buffer_sema={} "
            "(max_in_flight={}) stopper={}",
            self._ss_config.max_in_flight,
            self._ss_config.max_in_flight,
            self._ss_config.max_in_flight,
            type(self._ss_config.stopper).__name__,
        )
        self._running = True
        self._run_start_time = time.monotonic()

        await self._write_snapshot(
            total_mutants=self.metrics.iteration,
            programs_processed=self.metrics.programs_processed,
        )

        try:
            # Phase 0: drain initial seed population (already QUEUED by loader)
            await self._await_idle()
            await self._ingest_completed_programs()
            self.storage.snapshot.bump(incremental=True)
            await self._write_snapshot(
                programs_processed=self.metrics.programs_processed
            )

            if self._pre_step_hook:
                await self._pre_step_hook()

            self._dispatcher_task = asyncio.create_task(
                dispatcher_loop(self), name="ss-dispatcher"
            )
            self._ingestor_task = asyncio.create_task(
                ingestor_loop(self), name="ss-ingestor"
            )
            self._sampler_task = asyncio.create_task(
                backpressure_sampler_loop(self), name="ss-backpressure-sampler"
            )

            done, pending = await asyncio.wait(
                [self._dispatcher_task, self._ingestor_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

            loop_exc = None
            for t in done:
                if not t.cancelled():
                    exc = t.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        logger.error("[SteadyState] Loop failed: {}", exc)
                        loop_exc = exc

            if loop_exc is not None:
                raise loop_exc

        except asyncio.CancelledError:
            logger.debug("[SteadyState] run() cancelled")
            raise
        finally:
            self._running = False
            # asyncio.wait() does NOT cancel its waited tasks when the
            # outer coroutine is cancelled, so the dispatcher and ingestor
            # may still be running here. Cancel them explicitly; each task
            # cleans up its own spawned mutant tasks (releasing semaphore
            # slots) in its own finally block.
            for loop_task in (
                self._dispatcher_task,
                self._ingestor_task,
                self._sampler_task,
            ):
                if loop_task is not None and not loop_task.done():
                    loop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await loop_task

            # Sweep re-raises CancelledError after the inner poll_and_ingest
            # is settled, so cancellation reaches our awaiter without leaking
            # the inner task. The finalizer (post_run_hook) must still run
            # — cancellation is a shutdown signal, not a "skip cleanup" one.
            sweep_cancelled = False
            try:
                await self._final_ingestion_sweep(deadline_seconds=5.0)
            except asyncio.CancelledError:
                sweep_cancelled = True

            try:
                await self._post_run_hook.on_run_complete(self.storage)
            except Exception as exc:
                logger.error("[SteadyState] post-run hook failed: {}", exc)
            logger.info("[SteadyState] Stopped")
            if sweep_cancelled:
                # `from None` keeps the traceback clean: a Redis blip the
                # `suppress(Exception)` in the sweep swallowed would
                # otherwise dangle in __context__ and mislead the operator.
                raise asyncio.CancelledError from None

    async def _final_ingestion_sweep(self, *, deadline_seconds: float) -> None:
        """Drain DONE/DISCARDED out of ``_in_flight`` after the loops exit.

        Releases ``_buffer_sema`` slots that a mutant cancelled between
        ``_in_flight.add`` and the slot release would otherwise leak —
        ``mutant_task``'s ``finally`` guards ``slot_transferred=True`` and
        refuses to release, expecting the ingestor to. On normal completion
        the DAG may still be flipping QUEUED→RUNNING→DONE for the last few
        in-flight mutants, so we sleep between empty passes instead of giving
        up immediately, bounded by ``deadline_seconds``.

        Cancellation semantics: each ``poll_and_ingest`` pass is wrapped in
        an explicit :class:`asyncio.Task` so we can wait for it (briefly) to
        finish if our awaiter is cancelled. A bare ``asyncio.shield`` would
        let the inner task become detached on cancellation, where it would
        race ``_post_run_hook.on_run_complete`` and the engine teardown for
        access to ``storage`` and ``_in_flight``. Instead we cancel and
        await the inner on a best-effort timeout — the inner's cancel
        cleanup latency caps the wall-clock cost, not the timeout
        parameter; ``wait_for`` returns only once the inner is terminal,
        guaranteeing no zombie coroutine outlives this method. After the
        inner is settled we re-raise the originating ``CancelledError`` so
        the caller's teardown order is preserved (the ``BaseException``
        family — ``KeyboardInterrupt``, ``SystemExit`` — propagates
        intact; we only swallow ``Exception`` from the cleanup wait).

        Args:
            deadline_seconds: Wall-clock budget for this sweep. If the
                budget elapses before ``_in_flight`` drains, emits a
                WARNING with the stuck-id list so the operator can
                correlate with stranded DAGs in Redis.
        """
        sweep_deadline = time.monotonic() + deadline_seconds
        cancelled = False
        while self._in_flight and time.monotonic() < sweep_deadline:
            inner = asyncio.create_task(poll_and_ingest(self))
            try:
                handled = await asyncio.shield(inner)
            except asyncio.CancelledError:
                # Cancel inner and wait up to 1s so it can't run detached
                # past storage.close() (which would raise
                # ConnectionClosedError into a coroutine with no caller).
                # `suppress(CancelledError)` swallows a double-cancel from
                # wait_for; a TimeoutError is logged so the operator can
                # see the orphan risk rather than dropping it silently.
                inner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    try:
                        await asyncio.wait_for(inner, timeout=1.0)
                    except TimeoutError:
                        logger.warning(
                            "[SteadyState] final sweep inner-task did not "
                            "honor cancel within 1s — potential orphan "
                            "coroutine (storage.close() may follow)"
                        )
                    except Exception as exc:
                        logger.warning(
                            "[SteadyState] final sweep inner-task errored "
                            "on cancel: {}",
                            exc,
                        )
                cancelled = True
                break
            except Exception as exc:
                logger.warning("[SteadyState] final sweep failed: {}", exc)
                break
            if handled == 0:
                try:
                    await asyncio.sleep(self._ss_config.loop_interval)
                except asyncio.CancelledError:
                    cancelled = True
                    break

        # Observability: if the deadline elapsed before _in_flight
        # drained, surface count + IDs so an operator can correlate
        # with stuck DAGs in Redis (state=QUEUED/RUNNING) rather than
        # silently leaking semaphore slots from the engine's view.
        # Snapshot under the lock so the WARNING is consistent with
        # in-memory state at one instant — no torn read with another
        # poll_and_ingest() concurrently mutating _in_flight.
        # Skipped on cancel: the WARNING is for deadline-elapsed
        # diagnostics, not shutdown-was-aborted.
        if self._in_flight and not cancelled:
            async with self._in_flight_lock:
                stuck = sorted(self._in_flight)
            logger.warning(
                "[SteadyState] final sweep deadline elapsed with {} "
                "in-flight mutant(s) still pending; _buffer_sema slots "
                "will be released on next engine start. stuck_ids={}",
                len(stuck),
                stuck[:10] + (["..."] if len(stuck) > 10 else []),
            )

        # Propagate the originating cancellation. The caller's `finally`
        # in `run()` invoked us; if it was itself cancelled, the cancel
        # must reach the engine's awaiter rather than being silently
        # absorbed and letting `_post_run_hook.on_run_complete` execute
        # in a teardown context the supervisor didn't authorise.
        # `from None` strips the suppressed Exception (e.g. Redis blip)
        # from __context__ so the traceback isn't misleadingly chained.
        if cancelled:
            raise asyncio.CancelledError from None
