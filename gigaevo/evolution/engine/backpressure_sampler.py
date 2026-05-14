"""Periodic emitter for the BACKPRESSURE_SAMPLE canonical event.

One sample per ``config.backpressure_sample_interval`` while the engine is
running. The sample is the only published view of the two-sema model's
runtime behaviour — a flat log line ``producer_sema=N buffer_sema=M`` could
not be aggregated into a time series cheaply, and the structured event drops
straight into the existing ``log_audit`` / live-profiler plumbing.

Cadence is intentionally decoupled from ``loop_interval`` (the engine's
1Hz snapshot tick): a 1Hz event stream produces ~86k log lines per day on
a long run, swamping every other event with low-information samples.

Why a dedicated loop instead of folding it into the dispatcher
--------------------------------------------------------------
The dispatcher blocks on ``producer_sema.acquire()`` — if the pipeline is
full, *no* sample would land while the user is most curious about the cap
saturation. A standalone task with its own ``asyncio.sleep`` rhythm
guarantees fixed-cadence emission regardless of pipeline pressure.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import BackpressureSample

if TYPE_CHECKING:
    from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine


async def backpressure_sampler_loop(engine: SteadyStateEvolutionEngine) -> None:
    """Emit one ``BackpressureSample`` per ``backpressure_sample_interval``
    until cancelled.

    Cancellation behaviour: bare ``CancelledError`` propagates so the
    supervisor's ``await loop_task`` resolves promptly. Any other exception
    is logged at WARNING — the sampler is observability, not load-bearing,
    so we must not take the engine down with us if e.g. an event-validation
    crash sneaks in.
    """
    cfg = engine._ss_config
    cap = cfg.max_in_flight
    interval = cfg.backpressure_sample_interval
    # Bind once: the writer is created in EvolutionEngine.__init__ as
    # writer.bind(path=["evolution_engine"]); we add a "backpressure" segment
    # so each scalar lands at evolution_engine/backpressure/<metric> and gets
    # its own TensorBoard panel. Binding inside the loop would re-create the
    # bound writer 1Hz × cap_seconds, which is wasted allocation.
    bp_writer = engine._writer.bind(path=["backpressure"])
    try:
        while engine._running:
            try:
                async with engine._in_flight_lock:
                    in_flight = len(engine._in_flight)
                    llm_active = engine._llm_active
                producer_held = cap - engine._producer_sema._value
                buffer_held = cap - engine._buffer_sema._value
                sample = BackpressureSample(
                    producer_held=max(0, producer_held),
                    buffer_held=max(0, buffer_held),
                    in_flight=min(cap, in_flight),
                    max_in_flight=cap,
                    llm_active=max(0, llm_active),
                )
                emit(sample)
                # Mirror the sample into TensorBoard so operators see queue
                # depth in plots, not just in event logs. Wrapped so a writer
                # backend hiccup never crashes the observability sidecar.
                try:
                    bp_writer.scalar("producer_held", float(sample.producer_held))
                    bp_writer.scalar("buffer_held", float(sample.buffer_held))
                    bp_writer.scalar("in_flight", float(sample.in_flight))
                    bp_writer.scalar("llm_active", float(sample.llm_active))
                    bp_writer.scalar("max_in_flight", float(sample.max_in_flight))
                except Exception as scalar_exc:  # pragma: no cover — defensive
                    logger.warning(
                        "[BackpressureSampler] scalar write failed: {}", scalar_exc
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Diagnostic event — never crash the engine because of it.
                logger.warning("[BackpressureSampler] sample failed: {}", exc)
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
    except asyncio.CancelledError:
        # Quiet exit — cancel is the expected shutdown signal.
        raise
