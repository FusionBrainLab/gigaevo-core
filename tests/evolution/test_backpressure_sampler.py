"""SteadyStateEvolutionEngine emits BACKPRESSURE_SAMPLE periodically.

Why this matters
----------------
Operators of a running experiment cannot today tell whether ``max_in_flight``
is being utilised — the only published value is the boot banner. The engine
must emit a periodic structured snapshot so a) the live profiler can render a
concurrency-over-time band and b) the canonical-events watchdog can fire if
the engine pegs at the cap for long stretches.

The sampler must:
- Honour ``backpressure_sample_interval`` as its emission cadence (decoupled
  from ``loop_interval`` so a 1Hz engine tick doesn't dump 86k log lines/day).
- Report producer/buffer ``held`` counts derived from
  ``max_in_flight - sema._value`` (the asyncio internal counter is the
  source of truth here).
- Snapshot ``len(_in_flight)`` under ``_in_flight_lock`` so the wire value is
  consistent with the production accounting.
- Tear down cleanly with the other loops in ``run()``'s ``finally``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.backpressure_sampler import backpressure_sampler_loop
from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine


def _make_engine(
    max_in_flight: int = 8,
    loop_interval: float = 0.01,
    backpressure_sample_interval: float = 0.01,
) -> SteadyStateEvolutionEngine:
    cfg = SteadyStateEngineConfig(
        max_in_flight=max_in_flight,
        loop_interval=loop_interval,
        backpressure_sample_interval=backpressure_sample_interval,
    )
    writer = MagicMock()
    writer.bind.return_value = writer
    return SteadyStateEvolutionEngine(
        config=cfg,
        storage=AsyncMock(),
        strategy=AsyncMock(),
        mutation_operator=AsyncMock(),
        writer=writer,
        metrics_tracker=MagicMock(),
    )


def _collect_emitted_events(monkeypatch: pytest.MonkeyPatch) -> list:
    captured: list = []

    def _fake_emit(event) -> None:
        captured.append(event)

    monkeypatch.setattr(
        "gigaevo.evolution.engine.backpressure_sampler.emit",
        _fake_emit,
    )
    return captured


@pytest.mark.asyncio
async def test_sampler_emits_at_configured_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At least N samples fire over N*backpressure_sample_interval seconds."""
    captured = _collect_emitted_events(monkeypatch)
    engine = _make_engine(max_in_flight=4, backpressure_sample_interval=0.02)
    engine._running = True
    task = asyncio.create_task(backpressure_sampler_loop(engine))
    try:
        # Sleep just over 3 ticks — at least 3 samples should land.
        await asyncio.sleep(0.075)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert len(captured) >= 3, f"too few samples: {len(captured)}"
    assert all(ev.event == "BACKPRESSURE_SAMPLE" for ev in captured)


@pytest.mark.asyncio
async def test_sampler_records_actual_held_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """held = max_in_flight - sema._value (matches asyncio's internal counter)."""
    captured = _collect_emitted_events(monkeypatch)
    engine = _make_engine(max_in_flight=4, loop_interval=0.01)
    engine._running = True
    # Acquire some slots without releasing — simulates work-in-progress.
    await engine._producer_sema.acquire()
    await engine._producer_sema.acquire()
    await engine._buffer_sema.acquire()
    async with engine._in_flight_lock:
        engine._in_flight.add("aaaaaaaa")

    task = asyncio.create_task(backpressure_sampler_loop(engine))
    try:
        await asyncio.sleep(0.025)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert captured, "expected at least one sample"
    s = captured[0]
    assert s.producer_held == 2
    assert s.buffer_held == 1
    assert s.in_flight == 1
    assert s.max_in_flight == 4


@pytest.mark.asyncio
async def test_sampler_cadence_is_decoupled_from_loop_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a fast ``loop_interval`` must NOT speed up the sampler.

    Before the cadence split, BACKPRESSURE_SAMPLE rode on ``loop_interval``
    (1Hz default) and dumped ~86k log lines per day. The sampler must read
    ``backpressure_sample_interval`` exclusively.
    """
    captured = _collect_emitted_events(monkeypatch)
    # Fast engine tick, slow sample tick — sampler should respect the latter.
    engine = _make_engine(
        max_in_flight=4,
        loop_interval=0.001,
        backpressure_sample_interval=0.2,
    )
    engine._running = True
    task = asyncio.create_task(backpressure_sampler_loop(engine))
    try:
        # 0.05s is 50 loop_intervals but only 0.25 sample_intervals → ≤1 sample.
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert len(captured) <= 1, (
        f"sampler sped up to loop_interval cadence ({len(captured)} samples in 50ms)"
    )


@pytest.mark.asyncio
async def test_sampler_stops_when_engine_running_flag_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``_running=False`` must drain the sampler within one tick."""
    captured = _collect_emitted_events(monkeypatch)
    engine = _make_engine(max_in_flight=4, loop_interval=0.01)
    engine._running = True
    task = asyncio.create_task(backpressure_sampler_loop(engine))
    await asyncio.sleep(0.025)
    engine._running = False
    # Give it well under a real-world tick to exit.
    await asyncio.wait_for(task, timeout=0.1)
    assert task.done()
    assert not task.cancelled()
    assert captured, "expected at least one sample before stop"


@pytest.mark.asyncio
async def test_sampler_writes_scalars_to_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each emitted sample must also write 5 scalars to engine._writer.

    These scalars feed TensorBoard via the live profiler — without them,
    operators can only see queue depth in event logs, not in plots. The
    metrics are namespaced under ``backpressure/`` so they don't collide
    with other engine scalars.
    """
    _collect_emitted_events(monkeypatch)
    engine = _make_engine(max_in_flight=4, backpressure_sample_interval=0.01)
    engine._running = True

    # Track all scalar() calls on any writer derived from engine._writer.
    scalar_calls: list[tuple[str, float, tuple[str, ...]]] = []

    bound_writer = MagicMock()

    def _record_scalar(metric: str, value: float, **kw: object) -> None:
        path = tuple(kw.get("path", ())) if kw.get("path") else ()
        scalar_calls.append((metric, float(value), path))

    bound_writer.scalar.side_effect = _record_scalar
    bound_writer.bind.return_value = bound_writer
    engine._writer.bind.return_value = bound_writer

    # Pre-occupy slots so values are non-trivial.
    await engine._producer_sema.acquire()
    await engine._buffer_sema.acquire()
    await engine._buffer_sema.acquire()
    async with engine._in_flight_lock:
        engine._in_flight.add("aaaaaaaa")
        engine._llm_active = 1

    task = asyncio.create_task(backpressure_sampler_loop(engine))
    try:
        await asyncio.sleep(0.025)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Every expected metric must appear at least once.
    metric_names = {m for m, _, _ in scalar_calls}
    expected = {
        "producer_held",
        "buffer_held",
        "in_flight",
        "llm_active",
        "max_in_flight",
    }
    assert expected.issubset(metric_names), (
        f"missing scalars: {expected - metric_names}"
    )

    # Spot-check a known value: max_in_flight == 4.
    max_calls = [v for m, v, _ in scalar_calls if m == "max_in_flight"]
    assert max_calls and all(v == 4.0 for v in max_calls)

    # Held counts should reflect the pre-acquired slots.
    producer_calls = [v for m, v, _ in scalar_calls if m == "producer_held"]
    buffer_calls = [v for m, v, _ in scalar_calls if m == "buffer_held"]
    assert producer_calls and producer_calls[0] == 1.0
    assert buffer_calls and buffer_calls[0] == 2.0


@pytest.mark.asyncio
async def test_run_starts_and_cancels_the_sampler_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run()`` registers ``_sampler_task`` and cancels it in ``finally``."""
    engine = _make_engine(max_in_flight=2, loop_interval=0.01)

    # Replace dispatcher + ingestor with sleepers so run() doesn't pull on
    # the AsyncMock storage stack — we only want to check the sampler is
    # wired into the lifecycle.
    async def _sleeper(_e):
        await asyncio.sleep(0.05)
        return None

    monkeypatch.setattr(
        "gigaevo.evolution.engine.steady_state.dispatcher_loop", _sleeper
    )
    monkeypatch.setattr("gigaevo.evolution.engine.steady_state.ingestor_loop", _sleeper)
    # Skip Phase 0 — its helpers call storage methods we haven't mocked.
    monkeypatch.setattr(engine, "_await_idle", AsyncMock(return_value=None))
    monkeypatch.setattr(
        engine, "_ingest_completed_programs", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(engine, "_write_snapshot", AsyncMock(return_value=None))
    monkeypatch.setattr(engine, "_final_ingestion_sweep", AsyncMock(return_value=None))
    engine.storage.snapshot = MagicMock()

    captured = _collect_emitted_events(monkeypatch)

    await engine.run()

    # Sampler task must have been created AND torn down.
    assert engine._sampler_task is not None
    assert engine._sampler_task.done()
    # The cancel path lands as either cancelled or a clean return.
    # Either way at least one sample fired during the sleeper window.
    assert captured, "expected sampler emissions during run()"
