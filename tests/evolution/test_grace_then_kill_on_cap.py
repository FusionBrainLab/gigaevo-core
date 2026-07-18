"""Terminal drain honours a bounded post-cap grace before giving up.

When ``max_mutants`` is reached the engine drains in-flight child DAGs. A single
pathologically slow eval must not hold the whole run hostage up to
``terminal_drain_timeout_s``. ``post_cap_drain_grace_s`` bounds that wait: fast
evals still land, then the drain returns cleanly and teardown
(``dag_runner.stop()``) SIGKILLs the stragglers. ``None`` restores the legacy
drain-or-raise contract.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine import steady_state as steady_state_mod
from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine


def _make_engine(
    *,
    post_cap_drain_grace_s: float | None,
    terminal_drain_timeout_s: float,
) -> SteadyStateEvolutionEngine:
    cfg = SteadyStateEngineConfig(
        max_in_flight=4,
        loop_interval=0.01,
        terminal_drain_timeout_s=terminal_drain_timeout_s,
        post_cap_drain_grace_s=post_cap_drain_grace_s,
    )
    writer = MagicMock()
    writer.bind.return_value = writer
    engine = SteadyStateEvolutionEngine(
        config=cfg,
        storage=AsyncMock(),
        strategy=AsyncMock(),
        mutation_operator=AsyncMock(),
        writer=writer,
        metrics_tracker=MagicMock(),
    )
    # A live ingestor is not needed for the drain-timing contract; None skips
    # the "ingestor exited during drain" guard.
    engine._ingestor_task = None
    return engine


def test_default_grace_is_on() -> None:
    """The framework ships grace-then-kill on by default (30 s)."""
    assert SteadyStateEngineConfig().post_cap_drain_grace_s == 30.0


@pytest.mark.asyncio
async def test_grace_returns_cleanly_with_stragglers() -> None:
    """A straggler that never terminates does NOT raise: the drain returns
    once the grace elapses, leaving the straggler for teardown to kill."""
    engine = _make_engine(post_cap_drain_grace_s=0.05, terminal_drain_timeout_s=100.0)
    async with engine._in_flight_lock:
        engine._in_flight.add("straggler-1")

    started = time.monotonic()
    # Must return (no exception) — this is the grace-then-kill contract.
    await asyncio.wait_for(engine._await_terminal_drain(), timeout=2.0)
    elapsed = time.monotonic() - started

    # Returned on the grace budget, not after the (much larger) drain timeout.
    assert elapsed < 1.0
    # Straggler was abandoned, not drained — teardown (dag_runner.stop()) kills it.
    assert "straggler-1" in engine._in_flight


@pytest.mark.asyncio
async def test_grace_none_preserves_legacy_raise() -> None:
    """With the grace disabled, an undrained child still raises TimeoutError
    once terminal_drain_timeout_s elapses (legacy safety contract)."""
    engine = _make_engine(post_cap_drain_grace_s=None, terminal_drain_timeout_s=0.05)
    async with engine._in_flight_lock:
        engine._in_flight.add("straggler-2")

    with pytest.raises(TimeoutError, match="terminal drain timed out"):
        await asyncio.wait_for(engine._await_terminal_drain(), timeout=2.0)


def _fake_clock(monkeypatch, ticks: list[float]) -> None:
    """Drive ``steady_state``'s ``time.monotonic`` off a scripted tick list.

    Rebinds only the module's ``time`` name (a shim), so the stdlib clock the
    event loop relies on is untouched. The last tick is held once exhausted.
    """
    seq = iter(ticks)
    last = [ticks[0]]

    def _monotonic() -> float:
        try:
            last[0] = next(seq)
        except StopIteration:
            pass
        return last[0]

    monkeypatch.setattr(steady_state_mod, "time", SimpleNamespace(monotonic=_monotonic))


@pytest.mark.asyncio
async def test_delayed_wakeup_drain_earlier_raises(monkeypatch) -> None:
    """When the loop wakes past BOTH deadlines at once and the drain timeout is
    the earlier bound, the drain must RAISE — the grace elapsing too does not
    launder the straggler into a clean return."""
    engine = _make_engine(post_cap_drain_grace_s=0.02, terminal_drain_timeout_s=0.01)
    async with engine._in_flight_lock:
        engine._in_flight.add("straggler-3")

    # start=0.0 → drain_deadline=0.01, grace_deadline=0.02; single delayed wake
    # at 0.03 lands past both. Earlier bound is the drain timeout → raise.
    _fake_clock(monkeypatch, [0.0, 0.03])

    with pytest.raises(TimeoutError, match="terminal drain timed out"):
        await engine._await_terminal_drain()


@pytest.mark.asyncio
async def test_delayed_wakeup_grace_earlier_returns(monkeypatch) -> None:
    """Mirror of the above: when the grace is the earlier bound and the loop
    wakes past both deadlines, the grace wins and the drain returns cleanly."""
    engine = _make_engine(post_cap_drain_grace_s=0.02, terminal_drain_timeout_s=0.05)
    async with engine._in_flight_lock:
        engine._in_flight.add("straggler-4")

    # start=0.0 → grace_deadline=0.02, drain_deadline=0.05; wake at 0.06 past
    # both. Earlier bound is the grace → return, straggler left for teardown.
    _fake_clock(monkeypatch, [0.0, 0.06])

    await engine._await_terminal_drain()
    assert "straggler-4" in engine._in_flight


@pytest.mark.asyncio
async def test_fast_eval_within_grace_is_fully_drained() -> None:
    """An eval that terminates within the grace window is drained normally —
    the grace never fires, so no straggler is abandoned."""
    engine = _make_engine(post_cap_drain_grace_s=1.0, terminal_drain_timeout_s=100.0)
    async with engine._in_flight_lock:
        engine._in_flight.add("fast-1")

    async def _terminate_soon() -> None:
        await asyncio.sleep(0.03)
        async with engine._in_flight_lock:
            engine._in_flight.discard("fast-1")

    terminator = asyncio.create_task(_terminate_soon())
    try:
        await asyncio.wait_for(engine._await_terminal_drain(), timeout=2.0)
    finally:
        await terminator

    # Drained to empty — nothing abandoned.
    assert not engine._in_flight
