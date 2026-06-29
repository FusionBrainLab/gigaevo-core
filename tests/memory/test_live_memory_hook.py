"""Tests for LiveMemoryRefreshHook bounded-sweep behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import uuid

import pytest

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.memory.live_memory_hook import LiveMemoryRefreshHook
from gigaevo.programs.program import Program


class _StubStorage:
    def __init__(self, programs: list[Program]) -> None:
        self._programs = list(programs)

    async def get_all(self, *, exclude=None):  # type: ignore[no-untyped-def]
        return list(self._programs)


class _RecordingTracker(IncrementalPostRunHook):
    def __init__(self) -> None:
        self.calls: list[list[Program]] = []
        self.posterior_calls: list[list[Program] | None] = []

    async def on_run_complete(self, storage) -> None:  # type: ignore[no-untyped-def]
        pass

    async def run_increment(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None = None,
    ) -> None:
        self.calls.append(list(programs))
        self.posterior_calls.append(
            None if posterior_programs is None else list(posterior_programs)
        )


def _make_program(idx: int, created_at: datetime) -> Program:
    # Deterministic UUID5 so test assertions can compare on id; code field
    # required by Program schema.
    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"layer1-test-{idx}"))
    return Program(id=pid, code=f"# program {idx}", created_at=created_at)


@pytest.fixture
def five_programs() -> list[Program]:
    base = datetime(2026, 5, 24, 0, 0, 0, tzinfo=UTC)
    return [_make_program(i, base + timedelta(seconds=i)) for i in range(5)]


@pytest.mark.asyncio
async def test_unbounded_default_passes_all_programs(five_programs):
    """Default (max_programs_per_sweep=None) preserves legacy behaviour."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(tracker=tracker, storage=storage, refresh_every=1)

    await hook()

    assert len(tracker.calls) == 1
    assert {p.id for p in tracker.calls[0]} == {p.id for p in five_programs}


@pytest.mark.asyncio
async def test_bounded_sweep_passes_only_newest_n(five_programs):
    """max_programs_per_sweep=2 should pass the 2 NEWEST programs (by created_at)."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(
        tracker=tracker,
        storage=storage,
        refresh_every=1,
        max_programs_per_sweep=2,
    )

    await hook()

    assert len(tracker.calls) == 1
    passed_ids = {p.id for p in tracker.calls[0]}
    expected_newest = {five_programs[3].id, five_programs[4].id}
    assert passed_ids == expected_newest


@pytest.mark.asyncio
async def test_bounded_sweep_posterior_gets_full_pool(five_programs):
    """Capping the writer window must NOT starve the injection posterior:
    the full program pool (parent lineage intact) is passed as posterior_programs."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(
        tracker=tracker,
        storage=storage,
        refresh_every=1,
        max_programs_per_sweep=2,
    )

    await hook()

    # Writer window stays capped to the 2 newest.
    assert {p.id for p in tracker.calls[0]} == {
        five_programs[3].id,
        five_programs[4].id,
    }
    # Posterior population is the FULL pool, regardless of the writer cap.
    assert {p.id for p in tracker.posterior_calls[0]} == {p.id for p in five_programs}


@pytest.mark.asyncio
async def test_unbounded_posterior_gets_full_pool(five_programs):
    """Unbounded sweep also routes the full pool to the posterior channel."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(tracker=tracker, storage=storage, refresh_every=1)

    await hook()

    assert {p.id for p in tracker.posterior_calls[0]} == {p.id for p in five_programs}


@pytest.mark.asyncio
async def test_bounded_sweep_larger_than_pool_passes_all(five_programs):
    """max_programs_per_sweep > pool size returns the full pool, no error."""
    tracker = _RecordingTracker()
    storage = _StubStorage(five_programs)
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=storage, refresh_every=1, max_programs_per_sweep=100
    )

    await hook()

    assert {p.id for p in tracker.calls[0]} == {p.id for p in five_programs}


@pytest.mark.asyncio
async def test_cadence_gate_unchanged_by_bound():
    """Bounded hook still respects refresh_every cadence."""
    tracker = _RecordingTracker()
    storage = _StubStorage([_make_program(0, datetime.now(UTC))])
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=storage, refresh_every=3, max_programs_per_sweep=10
    )

    await hook()
    await hook()
    assert tracker.calls == []
    await hook()
    assert len(tracker.calls) == 1


@pytest.mark.asyncio
async def test_empty_storage_skips_without_error():
    """Empty storage + bounded sweep skips cleanly, no slice on empty list."""
    tracker = _RecordingTracker()
    storage = _StubStorage([])
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=storage, refresh_every=1, max_programs_per_sweep=5
    )

    await hook()

    assert tracker.calls == []


def test_plain_post_run_hook_rejected_at_init():
    """`pipeline=intra_extra_memory` without the writer on (`memory=full`) hands
    the hook a NullPostRunHook; that must fail at startup, not mid-run."""
    from gigaevo.evolution.engine.hooks import NullPostRunHook

    with pytest.raises(TypeError, match="memory=full"):
        LiveMemoryRefreshHook(
            tracker=NullPostRunHook(),  # type: ignore[arg-type]
            storage=_StubStorage([]),
            refresh_every=1,
        )


class _FlakyTracker(IncrementalPostRunHook):
    """Fails the first ``fail_times`` refreshes, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self._failures_left = fail_times
        self.successes = 0

    async def on_run_complete(self, storage) -> None:  # type: ignore[no-untyped-def]
        pass

    async def run_increment(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None = None,
    ) -> None:
        if self._failures_left > 0:
            self._failures_left -= 1
            raise RuntimeError("writer LLM unavailable")
        self.successes += 1


@pytest.fixture
def error_log():
    from loguru import logger

    messages: list[str] = []
    handle = logger.add(messages.append, level="ERROR", format="{message}")
    yield messages
    logger.remove(handle)


@pytest.mark.asyncio
async def test_refresh_failure_logs_error_and_reraises(five_programs, error_log):
    """A failed refresh must be loudly attributable in the run log (the
    engine's bounded wrapper swallows the traceback into its own channel)
    and still propagate so the engine keeps owning fault isolation."""
    tracker = _FlakyTracker(fail_times=1)
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=_StubStorage(five_programs), refresh_every=1
    )

    with pytest.raises(RuntimeError, match="writer LLM unavailable"):
        await hook()

    failures = [m for m in error_log if "[Memory][LiveRefresh] refresh FAILED" in m]
    assert len(failures) == 1
    assert "sweep" in failures[0]
    assert hook.consecutive_failures == 1


class _CancellingTracker(IncrementalPostRunHook):
    async def on_run_complete(self, storage) -> None:  # type: ignore[no-untyped-def]
        pass

    async def run_increment(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None = None,
    ) -> None:
        import asyncio

        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_cancelled_refresh_counts_failure_and_propagates(
    five_programs, error_log
):
    """CancelledError is a BaseException — the generic except arm misses it,
    so a cancelled sweep neither logs nor counts. It must do both AND still
    propagate so the engine's cancellation semantics stay intact."""
    import asyncio

    hook = LiveMemoryRefreshHook(
        tracker=_CancellingTracker(),
        storage=_StubStorage(five_programs),
        refresh_every=1,
    )

    with pytest.raises(asyncio.CancelledError):
        await hook()

    failures = [m for m in error_log if "[Memory][LiveRefresh] refresh FAILED" in m]
    assert len(failures) == 1
    assert hook.consecutive_failures == 1


@pytest.mark.asyncio
async def test_consecutive_failures_count_and_reset_on_success(
    five_programs, error_log
):
    tracker = _FlakyTracker(fail_times=2)
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=_StubStorage(five_programs), refresh_every=1
    )

    with pytest.raises(RuntimeError):
        await hook()
    with pytest.raises(RuntimeError):
        await hook()
    assert hook.consecutive_failures == 2

    await hook()
    assert hook.consecutive_failures == 0
    assert tracker.successes == 1
