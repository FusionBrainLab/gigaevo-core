"""Tests for the crash-event primitives."""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.dataplane.crash import CrashEvent, CrashWatchedHandle, OneShotFlag

# ── OneShotFlag ───────────────────────────────────────────────────────


class TestOneShotFlag:
    def test_unset_by_default(self) -> None:
        f = OneShotFlag()
        assert not f.is_set()

    def test_signal_sets(self) -> None:
        f = OneShotFlag()
        f.signal()
        assert f.is_set()

    def test_signal_idempotent(self) -> None:
        f = OneShotFlag()
        f.signal()
        f.signal()
        assert f.is_set()

    @pytest.mark.asyncio
    async def test_wait_returns_when_signalled(self) -> None:
        f = OneShotFlag()

        async def waiter() -> str:
            await f.wait()
            return "released"

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)  # let waiter park
        assert not task.done()
        f.signal()
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == "released"


# ── CrashWatchedHandle ────────────────────────────────────────────────


class _FakeResource:
    """Trivial inner resource for handle tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: int = 0

    async def do_work(self) -> str:
        self.calls += 1
        return f"work-{self.name}-{self.calls}"


class TestCrashWatchedHandle:
    @pytest.mark.asyncio
    async def test_normal_path_returns_value(self) -> None:
        inner = _FakeResource("a")
        flag = OneShotFlag()

        async def recover(_old: _FakeResource) -> CrashEvent[str, _FakeResource]:
            return CrashEvent(peer="dead", resource=_FakeResource("recovered"))

        handle: CrashWatchedHandle[_FakeResource, str, _FakeResource] = (
            CrashWatchedHandle(inner, flag, recover)
        )
        value, evt = await handle.call(lambda r: r.do_work())
        assert evt is None
        assert value == "work-a-1"

    @pytest.mark.asyncio
    async def test_signalled_flag_returns_crash_event(self) -> None:
        inner = _FakeResource("a")
        flag = OneShotFlag()
        flag.signal()
        recovered = _FakeResource("recovered")

        async def recover(_old: _FakeResource) -> CrashEvent[str, _FakeResource]:
            return CrashEvent(peer="peer-1", resource=recovered)

        handle: CrashWatchedHandle[_FakeResource, str, _FakeResource] = (
            CrashWatchedHandle(inner, flag, recover)
        )
        value, evt = await handle.call(lambda r: r.do_work())
        assert value is None
        assert evt is not None
        assert evt.peer == "peer-1"
        assert evt.resource is recovered
        # Inner was not invoked.
        assert inner.calls == 0

    @pytest.mark.asyncio
    async def test_replace_inner_after_recovery(self) -> None:
        original = _FakeResource("original")
        new_resource = _FakeResource("new")
        flag = OneShotFlag()

        async def recover(_old: _FakeResource) -> CrashEvent[str, _FakeResource]:
            return CrashEvent(peer="x", resource=new_resource)

        handle: CrashWatchedHandle[_FakeResource, str, _FakeResource] = (
            CrashWatchedHandle(original, flag, recover)
        )
        # Signal, observe recovery, then swap in the new resource with a fresh flag.
        flag.signal()
        _, evt = await handle.call(lambda r: r.do_work())
        assert evt is not None
        new_flag = OneShotFlag()
        handle.replace_inner(evt.resource, new_flag)
        # Subsequent calls now hit the recovered resource and don't observe the old flag.
        value, second_evt = await handle.call(lambda r: r.do_work())
        assert second_evt is None
        assert value == "work-new-1"
