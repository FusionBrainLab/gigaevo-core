"""Tests for gigaevo/utils/metrics_collector.py

Covers start_metrics_collector:
- sync collect_fn: numeric values written, non-numeric skipped
- async collect_fn: awaited correctly
- stop_flag: task terminates when flag goes True
- exceptions in collect_fn: swallowed, loop continues
- bool values: treated as numeric (cast to 1.0 / 0.0)
- empty metrics dict: writer.scalar not called
"""

from __future__ import annotations

import asyncio
from typing import Any

from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.trackers.base import LogWriter

# ---------------------------------------------------------------------------
# Minimal LogWriter implementation for tests
# ---------------------------------------------------------------------------


class RecordingWriter(LogWriter):
    """LogWriter that records every scalar() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def bind(self, path: list[str]) -> RecordingWriter:
        return self

    def scalar(self, metric: str, value: float, **kwargs: Any) -> None:
        self.calls.append((metric, value))

    def hist(self, metric: str, values: list[float], **kwargs: Any) -> None:
        pass

    def text(self, tag: str, text: str, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stop_after(n_calls: int):
    """Return a stop_flag that returns True after being called n_calls times."""
    counter = {"n": 0}

    def stop_flag() -> bool:
        counter["n"] += 1
        return counter["n"] > n_calls

    return stop_flag


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStartMetricsCollector:
    async def test_collects_numeric_metrics(self) -> None:
        """Sync collect_fn returning numeric values causes writer.scalar to be called."""
        writer = RecordingWriter()

        calls = {"n": 0}

        def collect_fn() -> dict[str, Any]:
            calls["n"] += 1
            return {"a": 1.0, "b": 2}

        stop_flag = _make_stop_after(1)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )

        await asyncio.wait_for(task, timeout=2.0)

        assert any(k == "a" for k, _ in writer.calls)
        assert any(k == "b" for k, _ in writer.calls)
        a_val = next(v for k, v in writer.calls if k == "a")
        b_val = next(v for k, v in writer.calls if k == "b")
        assert a_val == 1.0
        assert b_val == 2.0

    async def test_skips_non_numeric_values(self) -> None:
        """String values must NOT be passed to writer.scalar."""
        writer = RecordingWriter()

        def collect_fn() -> dict[str, Any]:
            return {"numeric": 42.0, "text": "hello", "also_text": ["list"]}

        stop_flag = _make_stop_after(1)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )
        await asyncio.wait_for(task, timeout=2.0)

        written_keys = {k for k, _ in writer.calls}
        assert "numeric" in written_keys
        assert "text" not in written_keys
        assert "also_text" not in written_keys

    async def test_async_collect_fn_is_awaited(self) -> None:
        """An async collect_fn is supported and its return value is used."""
        writer = RecordingWriter()

        async def collect_fn() -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"async_metric": 7.0}

        stop_flag = _make_stop_after(1)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )
        await asyncio.wait_for(task, timeout=2.0)

        written_keys = {k for k, _ in writer.calls}
        assert "async_metric" in written_keys
        assert next(v for k, v in writer.calls if k == "async_metric") == 7.0

    async def test_stops_immediately_when_flag_true(self) -> None:
        """Task completes quickly when stop_flag returns True on first check."""
        writer = RecordingWriter()

        def collect_fn() -> dict[str, Any]:
            return {"x": 1.0}

        # stop immediately on first check
        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=10.0,  # long interval — should not matter
            stop_flag=lambda: True,
        )
        # Task should finish within a short window since the loop exits immediately
        await asyncio.wait_for(task, timeout=1.0)

    async def test_exception_in_collect_fn_is_swallowed(self) -> None:
        """An exception raised by collect_fn must be swallowed; the loop continues."""
        writer = RecordingWriter()
        call_count = {"n": 0}

        def collect_fn() -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("collect failure")
            return {"ok": 99.0}

        # Stop after 2 iterations so both the error and the recovery are tested
        stop_flag = _make_stop_after(2)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )
        await asyncio.wait_for(task, timeout=3.0)

        # After the exception the loop must continue; "ok" must appear
        written_keys = {k for k, _ in writer.calls}
        assert "ok" in written_keys

    async def test_bool_values_collected_as_float(self) -> None:
        """bool is a subclass of int -> must be written via writer.scalar as 0.0 or 1.0."""
        writer = RecordingWriter()

        def collect_fn() -> dict[str, Any]:
            return {"flag_true": True, "flag_false": False}

        stop_flag = _make_stop_after(1)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )
        await asyncio.wait_for(task, timeout=2.0)

        written = dict(writer.calls)
        assert "flag_true" in written
        assert "flag_false" in written
        assert written["flag_true"] == 1.0
        assert written["flag_false"] == 0.0

    async def test_empty_metrics_dict_does_not_call_scalar(self) -> None:
        """If collect_fn returns an empty dict, writer.scalar must never be called."""
        writer = RecordingWriter()

        def collect_fn() -> dict[str, Any]:
            return {}

        stop_flag = _make_stop_after(2)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )
        await asyncio.wait_for(task, timeout=2.0)

        assert writer.calls == []

    async def test_none_metrics_dict_does_not_call_scalar(self) -> None:
        """If collect_fn returns None (falsy), writer.scalar must never be called."""
        writer = RecordingWriter()

        def collect_fn() -> dict[str, Any] | None:
            return None

        stop_flag = _make_stop_after(1)

        task = start_metrics_collector(
            writer=writer,
            collect_fn=collect_fn,
            interval=0.001,
            stop_flag=stop_flag,
        )
        await asyncio.wait_for(task, timeout=2.0)

        assert writer.calls == []

    async def test_task_name_is_set(self) -> None:
        """The returned asyncio.Task should carry the given task_name."""
        writer = RecordingWriter()

        task = start_metrics_collector(
            writer=writer,
            collect_fn=lambda: {},
            interval=0.001,
            stop_flag=lambda: True,
            task_name="my-collector",
        )
        assert task.get_name() == "my-collector"
        await asyncio.wait_for(task, timeout=1.0)
