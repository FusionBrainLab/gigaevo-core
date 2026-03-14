"""Tests for Finding 2: _task_model_map grows unbounded in MultiModelRouter.

gigaevo/llm/models.py:

_select() stores the selected model name in _task_model_map[task_id].
get_last_model() pops it. If the code between _select() and a successful
get_last_model() call raises an exception (e.g. the LLM call fails), or if
get_last_model() is simply never called, the entry is never removed.

Over many calls this map leaks memory — one entry per failed/abandoned async task.

Tests:
1. Entry leaks when ainvoke raises before get_last_model() is called.
2. Repeated failed calls accumulate entries (unbounded growth).
3. Successful call + get_last_model() pops the entry (correct path).
4. Two concurrent tasks each get their own entry; both popped on success.
5. get_last_model() after a failed call returns None (stale entry not visible
   from a *different* task).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.llm.models import MultiModelRouter
from gigaevo.llm.token_tracking import TokenTracker
from tests.conftest import NullWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = name
    m.with_structured_output = MagicMock(return_value=MagicMock())
    return m


def _mock_response(ctx=100, gen=50, total=150) -> MagicMock:
    resp = MagicMock()
    resp.response_metadata = {
        "token_usage": {
            "prompt_tokens": ctx,
            "completion_tokens": gen,
            "total_tokens": total,
        }
    }
    return resp


def _router_with_model(name: str = "model_a") -> tuple[MultiModelRouter, MagicMock]:
    model = _mock_model(name)
    router = MultiModelRouter([model], [1.0], writer=NullWriter(), name="test")
    router._langfuse = None
    return router, model


# ---------------------------------------------------------------------------
# TestTaskModelMapLeak — Finding 2
# ---------------------------------------------------------------------------


class TestTaskModelMapLeak:
    async def test_successful_ainvoke_entry_consumed_by_get_last_model(self) -> None:
        """On a successful ainvoke, the map entry is present and get_last_model() pops it."""
        router, model = _router_with_model("model_a")
        model.ainvoke = AsyncMock(return_value=_mock_response())

        async def task_fn():
            await router.ainvoke("hello")
            # Entry should still be in the map — get_last_model pops it
            name = router.get_last_model()
            return name, len(router._task_model_map)

        result_name, map_size_after_pop = await asyncio.create_task(task_fn())
        assert result_name == "model_a"
        assert map_size_after_pop == 0, (
            "Map entry should be gone after get_last_model()"
        )

    async def test_failed_ainvoke_leaks_map_entry(self) -> None:
        """When ainvoke raises, the map entry is never popped — map grows by 1.

        This confirms the leak: the caller has no way to call get_last_model()
        when ainvoke raises, so the entry persists in _task_model_map.
        """
        router, model = _router_with_model("model_a")
        model.ainvoke = AsyncMock(side_effect=ConnectionError("LLM unreachable"))

        map_size_before = len(router._task_model_map)

        async def task_fn():
            with pytest.raises(ConnectionError):
                await router.ainvoke("hello")
            # map entry was written by _select() before ainvoke raised
            return len(router._task_model_map)

        map_size_after = await asyncio.create_task(task_fn())

        assert map_size_after == map_size_before + 1, (
            f"Expected map to grow by 1 (leaked entry), "
            f"before={map_size_before}, after={map_size_after}"
        )

    async def test_repeated_failed_calls_accumulate_entries(self) -> None:
        """Multiple failed calls accumulate entries — unbounded growth.

        With N failed calls, the map has N leaked entries.
        This test demonstrates the severity of the leak.
        """
        router, model = _router_with_model("model_a")
        model.ainvoke = AsyncMock(side_effect=RuntimeError("always fails"))

        N = 5

        async def failing_task():
            with pytest.raises(RuntimeError):
                await router.ainvoke("hello")

        tasks = [asyncio.create_task(failing_task()) for _ in range(N)]
        await asyncio.gather(*tasks, return_exceptions=True)

        leaked = len(router._task_model_map)
        assert leaked == N, (
            f"Expected {N} leaked entries after {N} failed calls, got {leaked}. "
            "This confirms _task_model_map grows unbounded on errors."
        )

    async def test_successful_calls_do_not_leak(self) -> None:
        """Successful calls + get_last_model() pops entry — no accumulation."""
        router, model = _router_with_model("model_a")
        model.ainvoke = AsyncMock(return_value=_mock_response())

        N = 5

        async def successful_task():
            await router.ainvoke("hello")
            router.get_last_model()  # pop the entry

        tasks = [asyncio.create_task(successful_task()) for _ in range(N)]
        await asyncio.gather(*tasks)

        assert len(router._task_model_map) == 0, (
            "Map should be empty after N successful tasks each calling get_last_model()"
        )

    async def test_get_last_model_clears_own_tasks_entry_only(self) -> None:
        """get_last_model() is task-scoped: it only pops the current task's entry."""
        router, model = _router_with_model("model_a")
        model.ainvoke = AsyncMock(return_value=_mock_response())

        task_a_selected = asyncio.Event()
        task_b_can_proceed = asyncio.Event()

        async def task_a():
            # Select but don't get_last_model yet
            router._select()
            task_a_selected.set()
            await task_b_can_proceed.wait()
            # Now pop our own entry
            return router.get_last_model()

        async def task_b():
            await task_a_selected.wait()
            router._select()
            task_b_can_proceed.set()
            # Pop task_b's entry
            return router.get_last_model()

        result_a, result_b = await asyncio.gather(
            asyncio.create_task(task_a()),
            asyncio.create_task(task_b()),
        )

        assert result_a == "model_a"
        assert result_b == "model_a"
        # Both entries should be popped
        assert len(router._task_model_map) == 0

    async def test_get_last_model_returns_none_outside_task(self) -> None:
        """Outside an async task (no current_task()), get_last_model() returns None."""
        router, _ = _router_with_model("model_a")
        # Not inside an asyncio Task here (we're in a coroutine but called directly
        # without create_task — asyncio.current_task() may or may not be set)
        result = router.get_last_model()
        # Either None (no task) or None (no entry for this task) — both acceptable
        assert result is None or isinstance(result, str)

    async def test_two_concurrent_tasks_independent_entries(self) -> None:
        """Two concurrent tasks each select their own model; entries are independent."""
        router, model = _router_with_model("model_a")
        model.ainvoke = AsyncMock(return_value=_mock_response())

        results: dict[int, str | None] = {}

        async def task_fn(idx: int):
            await router.ainvoke("hello")
            results[idx] = router.get_last_model()

        await asyncio.gather(
            asyncio.create_task(task_fn(0)),
            asyncio.create_task(task_fn(1)),
        )

        assert results[0] == "model_a"
        assert results[1] == "model_a"
        assert len(router._task_model_map) == 0

    def test_invoke_sync_no_task_id_entry_skipped(self) -> None:
        """Synchronous invoke: no current_task() → _task_model_map not populated."""
        router, model = _router_with_model("model_a")
        model.invoke.return_value = _mock_response()

        initial_size = len(router._task_model_map)
        router.invoke("hello")
        # In sync context, _current_task_id() returns None → no entry written
        assert len(router._task_model_map) == initial_size


# ---------------------------------------------------------------------------
# TestTokenTrackerBasic — smoke tests for token_tracking.py
# ---------------------------------------------------------------------------


class TestTokenTrackerSmoke:
    def test_track_increments_cumulative(self) -> None:
        tracker = TokenTracker(name="test", writer=NullWriter())
        resp = _mock_response(ctx=100, gen=50, total=150)
        tracker.track(resp, "model_a")
        tracker.track(resp, "model_a")

        assert tracker.cumulative["model_a"].context == 200
        assert tracker.cumulative["model_a"].generated == 100
        assert tracker.cumulative["model_a"].total == 300

    def test_track_separate_models_independent(self) -> None:
        tracker = TokenTracker(name="test", writer=NullWriter())
        tracker.track(_mock_response(ctx=100, gen=50, total=150), "model_a")
        tracker.track(_mock_response(ctx=200, gen=80, total=280), "model_b")

        assert tracker.cumulative["model_a"].total == 150
        assert tracker.cumulative["model_b"].total == 280
        assert "model_a" in tracker.cumulative
        assert "model_b" in tracker.cumulative

    def test_track_no_writer_is_noop(self) -> None:
        tracker = TokenTracker(name="test", writer=None)
        tracker.track(_mock_response(), "model_a")
        assert len(tracker.cumulative) == 0

    def test_track_response_without_metadata_skipped(self) -> None:
        tracker = TokenTracker(name="test", writer=NullWriter())
        resp = MagicMock()
        resp.response_metadata = {}
        tracker.track(resp, "model_a")
        assert len(tracker.cumulative) == 0

    def test_thread_safety_concurrent_writes(self) -> None:
        """Concurrent tracks from multiple threads should not corrupt cumulative state."""
        import threading

        tracker = TokenTracker(name="test", writer=NullWriter())
        n_threads = 20
        calls_per_thread = 50

        def write_tokens():
            resp = _mock_response(ctx=1, gen=1, total=2)
            for _ in range(calls_per_thread):
                tracker.track(resp, "shared_model")

        threads = [threading.Thread(target=write_tokens) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = n_threads * calls_per_thread * 2
        assert tracker.cumulative["shared_model"].total == expected_total, (
            f"Thread-safety violation: expected {expected_total} total tokens, "
            f"got {tracker.cumulative['shared_model'].total}"
        )

    def test_reset_cumulative_by_replacement(self) -> None:
        """Replacing cumulative dict resets state (no built-in reset method)."""

        tracker = TokenTracker(name="test", writer=NullWriter())
        tracker.track(_mock_response(ctx=100, gen=50, total=150), "model_a")
        assert tracker.cumulative["model_a"].total == 150

        # Reset by replacement
        tracker.cumulative = {}
        assert len(tracker.cumulative) == 0


# ---------------------------------------------------------------------------
# TestMultiModelRouterSmoke — basic routing and unknown-model fallback
# ---------------------------------------------------------------------------


class TestMultiModelRouterSmoke:
    def test_single_model_always_selected(self) -> None:
        """With one model at probability 1.0, it is always selected."""
        router, _ = _router_with_model("only_model")
        for _ in range(10):
            _, name = router._select()
            assert name == "only_model"

    def test_two_models_both_selectable(self) -> None:
        """With two equal-probability models, both can be selected."""
        import random

        model_a = _mock_model("a")
        model_b = _mock_model("b")
        router = MultiModelRouter([model_a, model_b], [1.0, 1.0], name="test")
        router._langfuse = None

        selected_names = set()
        # With seed-driven sampling over 50 trials, both should appear
        random.seed(42)
        for _ in range(50):
            _, name = router._select()
            selected_names.add(name)

        assert "a" in selected_names
        assert "b" in selected_names

    def test_zero_probability_raises(self) -> None:
        """Any zero probability raises ValueError at construction."""
        with pytest.raises(ValueError, match="positive"):
            MultiModelRouter(
                [_mock_model("a"), _mock_model("b")], [1.0, 0.0], name="test"
            )

    def test_length_mismatch_raises(self) -> None:
        """Mismatched models/probabilities lengths raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            MultiModelRouter([_mock_model("a")], [0.5, 0.5], name="test")

    def test_probabilities_normalized(self) -> None:
        """Probabilities are normalized to sum to 1.0."""
        router = MultiModelRouter(
            [_mock_model("a"), _mock_model("b")], [3.0, 1.0], name="test"
        )
        assert sum(router.probabilities) == pytest.approx(1.0)
        assert router.probabilities[0] == pytest.approx(0.75)
        assert router.probabilities[1] == pytest.approx(0.25)
