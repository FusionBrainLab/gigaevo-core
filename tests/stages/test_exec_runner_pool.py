"""Tests for ``default_exec_runner_pool`` and the ambient-pool ContextVar.

The factory builds a fresh ``WorkerPool`` per call; to amortize
subprocess startup, callers pass ``pool=...`` or bind one via
``set_ambient_exec_runner_pool`` for the lifetime of an ``asyncio.run``.
Each ``WorkerPool`` binds its primitives to the loop it is first used on.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from gigaevo.programs.stages.python_executors.wrapper import (
    WorkerPool,
    default_exec_runner_pool,
    get_ambient_exec_runner_pool,
    reset_ambient_exec_runner_pool,
    run_exec_runner,
    set_ambient_exec_runner_pool,
)


def test_factory_returns_fresh_instance_per_call():
    """Two consecutive calls produce distinct ``WorkerPool`` objects."""
    a = default_exec_runner_pool()
    b = default_exec_runner_pool()
    assert isinstance(a, WorkerPool)
    assert isinstance(b, WorkerPool)
    assert a is not b


def test_factory_has_no_lru_cache_attribute():
    """The factory is not wrapped by ``functools.lru_cache``."""
    assert not hasattr(default_exec_runner_pool, "cache_clear")
    assert not hasattr(default_exec_runner_pool, "cache_info")


def test_each_pool_owns_its_asyncio_primitives():
    """Each pool gets its own queue/lock; no shared state survives across pools."""
    a = default_exec_runner_pool()
    b = default_exec_runner_pool()
    assert a._queue is not b._queue
    assert a._lock is not b._lock
    assert a._count == 0
    assert b._count == 0


def test_fresh_pool_is_safe_across_sequential_event_loops():
    """Distinct ``asyncio.run`` calls each receive a fresh pool with
    primitives bound to the local loop."""
    loop_a_pool: WorkerPool | None = None

    async def in_loop_a() -> WorkerPool:
        pool = default_exec_runner_pool()
        # Touch the lock so it binds to the current loop.
        async with pool._lock:
            pass
        return pool

    loop_a_pool = asyncio.run(in_loop_a())

    async def in_loop_b() -> WorkerPool:
        pool = default_exec_runner_pool()
        async with pool._lock:
            pass
        return pool

    loop_b_pool = asyncio.run(in_loop_b())
    assert loop_b_pool is not loop_a_pool
    assert loop_b_pool._lock is not loop_a_pool._lock
    assert loop_b_pool._queue is not loop_a_pool._queue


def test_ambient_pool_defaults_to_none():
    """No ambient pool is bound at import time."""
    assert get_ambient_exec_runner_pool() is None


def test_ambient_pool_round_trip():
    """``set`` then ``reset`` restores the previous binding."""
    pool = WorkerPool()
    token = set_ambient_exec_runner_pool(pool)
    try:
        assert get_ambient_exec_runner_pool() is pool
    finally:
        reset_ambient_exec_runner_pool(token)
    assert get_ambient_exec_runner_pool() is None


def test_run_exec_runner_reuses_ambient_pool_when_pool_is_none():
    """``run_exec_runner(pool=None)`` resolves to the bound ambient pool
    so two successive calls share one pool object."""
    seen_pools: list[WorkerPool] = []

    class _FakeProc:
        returncode = None

    async def fake_get_worker(self, script, env, cwd):
        seen_pools.append(self)
        return _FakeProc()

    async def fake_return_worker(self, proc):
        return None

    async def fake_run_via_worker(proc, data, timeout, max_memory_mb, max_output_size):
        return ("ok", b"", "")

    async def scenario() -> WorkerPool:
        ambient = WorkerPool()
        token = set_ambient_exec_runner_pool(ambient)
        try:
            with (
                patch.object(WorkerPool, "get_worker", fake_get_worker),
                patch.object(WorkerPool, "return_worker", fake_return_worker),
                patch(
                    "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                    fake_run_via_worker,
                ),
            ):
                await run_exec_runner(
                    code="def f():\n    return 1\n",
                    function_name="f",
                    timeout=5,
                )
                await run_exec_runner(
                    code="def f():\n    return 1\n",
                    function_name="f",
                    timeout=5,
                )
            return ambient
        finally:
            reset_ambient_exec_runner_pool(token)

    ambient = asyncio.run(scenario())
    assert len(seen_pools) == 2
    assert seen_pools[0] is ambient
    assert seen_pools[1] is ambient
    assert seen_pools[0] is seen_pools[1]


def test_explicit_pool_argument_overrides_ambient():
    """An explicit ``pool=`` kwarg wins over a bound ambient pool."""
    seen_pools: list[WorkerPool] = []

    class _FakeProc:
        returncode = None

    async def fake_get_worker(self, script, env, cwd):
        seen_pools.append(self)
        return _FakeProc()

    async def fake_return_worker(self, proc):
        return None

    async def fake_run_via_worker(proc, data, timeout, max_memory_mb, max_output_size):
        return ("ok", b"", "")

    async def scenario() -> tuple[WorkerPool, WorkerPool]:
        ambient = WorkerPool()
        explicit = WorkerPool()
        token = set_ambient_exec_runner_pool(ambient)
        try:
            with (
                patch.object(WorkerPool, "get_worker", fake_get_worker),
                patch.object(WorkerPool, "return_worker", fake_return_worker),
                patch(
                    "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                    fake_run_via_worker,
                ),
            ):
                await run_exec_runner(
                    code="def f():\n    return 1\n",
                    function_name="f",
                    timeout=5,
                    pool=explicit,
                )
            return ambient, explicit
        finally:
            reset_ambient_exec_runner_pool(token)

    ambient, explicit = asyncio.run(scenario())
    assert len(seen_pools) == 1
    assert seen_pools[0] is explicit
    assert seen_pools[0] is not ambient


def test_run_exec_runner_falls_back_to_default_when_no_ambient():
    """Without ambient binding, ``pool=None`` allocates a fresh per-call pool."""
    seen_pools: list[WorkerPool] = []

    class _FakeProc:
        returncode = None

    async def fake_get_worker(self, script, env, cwd):
        seen_pools.append(self)
        return _FakeProc()

    async def fake_return_worker(self, proc):
        return None

    async def fake_run_via_worker(proc, data, timeout, max_memory_mb, max_output_size):
        return ("ok", b"", "")

    async def scenario() -> None:
        assert get_ambient_exec_runner_pool() is None
        with (
            patch.object(WorkerPool, "get_worker", fake_get_worker),
            patch.object(WorkerPool, "return_worker", fake_return_worker),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                fake_run_via_worker,
            ),
        ):
            await run_exec_runner(
                code="def f():\n    return 1\n",
                function_name="f",
                timeout=5,
            )
            await run_exec_runner(
                code="def f():\n    return 1\n",
                function_name="f",
                timeout=5,
            )

    asyncio.run(scenario())
    assert len(seen_pools) == 2
    assert seen_pools[0] is not seen_pools[1]


@pytest.mark.asyncio
async def test_no_pool_none_call_sites_remain_unguarded():
    """Audit: production callers in stages must not rebuild a pool per call.

    The four production sites pass ``pool=None`` so they pick up the ambient
    pool. This test scans those modules' source for any explicit
    ``default_exec_runner_pool()`` invocation as a regression guard against
    future code paths bypassing the experiment-scoped lifecycle.
    """
    import inspect

    from gigaevo.programs.stages import runtime_metrics
    from gigaevo.programs.stages.optimization import utils as opt_utils
    from gigaevo.programs.stages.optimization.optuna import stage as optuna_stage
    from gigaevo.programs.stages.python_executors import execution

    for module in (runtime_metrics, opt_utils, optuna_stage, execution):
        source = inspect.getsource(module)
        assert "default_exec_runner_pool(" not in source, (
            f"{module.__name__} constructs an ad-hoc pool. Use the ambient "
            f"pool bound by run_experiment instead."
        )
