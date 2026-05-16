"""Regression tests for ``default_exec_runner_pool``.

The factory builds a fresh ``WorkerPool`` per call. Callers that want to
amortize subprocess startup across many ``run_exec_runner`` invocations must
hold a single pool reference and pass it via ``pool=...``.

Each ``WorkerPool`` binds its ``asyncio.Queue`` and ``asyncio.Lock`` to the
event loop it is first used on, so an instance must not survive across
distinct ``asyncio.run()`` invocations.
"""

from __future__ import annotations

import asyncio

from gigaevo.programs.stages.python_executors.wrapper import (
    WorkerPool,
    default_exec_runner_pool,
)


def test_factory_returns_fresh_instance_per_call():
    """Two consecutive calls produce distinct ``WorkerPool`` objects."""
    a = default_exec_runner_pool()
    b = default_exec_runner_pool()
    assert isinstance(a, WorkerPool)
    assert isinstance(b, WorkerPool)
    assert a is not b


def test_factory_has_no_lru_cache_attribute():
    """The factory must not be wrapped by ``functools.lru_cache``.

    A wrapped function would expose ``cache_clear`` / ``cache_info``; the
    plain function does not. The check fails fast if the cache decorator
    re-appears.
    """
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
    """A pool built in one event loop is not reused in a second.

    Simulates the multirun pattern: a hypothetical caller that requests the
    default factory in one ``asyncio.run`` and again in another receives two
    pools with independent ``asyncio`` primitives bound to the respective
    loops.
    """
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
