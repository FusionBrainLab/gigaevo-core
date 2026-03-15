"""WorkerPool adversarial tests — H2 and H3 from chaos-hacker round 5.

H2: _count must NOT be incremented when _start_worker_process raises.
    If it were, the pool would appear full and all subsequent calls would
    fall through to the one-shot fallback indefinitely.

H3: return_worker must treat returncode=0 (clean exit) as dead — the
    process must be discarded, not re-queued.  A re-queued dead worker
    would cause the next caller to receive EOF / BrokenPipe.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.programs.stages.python_executors.wrapper import WorkerPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alive_proc(pid: int = 99) -> MagicMock:
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.pid = pid
    proc.returncode = None  # still running
    proc.kill.return_value = None
    proc.wait = AsyncMock(return_value=0)
    for attr in ("stdin", "stdout", "stderr"):
        pipe = MagicMock()
        pipe.close.return_value = None
        setattr(proc, attr, pipe)
    proc._transport = MagicMock()
    return proc


def _make_dead_proc(returncode: int = 0, pid: int = 99) -> MagicMock:
    proc = _make_alive_proc(pid=pid)
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# H2 — _count does NOT leak when _start_worker_process raises
# ---------------------------------------------------------------------------


class TestWorkerPoolCountLeak:
    """H2 regression: spawn failure must not permanently inflate _count.

    get_worker() increments _count AFTER _start_worker_process() returns,
    so a subprocess-creation error propagates without touching the counter.
    These tests document and guard that invariant.
    """

    async def test_count_unchanged_when_spawn_raises_oserror(self):
        """OSError during spawn → _count stays 0, error propagates."""
        pool = WorkerPool(max_workers=2)
        assert pool._count == 0

        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._start_worker_process",
            new=AsyncMock(side_effect=OSError("binary not found")),
        ):
            with pytest.raises(OSError):
                await pool.get_worker("script.py", {}, None)

        assert pool._count == 0, (
            "_count was incremented before spawn succeeded — pool will appear "
            "permanently full after a failed spawn"
        )

    async def test_count_unchanged_when_spawn_raises_runtime_error(self):
        """RuntimeError during spawn → _count stays 0."""
        pool = WorkerPool(max_workers=2)

        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._start_worker_process",
            new=AsyncMock(side_effect=RuntimeError("event loop closed")),
        ):
            with pytest.raises(RuntimeError):
                await pool.get_worker("script.py", {}, None)

        assert pool._count == 0

    async def test_count_incremented_on_successful_spawn(self):
        """Successful spawn → _count becomes 1 (normal path, no leak)."""
        pool = WorkerPool(max_workers=2)
        alive = _make_alive_proc()

        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._start_worker_process",
            new=AsyncMock(return_value=alive),
        ):
            worker = await pool.get_worker("script.py", {}, None)

        assert pool._count == 1
        assert worker is alive

    async def test_pool_not_stuck_after_failed_then_successful_spawn(self):
        """After a failed spawn, a subsequent call can still create a worker."""
        pool = WorkerPool(max_workers=2)
        alive = _make_alive_proc()

        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._start_worker_process",
        ) as mock_spawn:
            mock_spawn.side_effect = [OSError("fail"), alive]
            with pytest.raises(OSError):
                await pool.get_worker("script.py", {}, None)

            # Second call must succeed — pool is not stuck at max_workers
            worker = await pool.get_worker("script.py", {}, None)

        assert pool._count == 1
        assert worker is alive


# ---------------------------------------------------------------------------
# H3 — return_worker discards workers with returncode=0 (clean exit = dead)
# ---------------------------------------------------------------------------


class TestReturnWorkerHealthCheck:
    """H3 regression: returncode=0 must be treated as dead (not re-queued).

    The health check uses `proc.returncode is not None`, which correctly
    identifies both clean exits (0) and crash exits (non-zero) as dead.
    These tests guard that a cleanly-exited worker is never re-queued.
    """

    async def test_returncode_zero_worker_discarded_not_requeued(self):
        """returncode=0 → worker discarded (count decremented, not re-queued)."""
        pool = WorkerPool(max_workers=2)
        pool._count = 1  # pretend one active worker

        dead = _make_dead_proc(returncode=0)
        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
            new=AsyncMock(),
        ):
            await pool.return_worker(dead)

        assert pool._count == 0, "count should be decremented for dead worker"
        assert pool._queue.empty(), "dead worker (returncode=0) must NOT be re-queued"

    async def test_returncode_nonzero_worker_discarded(self):
        """returncode=1 (crash) → worker discarded, count decremented."""
        pool = WorkerPool(max_workers=2)
        pool._count = 1

        crashed = _make_dead_proc(returncode=1)
        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
            new=AsyncMock(),
        ):
            await pool.return_worker(crashed)

        assert pool._count == 0
        assert pool._queue.empty()

    async def test_returncode_none_worker_requeued(self):
        """returncode=None (still alive) → worker returned to pool queue."""
        pool = WorkerPool(max_workers=2)
        pool._count = 1

        alive = _make_alive_proc()
        await pool.return_worker(alive)

        assert pool._count == 1, "count must stay 1 for a live returned worker"
        assert not pool._queue.empty(), "live worker must be re-queued for reuse"

    async def test_requeued_worker_is_the_same_object(self):
        """After return_worker, get_worker returns the exact same process object."""
        pool = WorkerPool(max_workers=2)
        pool._count = 1

        alive = _make_alive_proc()
        await pool.return_worker(alive)

        # The already-queued worker should be returned immediately (no spawn needed)
        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._start_worker_process",
            new=AsyncMock(side_effect=AssertionError("spawn should not be called")),
        ):
            worker = await pool.get_worker("script.py", {}, None)

        assert worker is alive
