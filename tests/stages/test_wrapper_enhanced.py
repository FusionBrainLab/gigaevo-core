"""Enhanced tests for gigaevo/programs/stages/python_executors/wrapper.py.

Covers low-level process management functions and WorkerPool edge-cases
that the existing test_python_executors.py does not exercise:
  - _kill_process_tree: killpg success, killpg fallback to proc.kill, all-failures
  - _monitor_rss_limit: over-limit kills, process done exits, psutil.NoSuchProcess
  - WorkerPool.return_worker / discard_worker: dead vs alive workers
  - WorkerPool saturation: blocks when full, shutdown kills idle
  - ExecRunnerError.__str__: includes returncode
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import psutil

from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    WorkerPool,
    _kill_process_tree,
    _monitor_rss_limit,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight mock subprocess process
# ---------------------------------------------------------------------------


def _make_mock_proc(
    *,
    pid: int = 12345,
    returncode: int | None = None,
    killpg_raises: bool = False,
    kill_raises: bool = False,
) -> MagicMock:
    """Build a mock asyncio.subprocess.Process with controllable behaviour."""
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.pid = pid
    proc.returncode = returncode

    # proc.kill() — synchronous
    if kill_raises:
        proc.kill.side_effect = ProcessLookupError("already dead")
    else:
        proc.kill.return_value = None

    # proc.wait() — async
    wait_future: asyncio.Future = asyncio.get_event_loop().create_future()
    wait_future.set_result(0)
    proc.wait.return_value = wait_future

    # stdin / stdout / stderr — mock pipes with close()
    for attr in ("stdin", "stdout", "stderr"):
        pipe = MagicMock()
        pipe.close.return_value = None
        setattr(proc, attr, pipe)

    # _transport for the GC-safety close
    proc._transport = MagicMock()
    proc._transport.close.return_value = None

    return proc


# ---------------------------------------------------------------------------
# TestKillProcessTree
# ---------------------------------------------------------------------------


class TestKillProcessTree:
    async def test_killpg_succeeds_and_pipes_closed(self) -> None:
        """When os.killpg succeeds, pipes and transport are still closed."""
        proc = _make_mock_proc()
        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.os.killpg"
            ) as mock_killpg,
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.os.getpgid",
                return_value=100,
            ),
        ):
            await _kill_process_tree(proc)
            mock_killpg.assert_called_once_with(100, 9)

        # Pipes should be closed
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()
        # Transport should be closed
        proc._transport.close.assert_called_once()

    async def test_killpg_fails_falls_back_to_proc_kill(self) -> None:
        """When os.killpg raises, we fall back to proc.kill()."""
        proc = _make_mock_proc()
        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.os.killpg",
                side_effect=ProcessLookupError("no such group"),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.os.getpgid",
                return_value=100,
            ),
        ):
            await _kill_process_tree(proc)

        proc.kill.assert_called_once()

    async def test_both_killpg_and_proc_kill_fail(self) -> None:
        """When both os.killpg and proc.kill() fail, no exception propagates."""
        proc = _make_mock_proc(kill_raises=True)
        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.os.killpg",
                side_effect=OSError("killpg failed"),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.os.getpgid",
                return_value=100,
            ),
        ):
            # Should not raise
            await _kill_process_tree(proc)

        # Pipes should still be cleaned up despite kill failures
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()


# ---------------------------------------------------------------------------
# TestMonitorRssLimit
# ---------------------------------------------------------------------------


class TestMonitorRssLimit:
    async def test_over_limit_kills_process(self) -> None:
        """When RSS exceeds max_bytes, the process tree is killed."""
        proc = _make_mock_proc()
        # returncode is None initially, then set to -9 after kill
        returncode_values = [None, None, -9]
        type(proc).returncode = property(
            lambda self: returncode_values.pop(0) if returncode_values else -9
        )

        mem_info = SimpleNamespace(rss=200 * 1024 * 1024)  # 200 MB
        mock_pproc = MagicMock()
        mock_pproc.memory_info.return_value = mem_info

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.psutil.Process",
                return_value=mock_pproc,
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            await _monitor_rss_limit(proc, max_bytes=100 * 1024 * 1024, interval_s=0.01)
            mock_kill.assert_awaited_once_with(proc)

    async def test_process_already_done_exits_loop(self) -> None:
        """When proc.returncode is not None, the monitor exits immediately."""
        proc = _make_mock_proc(returncode=0)

        mock_pproc = MagicMock()
        mock_pproc.memory_info.return_value = SimpleNamespace(rss=50)

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.psutil.Process",
                return_value=mock_pproc,
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            await _monitor_rss_limit(proc, max_bytes=100, interval_s=0.01)
            # Process already finished — should not kill
            mock_kill.assert_not_awaited()

    async def test_psutil_no_such_process_breaks_loop(self) -> None:
        """When psutil raises NoSuchProcess, the monitor stops gracefully."""
        proc = _make_mock_proc()
        # returncode stays None so the while loop would normally continue
        type(proc).returncode = property(lambda self: None)

        mock_pproc = MagicMock()
        mock_pproc.memory_info.side_effect = psutil.NoSuchProcess(pid=12345)

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.psutil.Process",
                return_value=mock_pproc,
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            await _monitor_rss_limit(proc, max_bytes=100, interval_s=0.01)
            # Should break, not kill
            mock_kill.assert_not_awaited()

    async def test_psutil_error_on_creation_returns(self) -> None:
        """When psutil.Process() raises psutil.Error, monitor returns immediately."""
        proc = _make_mock_proc()

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.psutil.Process",
                side_effect=psutil.Error("process gone"),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            await _monitor_rss_limit(proc, max_bytes=100, interval_s=0.01)
            mock_kill.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestWorkerPoolReturnDiscard
# ---------------------------------------------------------------------------


class TestWorkerPoolReturnDiscard:
    async def test_dead_worker_decrements_count(self) -> None:
        """return_worker with a dead proc (returncode != None) decrements _count."""
        pool = WorkerPool(max_workers=4)
        pool._count = 2

        proc = _make_mock_proc(returncode=1)
        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
            new_callable=AsyncMock,
        ) as mock_kill:
            await pool.return_worker(proc)
            mock_kill.assert_awaited_once_with(proc)

        assert pool._count == 1

    async def test_alive_worker_returned_to_queue(self) -> None:
        """return_worker with alive proc (returncode is None) puts it back in queue."""
        pool = WorkerPool(max_workers=4)
        pool._count = 1

        proc = _make_mock_proc(returncode=None)
        await pool.return_worker(proc)

        assert pool._queue.qsize() == 1
        assert pool._count == 1  # count unchanged

        # The returned proc should be the same one we put in
        got = pool._queue.get_nowait()
        assert got is proc

    async def test_discard_worker_kills_and_decrements(self) -> None:
        """discard_worker always kills the process and decrements _count."""
        pool = WorkerPool(max_workers=4)
        pool._count = 3

        proc = _make_mock_proc(returncode=None)
        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
            new_callable=AsyncMock,
        ) as mock_kill:
            await pool.discard_worker(proc)
            mock_kill.assert_awaited_once_with(proc)

        assert pool._count == 2


# ---------------------------------------------------------------------------
# TestWorkerPoolSaturation
# ---------------------------------------------------------------------------


class TestWorkerPoolSaturation:
    async def test_blocks_when_saturated_until_worker_returned(self) -> None:
        """get_worker blocks when pool is at max and queue is empty, unblocks on return."""
        pool = WorkerPool(max_workers=1)
        pool._count = 1  # Already at max

        # get_worker should block (no idle workers, at max count).
        # We return a worker after a short delay to unblock.
        proc = _make_mock_proc(returncode=None)

        async def _return_after_delay():
            await asyncio.sleep(0.05)
            await pool.return_worker(proc)

        asyncio.create_task(_return_after_delay())

        worker = await asyncio.wait_for(
            pool.get_worker("script.py", {}, None),
            timeout=2.0,
        )
        assert worker is proc

    async def test_shutdown_kills_idle_workers(self) -> None:
        """shutdown() kills all idle workers sitting in the queue."""
        pool = WorkerPool(max_workers=4)

        proc1 = _make_mock_proc(returncode=None)
        proc2 = _make_mock_proc(returncode=None)

        # Manually place workers in the idle queue
        pool._queue.put_nowait(proc1)
        pool._queue.put_nowait(proc2)
        pool._count = 2

        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
            new_callable=AsyncMock,
        ) as mock_kill:
            await pool.shutdown()
            assert mock_kill.await_count == 2

        assert pool._count == 0
        assert pool._queue.empty()


# ---------------------------------------------------------------------------
# TestExecRunnerError
# ---------------------------------------------------------------------------


class TestExecRunnerError:
    def test_str_includes_returncode(self) -> None:
        """String representation of ExecRunnerError includes the returncode."""
        err = ExecRunnerError(
            returncode=42, stderr="something went wrong", stdout_bytes=b""
        )
        assert "42" in str(err)
        assert "exec_runner failed" in str(err)

    def test_attributes_accessible(self) -> None:
        """All constructor attributes are accessible on the instance."""
        err = ExecRunnerError(returncode=7, stderr="err text", stdout_bytes=b"raw")
        assert err.returncode == 7
        assert err.stderr == "err text"
        assert err.stdout_bytes == b"raw"


# ---------------------------------------------------------------------------
# TestWorkerPoolCountUnderflow
# ---------------------------------------------------------------------------


class TestWorkerPoolCountUnderflow:
    async def test_discard_at_count_one_goes_to_zero(self) -> None:
        """discard_worker when _count is 1 should reduce it to 0."""
        pool = WorkerPool(max_workers=4)
        pool._count = 1

        proc = _make_mock_proc(returncode=None)
        with patch(
            "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
            new_callable=AsyncMock,
        ) as mock_kill:
            await pool.discard_worker(proc)
            mock_kill.assert_awaited_once()

        assert pool._count == 0


# ---------------------------------------------------------------------------
# TestMonitorRssMultiPoll
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestRunViaWorkerOutputSizeLimit (P1)
# ---------------------------------------------------------------------------


class TestRunViaWorkerOutputSizeLimit:
    """wrapper.py L223: After reading the 4-byte length prefix, if the
    declared response size exceeds max_output_size, raise ExecRunnerError
    before reading the body."""

    async def test_output_too_large_raises_before_reading_body(self) -> None:
        """When the response length prefix exceeds max_output_size, raise immediately."""
        import struct

        import pytest

        from gigaevo.programs.stages.python_executors.wrapper import _run_via_worker

        proc = _make_mock_proc(returncode=None)

        # Simulate worker writing a length prefix of 10MB
        big_length = 10 * 1024 * 1024
        len_bytes = struct.pack(">I", big_length)

        # Mock stdout to return the length prefix
        stdout_read = AsyncMock(side_effect=[len_bytes])
        proc.stdout.readexactly = stdout_read

        # Mock stdin
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()

        with pytest.raises(ExecRunnerError) as exc_info:
            await _run_via_worker(
                proc,
                data=b"test",
                timeout=10,
                max_memory_mb=None,
                max_output_size=1024,  # 1KB limit
            )

        # Check the stderr contains OutputTooLarge
        assert "OutputTooLarge" in exc_info.value.stderr

    async def test_output_within_limit_succeeds(self) -> None:
        """When response size is within max_output_size, read the body normally."""
        import struct

        import cloudpickle

        from gigaevo.programs.stages.python_executors.wrapper import _run_via_worker

        proc = _make_mock_proc(returncode=None)

        # Simulate a small response
        payload = cloudpickle.dumps({"result": 42})
        len_bytes = struct.pack(">I", len(payload))

        read_calls = [len_bytes, payload]
        proc.stdout.readexactly = AsyncMock(side_effect=read_calls)
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()

        value, stdout_bytes, stderr_text = await _run_via_worker(
            proc,
            data=b"test",
            timeout=10,
            max_memory_mb=None,
            max_output_size=1024 * 1024,  # 1MB limit — plenty
        )

        assert value == {"result": 42}
        assert stdout_bytes == b""


# ---------------------------------------------------------------------------
# TestRunExecRunnerFallback (P0)
# ---------------------------------------------------------------------------


def _make_mock_pool(worker_proc: MagicMock | None = None) -> MagicMock:
    """Create a mock WorkerPool that returns a controllable worker process.
    No real subprocesses are spawned."""
    pool = MagicMock()
    proc = worker_proc or _make_mock_proc(returncode=None)
    pool.get_worker = AsyncMock(return_value=proc)
    pool.return_worker = AsyncMock()
    pool.discard_worker = AsyncMock()
    pool._worker_proc = proc  # stash for assertion access
    return pool


class TestRunExecRunnerFallback:
    """wrapper.py L333-399: When the worker pool path fails (timeout, broken pipe),
    run_exec_runner falls back to a one-shot subprocess. This entire fallback
    path previously had zero tests.

    All tests use a fully mocked WorkerPool — no real subprocesses are spawned."""

    async def test_worker_timeout_triggers_oneshot_fallback(self) -> None:
        """When worker path raises TimeoutError, fallback to one-shot subprocess."""
        import cloudpickle

        from gigaevo.programs.stages.python_executors.wrapper import run_exec_runner

        pool = _make_mock_pool()

        result_payload = cloudpickle.dumps({"answer": 99})
        mock_oneshot = AsyncMock()
        mock_oneshot.communicate = AsyncMock(return_value=(result_payload, b""))
        mock_oneshot.returncode = 0

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                new_callable=AsyncMock,
                side_effect=TimeoutError("worker timed out"),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.asyncio.create_subprocess_exec",
                return_value=mock_oneshot,
            ),
        ):
            value, stdout, stderr = await run_exec_runner(
                code="def solve(): return 99",
                function_name="solve",
                timeout=10,
                pool=pool,
            )

        assert value == {"answer": 99}
        # Worker should have been discarded (not returned)
        pool.discard_worker.assert_awaited_once()

    async def test_oneshot_nonzero_exit_raises(self) -> None:
        """One-shot fallback with non-zero exit code raises ExecRunnerError."""
        import pytest

        from gigaevo.programs.stages.python_executors.wrapper import run_exec_runner

        pool = _make_mock_pool()

        mock_oneshot = AsyncMock()
        mock_oneshot.communicate = AsyncMock(return_value=(b"", b"Traceback: error\n"))
        mock_oneshot.returncode = 1

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                new_callable=AsyncMock,
                side_effect=BrokenPipeError("worker died"),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.asyncio.create_subprocess_exec",
                return_value=mock_oneshot,
            ),
        ):
            with pytest.raises(ExecRunnerError, match="exec_runner failed"):
                await run_exec_runner(
                    code="def solve(): return 1",
                    function_name="solve",
                    timeout=10,
                    pool=pool,
                )

    async def test_oneshot_output_too_large_raises(self) -> None:
        """One-shot fallback enforces max_output_size on stdout."""
        import pytest

        from gigaevo.programs.stages.python_executors.wrapper import run_exec_runner

        pool = _make_mock_pool()

        big_output = b"x" * 10000
        mock_oneshot = AsyncMock()
        mock_oneshot.communicate = AsyncMock(return_value=(big_output, b""))
        mock_oneshot.returncode = 0

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                new_callable=AsyncMock,
                side_effect=ConnectionResetError("worker lost"),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.asyncio.create_subprocess_exec",
                return_value=mock_oneshot,
            ),
        ):
            with pytest.raises(ExecRunnerError) as exc_info:
                await run_exec_runner(
                    code="def solve(): return 1",
                    function_name="solve",
                    timeout=10,
                    max_output_size=100,
                    pool=pool,
                )
            assert "OutputTooLarge" in exc_info.value.stderr

    async def test_oneshot_cloudpickle_failure_raises(self) -> None:
        """One-shot fallback: cloudpickle.loads() fails -> ExecRunnerError."""
        import pytest

        from gigaevo.programs.stages.python_executors.wrapper import run_exec_runner

        pool = _make_mock_pool()

        mock_oneshot = AsyncMock()
        mock_oneshot.communicate = AsyncMock(return_value=(b"not-valid-pickle", b""))
        mock_oneshot.returncode = 0

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._run_via_worker",
                new_callable=AsyncMock,
                side_effect=asyncio.IncompleteReadError(b"partial", 100),
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.asyncio.create_subprocess_exec",
                return_value=mock_oneshot,
            ),
        ):
            with pytest.raises(ExecRunnerError) as exc_info:
                await run_exec_runner(
                    code="def solve(): return 1",
                    function_name="solve",
                    timeout=10,
                    pool=pool,
                )
            assert "Invalid cloudpickle" in exc_info.value.stderr


class TestMonitorRssMultiPoll:
    async def test_rss_below_limit_loops_until_done(self) -> None:
        """When RSS stays below limit, monitor loops until process finishes."""
        proc = _make_mock_proc()
        # Process finishes after 3 checks
        returncode_values = [None, None, None, 0]
        type(proc).returncode = property(
            lambda self: returncode_values.pop(0) if returncode_values else 0
        )

        mem_info = SimpleNamespace(rss=10 * 1024 * 1024)  # 10 MB, below limit
        mock_pproc = MagicMock()
        mock_pproc.memory_info.return_value = mem_info

        with (
            patch(
                "gigaevo.programs.stages.python_executors.wrapper.psutil.Process",
                return_value=mock_pproc,
            ),
            patch(
                "gigaevo.programs.stages.python_executors.wrapper._kill_process_tree",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            await _monitor_rss_limit(proc, max_bytes=100 * 1024 * 1024, interval_s=0.01)
            # Should NOT kill since RSS is below limit
            mock_kill.assert_not_awaited()
            # memory_info should have been called at least once
            assert mock_pproc.memory_info.call_count >= 1
