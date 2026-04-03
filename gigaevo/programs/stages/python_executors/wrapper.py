from __future__ import annotations

import asyncio
from collections.abc import Sequence
import functools
import os
from pathlib import Path
import struct
import sys
from typing import Any

import cloudpickle
import psutil


class ExecRunnerError(Exception):
    """Child process failed. Carries returncode and stderr text."""

    def __init__(self, *, returncode: int, stderr: str, stdout_bytes: bytes):
        super().__init__(f"exec_runner failed (exit={returncode})")
        self.returncode = returncode
        self.stderr = stderr
        self.stdout_bytes = stdout_bytes


def _find_runner_script() -> Path:
    """Path to exec_runner.py in this package (same directory as this module)."""
    return Path(__file__).resolve().parent / "exec_runner.py"


def _prepend_sys_path(paths: Sequence[Path | str] | None) -> None:
    if not paths:
        return
    for path in paths:
        candidate = str(path)
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """
    Best-effort termination of a subprocess and its process group.
    Safe to call multiple times.
    """
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except Exception:
        pass

    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe and hasattr(pipe, "close"):
            try:
                pipe.close()  # type: ignore[union-attr]
            except Exception:
                pass

    # Close the subprocess transport to prevent "Event loop is closed"
    # warnings from BaseSubprocessTransport.__del__ during GC.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


async def _monitor_rss_limit(
    proc: asyncio.subprocess.Process, *, max_bytes: int, interval_s: float = 0.2
) -> None:
    await asyncio.sleep(0.1)

    try:
        pproc = psutil.Process(proc.pid)
    except psutil.Error:
        return

    while proc.returncode is None:
        try:
            mem = pproc.memory_info().rss
        except psutil.NoSuchProcess:
            break
        except psutil.Error:
            # If psutil can't read stats, just stop monitoring
            break

        if mem > max_bytes:
            await _kill_process_tree(proc)
            break

        await asyncio.sleep(interval_s)


async def _start_worker_process(
    script: str,
    env: dict[str, str],
    cwd: str | None,
) -> asyncio.subprocess.Process:
    """Start exec_runner in persistent worker mode (--worker)."""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        script,
        "--worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )


_MAX_POOL_WORKERS = 32


class WorkerPool:
    """
    Pool of persistent exec_runner subprocesses so multiple executor stages can run in parallel.
    Pass to run_exec_runner(pool=...) or use the default from default_exec_runner_pool().
    """

    __slots__ = ("max_workers", "_queue", "_count", "_lock")

    def __init__(self, max_workers: int | None = None):
        if max_workers is None:
            n = (os.cpu_count() or 4) * 2
            max_workers = max(1, min(_MAX_POOL_WORKERS, n))
        self.max_workers = max_workers
        self._queue: asyncio.Queue[asyncio.subprocess.Process] = asyncio.Queue()
        self._count = 0
        self._lock = asyncio.Lock()

    async def get_worker(
        self,
        script: str,
        env: dict[str, str],
        cwd: str | None,
    ) -> asyncio.subprocess.Process:
        """Get an available worker, or create one if under limit, or wait for one to be returned."""
        async with self._lock:
            try:
                proc = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                proc = None
            if proc is None and self._count < self.max_workers:
                proc = await _start_worker_process(script, env, cwd)
                self._count += 1
        if proc is not None:
            return proc
        return await self._queue.get()

    async def return_worker(self, proc: asyncio.subprocess.Process) -> None:
        """Return a healthy worker to the pool; if already dead, decrement count and kill."""
        if proc.returncode is not None:
            async with self._lock:
                self._count -= 1
            await _kill_process_tree(proc)
            return
        self._queue.put_nowait(proc)

    async def discard_worker(self, proc: asyncio.subprocess.Process) -> None:
        """Remove a dead worker from the pool and kill it."""
        async with self._lock:
            self._count -= 1
        await _kill_process_tree(proc)

    async def shutdown(self) -> None:
        """Kill all idle workers in the pool.

        Call before the event loop closes to avoid
        'RuntimeError: Event loop is closed' warnings from
        subprocess transport ``__del__`` methods.
        """
        while not self._queue.empty():
            try:
                proc = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._count -= 1
            await _kill_process_tree(proc)


@functools.lru_cache(maxsize=1)
def default_exec_runner_pool() -> WorkerPool:
    """Return the default worker pool (one per process, created on first use)."""
    return WorkerPool()


async def _run_via_worker(
    proc: asyncio.subprocess.Process,
    data: bytes,
    timeout: int,
    max_memory_mb: int | None,
    max_output_size: int | None,
) -> tuple[Any, bytes, str]:
    """
    Send one length-prefixed payload to the worker and read length-prefixed response.
    Returns (value, raw_stdout_bytes, stderr_text). Raises ExecRunnerError on worker-reported error.
    """
    monitor_task: asyncio.Task | None = None
    if max_memory_mb is not None and proc.returncode is None:
        limit_bytes = max_memory_mb * 1024 * 1024
        monitor_task = asyncio.create_task(
            _monitor_rss_limit(proc, max_bytes=limit_bytes)
        )

    try:
        # Write length (4-byte big-endian) + payload
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(struct.pack(">I", len(data)) + data)
        await proc.stdin.drain()

        # Read response length then body
        len_buf = await asyncio.wait_for(proc.stdout.readexactly(4), timeout=timeout)
        (n,) = struct.unpack(">I", len_buf)
        if max_output_size is not None and n > max_output_size:
            raise ExecRunnerError(
                returncode=0,
                stderr=f"OutputTooLarge: {n} bytes exceeds limit of {max_output_size} bytes",
                stdout_bytes=b"",
            )
        stdout = await asyncio.wait_for(proc.stdout.readexactly(n), timeout=timeout)
    except (
        TimeoutError,
        asyncio.IncompleteReadError,
        BrokenPipeError,
        ConnectionResetError,
    ):
        if monitor_task is not None and not monitor_task.done():
            monitor_task.cancel()
        await _kill_process_tree(proc)
        raise
    finally:
        if monitor_task is not None and not monitor_task.done():
            monitor_task.cancel()

    value = cloudpickle.loads(stdout)
    if isinstance(value, dict) and value.get("_error"):
        stderr_text = value.get("stderr", "")
        returncode = value.get("returncode", 1)
        raise ExecRunnerError(
            returncode=returncode,
            stderr=stderr_text,
            stdout_bytes=b"",
        )
    return value, b"", ""


async def run_exec_runner(
    *,
    code: str,
    function_name: str,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    python_path: Sequence[Path] | None = None,
    env_updates: dict[str, Any] | None = None,
    timeout: int,
    max_memory_mb: int | None = None,
    max_output_size: int | None = None,
    cwd: Path | None = None,
    runner_path: Path | None = None,
    pool: WorkerPool | None = None,
) -> tuple[Any, bytes, str]:
    """
    Run user code in an isolated subprocess with resource limits.

    Args:
        code: Python code to execute
        function_name: Function to call in the code
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        python_path: Additional paths to add to sys.path
        timeout: Maximum execution time in seconds
        max_memory_mb: Maximum memory in MB (None = unlimited)
        max_output_size: Maximum output size in bytes (None = unlimited)
        cwd: Working directory for subprocess
        runner_path: Path to exec_runner.py script
        pool: Worker pool for parallel runs; if None, uses default_exec_runner_pool().

    Returns:
        (result_object, raw_stdout_bytes, stderr_text)
        Note: raw_stdout_bytes will be empty b"" on success to save memory,
        unless deserialization fails.

    Raises:
        ExecRunnerError: On non-zero exit, output too large, or execution failure
        asyncio.TimeoutError: On timeout
    """
    script = str(runner_path or _find_runner_script())
    cwd_str = str(cwd) if cwd else None
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if python_path:
        python_path_entries = [str(p) for p in python_path]
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            python_path_entries + ([existing_pythonpath] if existing_pythonpath else [])
        )

    payload = {
        "code": code,
        "function_name": function_name,
        "python_path": [str(p) for p in (python_path or [])],
        "args": list(args or []),
        "kwargs": dict(kwargs or {}),
        "env": dict(env_updates) if env_updates else {},
    }
    data = cloudpickle.dumps(payload, protocol=cloudpickle.DEFAULT_PROTOCOL)

    worker_pool = pool if pool is not None else default_exec_runner_pool()
    worker = await worker_pool.get_worker(script, env, cwd_str)
    returned = False
    try:
        result = await _run_via_worker(
            worker,
            data,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            max_output_size=max_output_size,
        )
        await worker_pool.return_worker(worker)
        returned = True
        return result
    except ExecRunnerError:
        await worker_pool.return_worker(worker)
        returned = True
        raise
    except (
        TimeoutError,
        asyncio.IncompleteReadError,
        BrokenPipeError,
        ConnectionResetError,
    ):
        await worker_pool.discard_worker(worker)
        returned = True
    finally:
        if not returned:
            # CancelledError or any other unexpected exception — discard the
            # worker so it doesn't leak and exhaust the pool.
            await worker_pool.discard_worker(worker)

    # One-shot: spawn subprocess for this call only.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd_str,
        env=env,
        start_new_session=True,
    )

    monitor_task: asyncio.Task | None = None
    if max_memory_mb is not None:
        limit_bytes = max_memory_mb * 1024 * 1024
        monitor_task = asyncio.create_task(
            _monitor_rss_limit(proc, max_bytes=limit_bytes)
        )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=data), timeout=timeout
        )
    except (TimeoutError, asyncio.CancelledError):
        if monitor_task is not None:
            monitor_task.cancel()
        await _kill_process_tree(proc)
        raise
    finally:
        if monitor_task is not None and not monitor_task.done():
            monitor_task.cancel()

    returncode = proc.returncode
    stderr_text = stderr.decode("utf-8", errors="replace")

    if returncode == 0:
        if max_output_size is not None and len(stdout) > max_output_size:
            raise ExecRunnerError(
                returncode=0,
                stderr=f"OutputTooLarge: Output {len(stdout)} bytes exceeds limit of {max_output_size} bytes\n[stderr]\n{stderr_text}",
                stdout_bytes=b"",
            )

        try:
            _prepend_sys_path(python_path)
            value = cloudpickle.loads(stdout)
            del stdout
        except Exception as e:
            raise ExecRunnerError(
                returncode=0,
                stderr=f"Invalid cloudpickle payload: {e}\n[stderr]\n{stderr_text}",
                stdout_bytes=stdout,
            )
        return value, b"", stderr_text

    raise ExecRunnerError(
        returncode=returncode or -1, stderr=stderr_text, stdout_bytes=stdout
    )
