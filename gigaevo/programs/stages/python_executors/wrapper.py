"""Loky-backed Python subprocess executor.

Workers spawn lazily via :func:`loky.get_reusable_executor`, are reused
across calls, and on Linux receive ``PR_SET_PDEATHSIG`` so they die with
the parent.  Each call's result is spilled to a file in
:attr:`ExecutorConfig.spill_dir` and the parent reads it via ``mmap`` —
result size is bounded by free space on the spill volume, not parent RAM.

User code is *not* sandboxed: env scrub and signal-handler reset are the
only guard rails.  Cloudpickle deserialisation in the parent can run
``__reduce__`` gadgets — trust user code as much as you trust whoever
generated it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass
import mmap
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
from typing import Any

import cloudpickle
from loguru import logger
from loky import get_reusable_executor

from gigaevo.programs.stages.python_executors.exec_runner import (
    WorkerCall,
    WorkerError,
    _run_one,
)


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Tunables for the loky-backed executor."""

    max_workers: int | None = None
    idle_timeout_s: int = 300
    spill_dir: Path = Path(tempfile.gettempdir()) / f"gigaevo-{os.getuid()}"

    @classmethod
    def from_env(cls) -> ExecutorConfig:
        """Construct from ``GIGAEVO_EXECUTOR_*`` env vars.

        Under pytest-xdist, ``max_workers`` auto-caps to
        ``cpu_count // PYTEST_XDIST_WORKER_COUNT`` to avoid an N×cpu_count
        fork-bomb; an explicit ``GIGAEVO_EXECUTOR_MAX_WORKERS`` overrides.
        """

        def _pos_int(key: str, default: int | None) -> int | None:
            raw = os.environ.get(key)
            if not raw:
                return default
            try:
                v = int(raw)
            except ValueError:
                return default
            return v if v > 0 else default

        spill = os.environ.get("GIGAEVO_EXECUTOR_SPILL_DIR")
        # ``resolve(strict=False)`` collapses ``..``/``.`` without requiring
        # the path to exist; the mkdir below creates it.
        spill_path = (
            Path(spill).resolve(strict=False)
            if spill
            else Path(tempfile.gettempdir()) / f"gigaevo-{os.getuid()}"
        )

        max_workers = _pos_int("GIGAEVO_EXECUTOR_MAX_WORKERS", None)
        if max_workers is None:
            xdist = _pos_int("PYTEST_XDIST_WORKER_COUNT", None)
            if xdist and xdist > 1:
                max_workers = max(1, (os.cpu_count() or 1) // xdist)

        return cls(
            max_workers=max_workers,
            idle_timeout_s=_pos_int("GIGAEVO_EXECUTOR_IDLE_TIMEOUT_S", 300) or 300,
            spill_dir=spill_path,
        )


_CONFIG: ExecutorConfig = ExecutorConfig.from_env()
# 0o700 is honoured only on creation; pre-existing dirs keep their mode.
_CONFIG.spill_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

# Loky reuses a pool only when kwargs match the cached call's kwargs.
# Rebuilding the env dict each submit (os.environ mutates over a process's
# life) would tear the pool down on every change and break in-flight tasks
# with BrokenProcessPool.  Capture once on first use.
_WORKER_ENV: dict[str, str] = {}


class ExecRunnerError(Exception):
    """User-code failure inside a worker.  ``stderr`` carries the traceback."""

    def __init__(self, *, returncode: int, stderr: str):
        tail = (stderr or "").rstrip() or "(no stderr)"
        super().__init__(f"exec_runner failed (exit={returncode}): {tail}")
        self.returncode = returncode
        self.stderr = stderr


# Worker os.environ is restricted to this whitelist (plus the GIGAEVO_* /
# LOKY_* prefixes).  Loky's env= kwarg is additive, not replace, so the
# actual scrub happens in _worker_init by deleting keys outside this set.
_ENV_WHITELIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_MESSAGES",
        "LC_NUMERIC",
        "LC_TIME",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "PYTHONHASHSEED",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PYENV_ROOT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "JAX_TRACEBACK_FILTERING",
        "TMPDIR",
        "TEMP",
        "TMP",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)


def _scrub_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in env.items()
        if k in _ENV_WHITELIST or k.startswith("GIGAEVO_") or k.startswith("LOKY_")
    }


def _worker_init() -> None:
    """Reset signals, scrub env, and request SIGTERM-on-parent-death (Linux)."""
    # SIGINT -> default_int_handler (raises KeyboardInterrupt, which
    # _run_one's BaseException catch turns into a structured error).
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    for sig_name in ("SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGCHLD"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(Exception):
            signal.signal(sig, signal.SIG_DFL)

    for key in list(os.environ.keys()):
        if key in _ENV_WHITELIST:
            continue
        if key.startswith("GIGAEVO_") or key.startswith("LOKY_"):
            continue
        os.environ.pop(key, None)

    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        rc = libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
        if rc != 0:
            logger.debug(
                "[exec_wrapper] PR_SET_PDEATHSIG failed rc={} errno={}",
                rc,
                ctypes.get_errno(),
            )
    except Exception as exc:
        logger.debug("[exec_wrapper] PR_SET_PDEATHSIG unavailable: {}", exc)


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Worker return envelope: either ``spill_path`` (success) or ``error``."""

    spill_path: str | None
    error: WorkerError | None
    peak_rss_kb: int
    wall_time_s: float
    user_time_s: float
    sys_time_s: float
    worker_pid: int


def _error_from_exc(exc: BaseException, prefix: str) -> WorkerError:
    buf = [prefix, "\n"]
    buf.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return WorkerError(stderr="".join(buf))


def _run_task(call: WorkerCall, spill_dir: str) -> WorkerResult:
    """Loky worker entry point.  Picklable; runs in the child process."""
    import resource as _resource

    t0 = time.monotonic()
    ru_before = _resource.getrusage(_resource.RUSAGE_SELF)
    result, error = _run_one(call)

    spill_path: str | None = None
    if error is None:
        # mkstemp: O_CREAT|O_EXCL (no symlink reuse) + mode 0600; matched
        # parent-side by O_NOFOLLOW in _load_spill.
        try:
            fd, spill_path = tempfile.mkstemp(
                prefix="gevo-result-", suffix=".pkl", dir=spill_dir
            )
        except OSError as exc:
            error = _error_from_exc(
                exc, f"Failed to create spill file in {spill_dir!r}:"
            )
        else:
            try:
                try:
                    f = os.fdopen(fd, "wb")
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    raise
                with f:
                    cloudpickle.dump(result, f, protocol=5)
            except BaseException as exc:
                with contextlib.suppress(OSError):
                    os.unlink(spill_path)
                spill_path = None
                error = _error_from_exc(
                    exc, "Failed to serialise result via cloudpickle:"
                )

    ru_after = _resource.getrusage(_resource.RUSAGE_SELF)
    return WorkerResult(
        spill_path=spill_path,
        error=error,
        peak_rss_kb=int(ru_after.ru_maxrss),
        wall_time_s=time.monotonic() - t0,
        user_time_s=float(ru_after.ru_utime - ru_before.ru_utime),
        sys_time_s=float(ru_after.ru_stime - ru_before.ru_stime),
        worker_pid=os.getpid(),
    )


def _get_executor() -> Any:
    if not _WORKER_ENV:
        _WORKER_ENV.update(_scrub_env(os.environ))
    return get_reusable_executor(
        max_workers=_CONFIG.max_workers,
        timeout=_CONFIG.idle_timeout_s,
        initializer=_worker_init,
        env=_WORKER_ENV,
        context="loky",
    )


def shutdown_executor(*, wait: bool = False) -> None:
    """Kill all loky workers if any were spawned.

    ``wait=True`` blocks until loky's manager thread has finished its
    shutdown sequence (incl. firing done-callbacks); use it from explicit
    teardown paths so spill-unlink callbacks fire before exit.  The
    ``wait=False`` default is for the timeout branch of
    :func:`run_exec_runner`, which must return promptly to propagate
    :class:`TimeoutError`.  Racy under concurrent submits by design.
    """
    _WORKER_ENV.clear()
    try:
        from loky import reusable_executor as _re
    except ImportError:
        return
    executor = getattr(_re, "_executor", None)
    if executor is None:
        return
    with contextlib.suppress(Exception):
        executor.shutdown(kill_workers=True, wait=wait)


def _load_spill(spill_path: str) -> Any:
    """Read a cloudpickle spill file via mmap.

    ``O_NOFOLLOW`` defends against a same-UID attacker swapping the path
    for a symlink between worker exit and parent read: cloudpickle.loads
    on attacker-controlled bytes is an RCE primitive in the parent.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(spill_path, flags)
    with os.fdopen(fd, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size == 0:
            return cloudpickle.loads(b"")
        try:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return cloudpickle.loads(mm)
        except (ValueError, OSError):
            # Some filesystems (rare tmpfs configs, NFS) reject mmap.
            f.seek(0)
            return cloudpickle.loads(f.read())


def _unlink_spill_on_done(fut: Any) -> None:
    try:
        result = fut.result()
    except BaseException:
        return
    spill = getattr(result, "spill_path", None)
    if spill:
        with contextlib.suppress(OSError):
            os.unlink(spill)


async def run_exec_runner(
    *,
    code: str,
    function_name: str,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    python_path: Sequence[Path] | None = None,
    env_updates: dict[str, Any] | None = None,
    timeout: int,
) -> Any:
    """Run a user function in a loky-managed subprocess and return its value.

    Raises :class:`ExecRunnerError` on user-code failure (``.stderr`` holds
    the traceback) or :class:`asyncio.TimeoutError` on wall-clock overrun.
    On timeout the worker pool is torn down; subsequent calls respawn it.
    """
    call = WorkerCall(
        code=code,
        function_name=function_name,
        args=list(args or []),
        kwargs=dict(kwargs or {}),
        python_path=[str(p) for p in (python_path or [])],
        env={k: (None if v is None else str(v)) for k, v in (env_updates or {}).items()},
    )

    executor = _get_executor()
    fut = executor.submit(_run_task, call, str(_CONFIG.spill_dir))

    try:
        result: WorkerResult = await asyncio.wait_for(
            asyncio.wrap_future(fut), timeout=timeout
        )
    except TimeoutError:
        # Loky has no public per-future kill; tearing the pool down is the
        # only available primitive.  Other in-flight tasks become
        # BrokenProcessPool and will be retried by their callers.
        shutdown_executor()
        raise
    except asyncio.CancelledError:
        # Don't tear the pool down; just unlink the spill if the worker
        # still produces one after cancellation.
        fut.add_done_callback(_unlink_spill_on_done)
        raise

    logger.trace(
        "[run_exec_runner] fn={} pid={} wall={:.3f}s rss={:.1f}MB user={:.3f}s sys={:.3f}s",
        function_name,
        result.worker_pid,
        result.wall_time_s,
        result.peak_rss_kb / 1024.0,
        result.user_time_s,
        result.sys_time_s,
    )

    if result.error is not None:
        raise ExecRunnerError(
            returncode=result.error.returncode,
            stderr=result.error.stderr,
        )

    assert result.spill_path is not None
    try:
        return _load_spill(result.spill_path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(result.spill_path)
