"""Loky-backed Python subprocess executor.

Replaces the legacy hand-rolled WorkerPool + length-prefixed cloudpickle
protocol with :func:`loky.get_reusable_executor`.  Workers spawn lazily on
first :meth:`submit`, are cached across calls, and on Linux receive
``PR_SET_PDEATHSIG`` so they die with the parent.

Results are spilled to a per-call file under :data:`ExecutorConfig.spill_dir`
and the parent reads them back via :mod:`mmap`; result size is bounded by
free disk space rather than parent RAM.
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
    """Tunables for the loky-backed executor.

    The defaults match the historical hand-rolled WorkerPool: no
    preallocation (``max_workers=None`` → loky picks ``cpu_count()``),
    a generous idle timeout so DAGs with bursty traffic keep workers warm,
    and result spill to ``$TMPDIR`` (typically tmpfs).
    """

    max_workers: int | None = None
    idle_timeout_s: int = 300
    spill_dir: Path = Path("/tmp")

    @classmethod
    def from_env(cls) -> ExecutorConfig:
        """Construct from ``GIGAEVO_EXECUTOR_*`` environment variables."""
        return cls(
            max_workers=_env_int_or("GIGAEVO_EXECUTOR_MAX_WORKERS", None),
            idle_timeout_s=_env_int_or("GIGAEVO_EXECUTOR_IDLE_TIMEOUT_S", 300) or 300,
            spill_dir=_env_path_or(
                "GIGAEVO_EXECUTOR_SPILL_DIR", Path(tempfile.gettempdir())
            ),
        )


def _env_int_or(key: str, default: int | None) -> int | None:
    raw = os.environ.get(key)
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return v if v > 0 else default


def _env_path_or(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw) if raw else default


_CONFIG: ExecutorConfig = ExecutorConfig.from_env()
# Custom spill dirs may not exist yet; opt into create-on-import so
# callers don't have to remember to mkdir before the first submit.
_CONFIG.spill_dir.mkdir(parents=True, exist_ok=True)


class ExecRunnerError(Exception):
    """User-code failure inside a worker.  ``stderr`` carries the traceback."""

    def __init__(self, *, returncode: int, stderr: str):
        tail = (stderr or "").rstrip() or "(no stderr)"
        super().__init__(f"exec_runner failed (exit={returncode}): {tail}")
        self.returncode = returncode
        self.stderr = stderr


# User code is sandboxed; the worker's os.environ is restricted to this
# whitelist (plus the GIGAEVO_* / LOKY_* prefixes).  Keys outside the set
# — including any API tokens and secrets in the parent's env — are dropped.
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
    """Per-worker startup hook.

    * Resets the signal handlers loky installs so user code sees the
      default Python dispositions (KeyboardInterrupt on SIGINT, termination
      on SIGTERM/SIGHUP/SIGQUIT, ignore on SIGPIPE).
    * Scrubs ``os.environ`` down to the whitelist (loky's ``env=`` kwarg is
      additive, so non-whitelisted keys must be dropped explicitly here).
    * On Linux, asks the kernel to SIGTERM this worker if the parent dies
      (replaces the legacy orphan-reaper in ``tools/flush.py``).
    """
    # Restore default signal dispositions.  SIGINT goes to Python's
    # ``default_int_handler`` (which raises KeyboardInterrupt — preserved by
    # ``_run_one``'s ``except BaseException``); the rest go to OS defaults.
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    for sig_name in ("SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGCHLD"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(Exception):
            signal.signal(sig, signal.SIG_DFL)
    # SIGPIPE: leave Python's default (SIG_IGN; broken-pipe raises
    # BrokenPipeError instead of killing the worker on stdout closure).

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

        _PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Compact return value from :func:`_run_task`.

    On success ``spill_path`` references a cloudpickle file on disk that
    the parent reads and unlinks.  On failure ``error`` holds the
    structured error (and ``spill_path`` is ``None``).  Resource
    accounting is always populated.
    """

    spill_path: str | None
    error: WorkerError | None
    peak_rss_kb: int
    wall_time_s: float
    user_time_s: float
    sys_time_s: float
    worker_pid: int


def _run_task(call: WorkerCall, spill_dir: str) -> WorkerResult:
    """Loky worker entry point.  Picklable; runs in the child process."""
    import resource as _resource

    t0 = time.monotonic()
    ru_before = _resource.getrusage(_resource.RUSAGE_SELF)
    result, error = _run_one(call)

    spill_path: str | None = None
    if error is None:
        try:
            fd, spill_path = tempfile.mkstemp(
                prefix="gevo-result-", suffix=".pkl", dir=spill_dir
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    cloudpickle.dump(result, f, protocol=5)
            except BaseException as exc:
                with contextlib.suppress(OSError):
                    os.unlink(spill_path)
                spill_path = None
                error = _error_from_exc(
                    exc, "Failed to serialise result via cloudpickle:"
                )
        except OSError as exc:
            error = _error_from_exc(
                exc, f"Failed to create spill file in {spill_dir!r}:"
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


def _error_from_exc(exc: BaseException, prefix: str) -> WorkerError:
    buf = [prefix, "\n"]
    buf.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return WorkerError(stderr="".join(buf))


def _get_executor() -> Any:
    return get_reusable_executor(
        max_workers=_CONFIG.max_workers,
        timeout=_CONFIG.idle_timeout_s,
        initializer=_worker_init,
        env=_scrub_env(os.environ),
        context="loky",
    )


def shutdown_executor() -> None:
    """Kill all loky workers if any were spawned.

    Reaches into loky's module-global cache so we don't pay the spawn cost
    of constructing a pool just to tear it down.  Safe to call repeatedly.
    """
    try:
        from loky import reusable_executor as _re
    except ImportError:
        return
    executor = getattr(_re, "_executor", None)
    if executor is None:
        return
    with contextlib.suppress(Exception):
        executor.shutdown(kill_workers=True, wait=False)


def _load_spill(spill_path: str) -> Any:
    """Read a cloudpickle spill file, preferring mmap to skip a bytes copy."""
    size = os.path.getsize(spill_path)
    with open(spill_path, "rb") as f:
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
    """Done-callback used on cancellation: unlink the spill if produced."""
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
        # Loky has no public per-future kill; tearing the pool down is
        # the only available primitive.  Other in-flight tasks become
        # BrokenProcessPool and will be retried by their callers.
        shutdown_executor()
        raise
    except asyncio.CancelledError:
        # Don't tear the pool down — that punishes concurrent tasks.
        # Register a done-callback to unlink the spill if the worker
        # still produces one.
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
