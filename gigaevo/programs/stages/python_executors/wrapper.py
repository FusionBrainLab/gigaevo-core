"""Loky-backed Python subprocess executor.

Replaces the legacy hand-rolled WorkerPool + length-prefixed cloudpickle
protocol with :func:`loky.get_reusable_executor`.  Workers spawn lazily on
first :meth:`submit`, are cached across calls, and on Linux receive
``PR_SET_PDEATHSIG`` so they die with the parent.

Results are spilled to a per-call file under :data:`SPILL_DIR` and the
parent reads them back via :mod:`mmap`; result size is bounded by free
disk space rather than parent RAM.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import io
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


MAX_WORKERS: int | None = _env_int_or("GIGAEVO_EXECUTOR_MAX_WORKERS", None)
IDLE_TIMEOUT_S: int = _env_int_or("GIGAEVO_EXECUTOR_IDLE_TIMEOUT_S", 300) or 300
SPILL_DIR: Path = _env_path_or(
    "GIGAEVO_EXECUTOR_SPILL_DIR", Path(tempfile.gettempdir())
)
# Custom spill dirs may not exist yet; opt into create-on-import so callers
# don't have to remember to mkdir before the first submit.
SPILL_DIR.mkdir(parents=True, exist_ok=True)


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


def _scrub_env(env: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in env.items()
        if k in _ENV_WHITELIST or k.startswith("GIGAEVO_") or k.startswith("LOKY_")
    }


def _worker_init() -> None:
    """Linux only: have the kernel SIGTERM this worker if the parent dies."""
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
class _ResultEnvelope:
    """Compact return value from :func:`_run_task`.

    On success ``spill_path`` references a cloudpickle file on disk that
    the parent reads and unlinks.  On failure ``error`` holds the
    structured error dict (and ``spill_path`` is ``None``).  Resource
    accounting is always populated.
    """

    spill_path: str | None
    error: dict[str, Any] | None
    peak_rss_kb: int
    wall_time_s: float
    user_time_s: float
    sys_time_s: float
    worker_pid: int


def _error_envelope_from_exc(exc: BaseException, prefix: str) -> dict[str, Any]:
    buf = io.StringIO()
    buf.write(prefix)
    buf.write("\n")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=buf)
    return {"_error": True, "stderr": buf.getvalue(), "returncode": 1}


def _run_task(payload: dict[str, Any], spill_dir: str) -> _ResultEnvelope:
    """Loky worker entry point.  Picklable; runs in the child process."""
    import resource as _resource

    from gigaevo.programs.stages.python_executors.exec_runner import _run_one

    t0 = time.monotonic()
    ru_before = _resource.getrusage(_resource.RUSAGE_SELF)
    result, error = _run_one(payload)

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
                error = _error_envelope_from_exc(
                    exc, "Failed to serialise result via cloudpickle:"
                )
        except OSError as exc:
            error = _error_envelope_from_exc(
                exc, f"Failed to create spill file in {spill_dir!r}:"
            )

    ru_after = _resource.getrusage(_resource.RUSAGE_SELF)
    return _ResultEnvelope(
        spill_path=spill_path,
        error=error,
        peak_rss_kb=int(ru_after.ru_maxrss),
        wall_time_s=time.monotonic() - t0,
        user_time_s=float(ru_after.ru_utime - ru_before.ru_utime),
        sys_time_s=float(ru_after.ru_stime - ru_before.ru_stime),
        worker_pid=os.getpid(),
    )


def _get_executor() -> Any:
    return get_reusable_executor(
        max_workers=MAX_WORKERS,
        timeout=IDLE_TIMEOUT_S,
        initializer=_worker_init,
        env=_scrub_env(dict(os.environ)),
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
        envelope = fut.result()
    except BaseException:
        return
    spill = getattr(envelope, "spill_path", None)
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
    payload: dict[str, Any] = {
        "code": code,
        "function_name": function_name,
        "python_path": [str(p) for p in (python_path or [])],
        "args": list(args or []),
        "kwargs": dict(kwargs or {}),
        "env": dict(env_updates) if env_updates else {},
    }

    executor = _get_executor()
    fut = executor.submit(_run_task, payload, str(SPILL_DIR))

    try:
        envelope: _ResultEnvelope = await asyncio.wait_for(
            asyncio.wrap_future(fut), timeout=timeout
        )
    except TimeoutError:
        # Loky has no public per-future kill; tearing the pool down is the
        # only available primitive.  Other in-flight tasks become
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
        envelope.worker_pid,
        envelope.wall_time_s,
        envelope.peak_rss_kb / 1024.0,
        envelope.user_time_s,
        envelope.sys_time_s,
    )

    if envelope.error is not None:
        raise ExecRunnerError(
            returncode=int(envelope.error.get("returncode", 1)),
            stderr=str(envelope.error.get("stderr", "")),
        )

    assert envelope.spill_path is not None
    try:
        return _load_spill(envelope.spill_path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(envelope.spill_path)
