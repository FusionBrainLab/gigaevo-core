"""Loky-backed Python subprocess executor.

Workers spawn lazily inside :class:`LokyBackend`, are reused across calls,
and on Linux receive ``PR_SET_PDEATHSIG`` so they die with the parent.
Each call's result is spilled to a file in :attr:`WorkerConfig.spill_dir`
and the parent reads it via ``mmap`` — result size is bounded by free
space on the spill volume, not parent RAM.

User code is *not* sandboxed: env scrub and signal-handler reset are the
only guard rails.  Cloudpickle deserialisation in the parent can run
``__reduce__`` gadgets — trust user code as much as you trust whoever
generated it.

Multiple :class:`LokyBackend` instances can coexist with isolated pools,
spill directories and env scrub rules; pool-level state is no longer
module-global.  The module-level :func:`run_exec_runner` and
:func:`shutdown_executor` are thin wrappers over a process-scoped
default singleton for backward compatibility with existing call sites.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time
import traceback
from typing import Any
import uuid

import cloudpickle
from loguru import logger
import loky
from loky.backend.context import get_context

from gigaevo.programs.stages.python_executors.exec_runner import (
    WorkerCall,
    WorkerError,
    _run_one,
)

# ---------------------------------------------------------------------------
# Defaults for env scrubbing
# ---------------------------------------------------------------------------

DEFAULT_ENV_WHITELIST: frozenset[str] = frozenset(
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

DEFAULT_ENV_PREFIXES: tuple[str, ...] = ("GIGAEVO_", "LOKY_")

_WORKER_INIT_ENV_KEY = "_GIGAEVO_WORKER_INIT_CONFIG"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Per-pool configuration for a :class:`LokyBackend`.

    All fields are immutable after construction.  Two backends with
    different ``pool_id`` get isolated spill directories and run in
    independent loky pools — they do not share state, do not share the
    underlying executor, and can be shut down independently.
    """

    max_workers: int | None = None
    idle_timeout_s: int = 300
    spill_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / f"gigaevo-{os.getuid()}"
    )
    env_whitelist: frozenset[str] = DEFAULT_ENV_WHITELIST
    env_prefixes: tuple[str, ...] = DEFAULT_ENV_PREFIXES
    # Forward-compatible field; not enforced today.  Wired up in a later
    # PR alongside per-worker call counters from the upcoming
    # ExecutorBackend distributed implementations.
    max_calls_before_recycle: int | None = None
    enable_pdeathsig: bool = True
    pool_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    node_id: str = field(default_factory=socket.gethostname)

    @classmethod
    def from_env(cls) -> WorkerConfig:
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


class ExecRunnerError(Exception):
    """User-code failure inside a worker.  ``stderr`` carries the traceback."""

    def __init__(self, *, returncode: int, stderr: str):
        tail = (stderr or "").rstrip() or "(no stderr)"
        super().__init__(f"exec_runner failed (exit={returncode}): {tail}")
        self.returncode = returncode
        self.stderr = stderr


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
    # Stable identifiers attached to every result so multi-pool and
    # multi-host deployments can attribute log events back to source.
    # ``worker_id`` is generated once per worker process in :func:`_worker_init`;
    # ``node_id`` is taken from :attr:`WorkerConfig.node_id`.
    worker_id: str = ""
    node_id: str = ""


# ---------------------------------------------------------------------------
# Module-level helpers (used by both backend methods and worker init)
# ---------------------------------------------------------------------------


def _scrub_env(
    env: Mapping[str, str],
    whitelist: frozenset[str],
    prefixes: tuple[str, ...],
) -> dict[str, str]:
    """Filter env to whitelisted keys + prefixed keys."""
    return {
        k: v
        for k, v in env.items()
        if k in whitelist or any(k.startswith(p) for p in prefixes)
    }


def _error_from_exc(exc: BaseException, prefix: str) -> WorkerError:
    buf = [prefix, "\n"]
    buf.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return WorkerError(stderr="".join(buf))


def _worker_init() -> None:
    """Reset signals, scrub env, request SIGTERM-on-parent-death (Linux).

    Runs once inside each loky worker on spawn.  Reads scrub rules from
    the env variable :data:`_WORKER_INIT_ENV_KEY` (set by the parent
    :class:`LokyBackend` when constructing the worker env); falls back to
    module defaults if absent so direct loky usage outside this module
    still gets sensible behavior.
    """
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

    raw = os.environ.get(_WORKER_INIT_ENV_KEY)
    node_id = ""
    if raw:
        try:
            decoded = json.loads(base64.b64decode(raw).decode("utf-8"))
            whitelist = frozenset(decoded["whitelist"])
            prefixes = tuple(decoded["prefixes"])
            enable_pdeathsig = bool(decoded["pdeathsig"])
            node_id = str(decoded.get("node_id", ""))
        except Exception as exc:
            logger.debug(
                "[exec_wrapper] worker init config decode failed, using defaults: {}",
                exc,
            )
            whitelist = DEFAULT_ENV_WHITELIST
            prefixes = DEFAULT_ENV_PREFIXES
            enable_pdeathsig = True
    else:
        whitelist = DEFAULT_ENV_WHITELIST
        prefixes = DEFAULT_ENV_PREFIXES
        enable_pdeathsig = True

    for key in list(os.environ.keys()):
        if key in whitelist or any(key.startswith(p) for p in prefixes):
            continue
        os.environ.pop(key, None)

    # Worker identity: unique uuid generated once per worker process, plus
    # the parent's node_id.  Surfaced via os.environ (GIGAEVO_* prefix
    # passes the scrub above) so user code can read them too if it wants.
    worker_id = uuid.uuid4().hex[:12]
    os.environ["GIGAEVO_WORKER_ID"] = worker_id
    if node_id:
        os.environ["GIGAEVO_NODE_ID"] = node_id

    if not enable_pdeathsig:
        return
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
        worker_id=os.environ.get("GIGAEVO_WORKER_ID", ""),
        node_id=os.environ.get("GIGAEVO_NODE_ID", ""),
    )


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


# ---------------------------------------------------------------------------
# LokyBackend — class-encapsulated pool
# ---------------------------------------------------------------------------


class LokyBackend:
    """Local-subprocess executor backed by loky.

    Encapsulates the loky pool instance, the scrubbed env, and the
    per-pool spill directory.  Multiple instances can coexist with
    isolated state.  Lazily creates the underlying pool on first
    :meth:`execute` call; idempotent shutdown.
    """

    def __init__(self, config: WorkerConfig | None = None) -> None:
        self._config = config if config is not None else WorkerConfig.from_env()
        self._executor: loky.ProcessPoolExecutor | None = None
        self._worker_env: dict[str, str] = {}

    @property
    def config(self) -> WorkerConfig:
        return self._config

    def _build_worker_env(self) -> dict[str, str]:
        """Build the env dict passed to loky workers.

        Includes the scrubbed parent env plus an encoded init-config blob
        in :data:`_WORKER_INIT_ENV_KEY` so :func:`_worker_init` knows
        which scrub rules to apply on the worker side (loky's ``env=``
        kwarg is additive, not replace).
        """
        scrubbed = _scrub_env(
            os.environ, self._config.env_whitelist, self._config.env_prefixes
        )
        init_config = {
            "whitelist": list(self._config.env_whitelist),
            "prefixes": list(self._config.env_prefixes),
            "pdeathsig": self._config.enable_pdeathsig,
            "node_id": self._config.node_id,
        }
        scrubbed[_WORKER_INIT_ENV_KEY] = base64.b64encode(
            json.dumps(init_config).encode("utf-8")
        ).decode("ascii")
        return scrubbed

    def _get_executor(self) -> loky.ProcessPoolExecutor:
        if self._executor is None:
            # 0o700 is honoured only on creation; pre-existing dirs keep their mode.
            self._config.spill_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._worker_env = self._build_worker_env()
            self._executor = loky.ProcessPoolExecutor(
                max_workers=self._config.max_workers,
                timeout=self._config.idle_timeout_s,
                initializer=_worker_init,
                env=self._worker_env,
                context=get_context("loky"),
            )
        return self._executor

    async def execute(self, call: WorkerCall, *, deadline_s: int) -> Any:
        """Run *call* in a worker; return its value or raise.

        Raises :class:`ExecRunnerError` on user-code failure (``.stderr``
        carries the traceback) or :class:`asyncio.TimeoutError` on
        wall-clock overrun.  On timeout the worker pool is torn down;
        subsequent calls respawn it.
        """
        executor = self._get_executor()
        fut = executor.submit(_run_task, call, str(self._config.spill_dir))

        try:
            result: WorkerResult = await asyncio.wait_for(
                asyncio.wrap_future(fut), timeout=deadline_s
            )
        except TimeoutError:
            # Loky has no public per-future kill; tearing the pool down is
            # the only available primitive.  Other in-flight tasks become
            # BrokenProcessPool and will be retried by their callers.
            self._shutdown_sync(wait=False)
            raise
        except asyncio.CancelledError:
            # Don't tear the pool down; just unlink the spill if the
            # worker still produces one after cancellation.
            fut.add_done_callback(_unlink_spill_on_done)
            raise

        logger.trace(
            "[LokyBackend:{}|{}:{}:{}] fn={} wall={:.3f}s rss={:.1f}MB user={:.3f}s sys={:.3f}s",
            self._config.pool_id,
            result.node_id or self._config.node_id,
            result.worker_id,
            result.worker_pid,
            call.function_name,
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

    def _shutdown_sync(self, *, wait: bool = False) -> None:
        """Sync shutdown — the body of :meth:`shutdown` without async wrap.

        Separated so module-level :func:`shutdown_executor` can call it
        from synchronous contexts (atexit, conftest teardown) without
        going through ``asyncio.run``.
        """
        if self._executor is None:
            return
        with contextlib.suppress(Exception):
            self._executor.shutdown(kill_workers=True, wait=wait)
        self._executor = None
        self._worker_env.clear()

    async def shutdown(self, *, wait: bool = False) -> None:
        """Kill all workers if any were spawned.

        ``wait=True`` blocks until loky's manager thread has finished its
        shutdown sequence (incl. firing done-callbacks); use it from
        explicit teardown paths so spill-unlink callbacks fire before
        exit.  The ``wait=False`` default is for the timeout branch of
        :meth:`execute`, which must return promptly to propagate
        :class:`TimeoutError`.  Racy under concurrent submits by design.
        """
        self._shutdown_sync(wait=wait)


# ---------------------------------------------------------------------------
# Default singleton for backward compatibility
# ---------------------------------------------------------------------------

_default_backend: LokyBackend | None = None


def default_loky_backend() -> LokyBackend:
    """Get or create the process-scoped default :class:`LokyBackend`."""
    global _default_backend
    if _default_backend is None:
        _default_backend = LokyBackend()
    return _default_backend


def shutdown_executor(*, wait: bool = False) -> None:
    """Backward-compat: shut down the default :class:`LokyBackend` singleton.

    Sync wrapper; safe to call from atexit handlers or sync test teardown.
    From an async context, prefer ``await default_loky_backend().shutdown()``.
    """
    global _default_backend
    if _default_backend is None:
        return
    backend = _default_backend
    _default_backend = None
    backend._shutdown_sync(wait=wait)


# Module-level backward-compat shims so existing callers that reach for
# the old internals (`_CONFIG`, `_get_executor()`) continue to work
# against the default singleton.
class _ConfigProxy:
    """Read access to the default backend's :class:`WorkerConfig`."""

    def __getattr__(self, name: str) -> Any:
        return getattr(default_loky_backend().config, name)


_CONFIG = _ConfigProxy()


def _get_executor() -> loky.ProcessPoolExecutor:
    """Backward-compat: return the default singleton's underlying executor."""
    return default_loky_backend()._get_executor()


# ---------------------------------------------------------------------------
# Public surface used by callers
# ---------------------------------------------------------------------------


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
    On timeout the default-backend pool is torn down; subsequent calls
    respawn it.

    Wrapper over the process-scoped default :class:`LokyBackend`; callers
    that want isolated pools should instantiate a :class:`LokyBackend`
    directly and call :meth:`LokyBackend.execute`.
    """
    call = WorkerCall(
        code=code,
        function_name=function_name,
        args=list(args or []),
        kwargs=dict(kwargs or {}),
        python_path=[str(p) for p in (python_path or [])],
        env={k: (None if v is None else str(v)) for k, v in (env_updates or {}).items()},
    )
    return await default_loky_backend().execute(call, deadline_s=timeout)
