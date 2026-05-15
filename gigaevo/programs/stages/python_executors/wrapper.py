"""Loky-backed Python subprocess executor.

Replaces the legacy hand-rolled WorkerPool + length-prefixed cloudpickle
protocol with :func:`loky.get_reusable_executor`.  Workers spawn lazily on
first :meth:`submit`, are cached across calls, and on Linux receive
``PR_SET_PDEATHSIG`` so they die with the parent.

Results are spilled to a per-call file under :data:`ExecutorConfig.spill_dir`
and the parent reads them back via :mod:`mmap`; result size is bounded by
free disk space rather than parent RAM.

Spill backing storage
=====================
The default ``spill_dir`` is ``$TMPDIR/gigaevo-<uid>``.  On most Linux
distributions ``$TMPDIR`` resolves to ``/tmp`` which is mounted ``tmpfs``
— i.e. RAM-backed.  In that configuration the "spill bounds result size
by free disk space" claim is misleading: a B-byte result transits the
worker's serialise buffer (B bytes), lands in the tmpfs page cache
(another B bytes), then ``mmap``-faults into the parent's address space.
Peak memory cost is roughly ``2*B + epsilon``.

To genuinely cap result size against a disk budget rather than RAM,
point ``GIGAEVO_EXECUTOR_SPILL_DIR`` at a path on a real filesystem
(e.g. ``/var/tmp/gigaevo-<uid>``, or an SSD-backed working directory).
Inspect at runtime with ``findmnt -no FSTYPE $TMPDIR``.

Result pickle protocol
======================
We use ``cloudpickle.dump(..., protocol=5)`` with **no**
``buffer_callback``.  Protocol 5 introduced the PEP 574 out-of-band
buffer mechanism, but the ``buffer_callback=None`` default falls back
to inline ``BYTEARRAY8`` opcodes for :class:`pickle.PickleBuffer`
instances — concretely, numpy/torch tensors are serialised by *copying*
their bytes into the pickle stream and materialise into freshly-allocated
buffers on unpickle.  The unpickled tensor in the parent therefore
**does not alias** the spill ``mmap``, so closing the ``mmap`` context
in :func:`_load_spill` does not invalidate the result.  See
https://docs.python.org/3/library/pickle.html#out-of-band-buffers .

Adopting ``buffer_callback`` would skip the bytes copy on both sides
and shrink the metadata pickle from O(payload) to O(metadata) — for a
1 GB float32 array, from ~1 GB down to ~120 B — at the cost of a
multi-file spill format (manifest + N buffer sidecars) and the parent
having to keep the buffer ``mmap``-s alive for the lifetime of the
unpickled result.  That's exactly the dangling-mmap hazard the current
single-file inline design avoids.  Tracked as a future optimisation;
the inline path is correct.

Worker state lifecycle
======================
Loky pool workers are reused across calls.  Per-call mutations are
restored by :func:`exec_runner._run_one`'s ``finally`` block:
``sys.path``, ``os.getcwd()``, and ``os.environ`` (via ``_scoped_env``)
all snapshot on entry and restore on exit, so ``python_path`` and
``env_updates`` do **not** accumulate across calls within the same
worker.  What *does* persist across calls in a long-lived worker:

* ``sys.modules`` entries created by user ``import`` statements (matches
  CPython ``-c`` semantics — once a module is imported its state lives
  until interpreter teardown);
* ``sys.modules["user_code"]``, **replaced** (not appended) each call,
  so the synthetic-module slot is at most one entry regardless of call
  count;
* ``linecache.cache["user_code.py"]``, **overwritten** each call;
* ``cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES``, a *set* keyed
  by module name — re-registering ``"user_code"`` is idempotent so the
  set stays at most one element regardless of call count.

All four cases are bounded.  User code that mutates other globals —
``sys.argv``, signal handlers installed mid-call, monkey-patched
stdlib modules — is **not** restored, by design: correctly unwinding
arbitrary monkey patches is undecidable, and CPython itself doesn't
do it across ``exec()`` calls.

Start method
============
``context="loky"`` selects loky's ``Popen`` (``loky.backend.popen_loky_posix``),
which dispatches to ``_posixsubprocess.fork_exec`` with ``close_fds=True`` and
an explicit ``keep_fds`` allowlist.  Mechanically this is fork+exec — never a
bare ``fork()`` — so children get a *fresh Python interpreter* re-launched
from ``sys.executable``, not a memory snapshot of the parent.  Practical
consequences:

* No parent module state is inherited.  CUDA contexts, ``torch`` lazy
  initialisation, ``langfuse`` handlers, ``loguru`` sinks, open Redis
  connections, ``asyncio`` event-loop state, ``atexit`` registrations —
  none survive into the worker.  The worker starts as a clean Python
  process and only sees what loky reduces over the spawn pipe.
* No parent file descriptors leak: ``close_fds=True`` slams everything
  shut except the loky-owned pipes and stdin/stdout/stderr.  Verified
  empirically against ``/proc/self/fd`` (probe in the bug-hunt report).
* ``os.environ`` in the worker is what loky's ``env=`` kwarg sets, not
  the parent's mutable ``os.environ`` — see :func:`_get_executor` below
  for the snapshot policy.
* Worker cold-start cost is ~spawn (~100ms), not ~fork (~1ms).  The pool
  is reused across calls so this is amortised over the run.

Trust boundary
==============
User code runs *unsandboxed* inside the worker.  We only contain ambient
secrets (env scrub), process lifetime (``PR_SET_PDEATHSIG``), and signal
dispositions (default handlers).  We do **not** restrict CPU, memory,
``fork``, file-descriptor count, or filesystem access; nor do we treat
the worker's return value as untrusted — :mod:`cloudpickle` happily
executes ``__reduce__`` gadgets on unpickle in the *parent*.  The model
is "trust the code as much as you trust whoever generated it"; in the
evolutionary search loop this is an LLM, so the boundary lives at the
LLM call, not here.
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

    Defaults match the historical hand-rolled WorkerPool: no
    preallocation (``max_workers=None`` → loky picks ``cpu_count()``),
    a generous idle timeout so DAGs with bursty traffic keep workers
    warm, and result spill to ``$TMPDIR`` (typically tmpfs).
    """

    max_workers: int | None = None
    idle_timeout_s: int = 300
    spill_dir: Path = Path(tempfile.gettempdir()) / f"gigaevo-{os.getuid()}"

    @classmethod
    def from_env(cls) -> ExecutorConfig:
        """Construct from ``GIGAEVO_EXECUTOR_*`` environment variables.

        The spill directory is resolved (``..`` collapsed) but symlinks are
        *not* followed, so an operator who points the env var at a symlink
        gets exactly the path they asked for — auditable in logs.  The
        default is ``$TMPDIR/gigaevo-<uid>`` (not bare ``$TMPDIR``) so
        spill files never share a world-readable directory with other
        users' data.

        Under pytest-xdist (``PYTEST_XDIST_WORKER_COUNT`` set), if
        ``GIGAEVO_EXECUTOR_MAX_WORKERS`` is not explicitly set, the
        per-process worker count is auto-capped to
        ``max(1, cpu_count // xdist_worker_count)``.  Otherwise an 8-xdist /
        28-CPU host would fork 8×28 = 224 loky workers, exhausting the
        process table and oversubscribing the scheduler.  An explicit
        ``GIGAEVO_EXECUTOR_MAX_WORKERS`` always wins — operators who know
        what they're doing aren't second-guessed.
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

        idle = _pos_int("GIGAEVO_EXECUTOR_IDLE_TIMEOUT_S", 300) or 300
        spill = os.environ.get("GIGAEVO_EXECUTOR_SPILL_DIR")
        if spill:
            # ``Path.resolve(strict=False)`` collapses ``..`` and ``.``
            # segments without requiring the path to exist yet (the
            # module-level ``mkdir(parents=True, exist_ok=True)`` below
            # creates it).  Resolution defeats traversal smuggled through
            # otherwise-innocuous parent segments.
            spill_path = Path(spill).resolve(strict=False)
        else:
            spill_path = Path(tempfile.gettempdir()) / f"gigaevo-{os.getuid()}"

        max_workers = _pos_int("GIGAEVO_EXECUTOR_MAX_WORKERS", None)
        if max_workers is None:
            xdist_count = _pos_int("PYTEST_XDIST_WORKER_COUNT", None)
            if xdist_count and xdist_count > 1:
                cpu = os.cpu_count() or 1
                max_workers = max(1, cpu // xdist_count)

        return cls(
            max_workers=max_workers,
            idle_timeout_s=idle,
            spill_dir=spill_path,
        )


_CONFIG: ExecutorConfig = ExecutorConfig.from_env()
# Custom spill dirs may not exist yet; opt into create-on-import so
# callers don't have to remember to mkdir before the first submit.
# ``mode=0o700`` keeps spill files unreadable by other UIDs even on
# shared hosts (only honoured on directory creation — pre-existing
# dirs retain their mode, which is the operator's choice to make).
_CONFIG.spill_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

# Capture the scrubbed env exactly once at import time.  Loky's
# ``get_reusable_executor`` reuses a cached pool only when the kwargs of the
# current call equal those used to construct it (``kwargs == _executor_kwargs``
# in ``loky.reusable_executor``).  If we rebuilt the dict on every submit
# from ``os.environ`` — which mutates throughout a process's lifetime — the
# pool would be torn down and respawned whenever any whitelisted variable
# changed, also breaking concurrent in-flight tasks with ``BrokenProcessPool``.
_WORKER_ENV: dict[str, str] = {}  # populated lazily; see _get_executor()


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
        rc = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if rc != 0:
            # Containers with restricted ``prctl`` (seccomp denylist,
            # gVisor, some sandboxed CI runners) will fail this; without
            # the death-signal, workers become orphans if the parent dies
            # uncleanly.  Log so operators can spot it; don't fail
            # initialization — the loky pool itself still functions.
            errno_ = ctypes.get_errno()
            logger.debug(
                "[exec_wrapper] PR_SET_PDEATHSIG failed rc={} errno={}", rc, errno_
            )
    except Exception as exc:
        logger.debug("[exec_wrapper] PR_SET_PDEATHSIG unavailable: {}", exc)


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
            # mkstemp guarantees ``O_CREAT | O_EXCL`` (so a pre-existing
            # symlink at the chosen path causes failure rather than
            # silent reuse) and mode 0600 (only this UID can read the
            # spilled result).  See ``_load_spill`` for the matching
            # ``O_NOFOLLOW`` parent-side hardening.
            fd, spill_path = tempfile.mkstemp(
                prefix="gevo-result-", suffix=".pkl", dir=spill_dir
            )
        except OSError as exc:
            error = _error_from_exc(
                exc, f"Failed to create spill file in {spill_dir!r}:"
            )
        else:
            try:
                # ``os.fdopen`` takes ownership of fd on success; on
                # failure (e.g. EMFILE) it does *not*, so close it
                # ourselves to avoid leaking a descriptor.
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


def _error_from_exc(exc: BaseException, prefix: str) -> WorkerError:
    buf = [prefix, "\n"]
    buf.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return WorkerError(stderr="".join(buf))


def _get_executor() -> Any:
    # Freeze the env on first use rather than at import time: tests
    # (e.g. ``isolated_spill_dir``) and early bootstrap code legitimately
    # mutate ``os.environ`` after the wrapper module is loaded but before
    # the first submit.  After that first call the env is stable for the
    # life of the pool, so subsequent calls reuse the existing workers.
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

    Reaches into loky's module-global cache so we don't pay the spawn cost
    of constructing a pool just to tear it down.  Safe to call repeatedly.

    ``wait`` controls whether we block until loky's executor-manager thread
    has finished its shutdown sequence (SIGKILLing workers, draining the
    result queue, firing done-callbacks for pending futures).  The default
    ``wait=False`` is the historical async path: callers in the timeout
    branch of :func:`run_exec_runner` need to return promptly so the
    original :class:`TimeoutError` propagates without blocking the event
    loop on an OS-level reap.  The ``wait=True`` path is for explicit
    teardown — run.py's ``finally`` block, session-end fixtures — where
    we want spill-unlink callbacks to fire *before* the process exits, so
    ``/tmp/gigaevo-<uid>`` doesn't accumulate orphaned spill files across
    crash/restart cycles.

    Note: this is racy by design.  If a concurrent ``asyncio.gather`` task
    is mid-``executor.submit(...)`` when shutdown happens, that submit (or
    the in-flight future) surfaces as :class:`loky.process_executor.BrokenProcessPool`
    in the caller.  ``run_exec_runner``'s ``CancelledError`` path
    deliberately does *not* call this.
    """
    # Re-capture the env on next submit: caller is signalling "fresh pool"
    # and may have legitimately mutated the env since first capture.
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
    """Read a cloudpickle spill file, preferring mmap to skip a bytes copy.

    The file was created by the worker via ``tempfile.mkstemp`` (mode
    0600, ``O_EXCL`` so no pre-existing symlink can be reused).  We open
    it here with ``O_NOFOLLOW`` as defence-in-depth: a same-UID attacker
    who manages to replace the path with a symlink between worker exit
    and parent read will trip ``ELOOP`` rather than feed an attacker-
    controlled file into :func:`cloudpickle.loads` (an RCE primitive in
    the parent process).
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
