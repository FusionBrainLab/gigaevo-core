"""Pluggable executor backend interface.

The single Protocol declared here is the swap-point between
:func:`gigaevo.programs.stages.python_executors.wrapper.run_exec_runner`
and the concrete subprocess pool.  :class:`LokyBackend` in
:mod:`gigaevo.programs.stages.python_executors.wrapper` is the only
implementation today; future backends (Redis-Stream-mediated remote
workers, HTTP submission services, Ray actors) implement the same
contract and slot in at call sites without modification.

The Protocol is intentionally minimal: ``execute`` and ``shutdown``.
Lifecycle hooks (``on_submit`` / ``on_complete`` / ``on_shutdown``) are
declared as well but are optional — backends that have nothing useful to
notify can leave the registration methods as no-ops.  Per-worker-process
lifecycle (``on_worker_start``, ``on_worker_exit``) is deliberately
absent: it requires cross-process signalling that has no immediate
consumer on local-subprocess backends, and becomes natural to wire up
only when the worker layer itself emits network events.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gigaevo.programs.stages.python_executors.exec_runner import WorkerCall
    from gigaevo.programs.stages.python_executors.wrapper import ExecutionMetrics


SubmitHandler = Callable[["WorkerCall"], None]
"""Handler invoked just before a call is dispatched to the underlying executor."""

CompleteHandler = Callable[
    ["WorkerCall", "ExecutionMetrics | None", BaseException | None],
    None,
]
"""Handler invoked after execution finishes.

``metrics`` is ``None`` when the call did not produce a result envelope —
for example on wall-clock timeout, on cancellation before the worker
reported back, or on a failure to construct the worker pool before the
call was ever submitted.  In those cases ``exc`` carries the reason.
``metrics`` is non-None whenever the worker actually ran the call,
including the failure case where the worker reported a ``WorkerError``."""

ShutdownHandler = Callable[[], None]
"""Handler invoked once when :meth:`ExecutorBackend.shutdown` is called."""


@runtime_checkable
class ExecutorBackend(Protocol):
    """Subprocess / remote executor abstraction.

    Implementations encapsulate the lifecycle of one logical pool of
    worker processes (local subprocesses, remote workers reachable over
    a network, or anything else that can run a :class:`WorkerCall`).
    """

    async def execute(self, call: WorkerCall, *, deadline_s: int) -> Any:
        """Run *call* in a worker; return its value or raise.

        Contract:

        - Returns the unpacked user-function return value on success.
        - Raises ``ExecRunnerError`` on user-code failure with ``.stderr``
          populated by the formatted traceback.
        - Raises :class:`asyncio.TimeoutError` on wall-clock overrun
          (``deadline_s`` exceeded).
        - May raise other backend-specific exceptions for infrastructure
          failures (broken pool, network error, …); callers treat these
          as transient and retry per their own policy.
        """
        ...

    async def shutdown(self, *, wait: bool = False) -> None:
        """Release all backend-owned resources.

        ``wait=True`` blocks until cleanup completes (used during
        explicit teardown so done-callbacks fire before exit).
        ``wait=False`` returns promptly (used in the timeout branch of
        :meth:`execute` where prompt return matters more than complete
        cleanup).  Idempotent: calling twice is safe.
        """
        ...

    def on_submit(self, handler: SubmitHandler) -> None:
        """Register a handler fired immediately before a call is submitted."""
        ...

    def on_complete(self, handler: CompleteHandler) -> None:
        """Register a handler fired after execution finishes (success or failure)."""
        ...

    def on_shutdown(self, handler: ShutdownHandler) -> None:
        """Register a handler fired once when :meth:`shutdown` is called."""
        ...
