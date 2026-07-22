"""Loguru sink that emits EXCEPTION canonical events on `logger.exception(...)`.

Every `logger.exception(...)` call produces a log record with `exception`
populated. This sink fires once per such record to emit a single
[EXCEPTION] canonical event line.

Two properties keep the sink off the critical path of the asyncio event loop:

* ``filter=_should_emit`` decides *before* the record is handed over, on the
  calling thread. The ordinary DEBUG/INFO firehose is rejected there and never
  reaches the sink's queue at all — only records that will really produce an
  EXCEPTION event do.
* The sink is a ``_NonBlockingSink``: the calling thread only does a
  ``put_nowait`` and ``emit()`` runs on the sink's daemon thread. That both
  avoids re-entering loguru's handler lock (``emit()`` itself calls
  ``logger.info()``) and makes a stalled write drop records rather than
  back-pressure the caller.

Recursion guard: canonical-event lines are tagged with
`extra["canonical_event"] = True` by `emit()`. The filter drops those records so
the sink's own EXCEPTION lines never trigger a second EXCEPTION pass.
"""

from __future__ import annotations

from loguru import logger

from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import Exception_
from gigaevo.utils.logger_setup import _NonBlockingSink


def _should_emit(record) -> bool:
    """Pre-enqueue gate, run by loguru on the calling thread.

    Filtering here rather than inside the sink body is the whole point: a record
    rejected here is never queued, so ordinary logging cannot fill the sink's
    queue and stall the producer.
    """
    if record["extra"].get("canonical_event"):
        return False
    return record["exception"] is not None


def _emit_exception_event(message) -> None:
    if type(message) is str:
        # A back-pressure marker written by the sink itself, not a log record.
        return
    record = message.record
    exc_info = record["exception"]
    # A caller reporting a *returned* failure (a stage result carrying its own
    # error) has no active exception for loguru to introspect, but does know the
    # class name — it binds it as extra["exc_type"]. Prefer that; fall back to
    # the live exception tuple raised through an `except` block.
    exc_type = record["extra"].get("exc_type")
    if not exc_type:
        exc_type = exc_info.type.__name__ if exc_info and exc_info.type else "Unknown"
    where = f"{record.get('name') or '?'}:{record.get('function') or '?'}"
    msg_head = str(record.get("message") or "")[:200]
    try:
        emit(
            Exception_(
                where=where,
                exc_type=exc_type,
                msg_head=msg_head,
            )
        )
    except Exception:  # pragma: no cover — never fail the sink
        pass


def install_exception_sink() -> int:
    """Install the EXCEPTION-emitting sink on the global loguru logger.

    Returns the sink handler id so callers (including tests) can remove it
    with `logger.remove(id)`.
    """
    sink = _NonBlockingSink(_emit_exception_event, name="log-exception")
    return logger.add(
        sink,
        level="DEBUG",
        format="{message}",
        filter=_should_emit,
        enqueue=False,
    )
