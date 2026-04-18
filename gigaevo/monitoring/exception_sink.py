"""Loguru sink that emits EXCEPTION canonical events on `logger.exception(...)`.

Every `logger.exception(...)` call produces a log record with `exception`
populated. This sink fires once per such record to emit a single
[EXCEPTION] canonical event line.

Recursion guard: canonical-event lines are tagged with
`extra["canonical_event"] = True` by `emit()`. The sink filters those
records out so its own emitted EXCEPTION lines never trigger a second
EXCEPTION pass.
"""

from __future__ import annotations

from loguru import logger

from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import Exception_


def install_exception_sink() -> int:
    """Install the EXCEPTION-emitting sink on the global loguru logger.

    Returns the sink handler id so callers (including tests) can remove it
    with `logger.remove(id)`.
    """

    def _sink(message):
        record = message.record
        # Never recurse on our own canonical-event emissions.
        if record["extra"].get("canonical_event"):
            return
        exc_info = record.get("exception")
        if exc_info is None:
            return
        exc_type = exc_info.type.__name__ if exc_info.type else "Unknown"
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

    return logger.add(_sink, level="DEBUG", format="{message}")
