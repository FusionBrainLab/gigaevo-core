import atexit
from datetime import UTC, datetime
import os
import queue
import re
import sys
import threading
import time

from loguru import logger

# Upper bound on records queued but not yet flushed to a sink's real
# destination. When a destination stalls (an NFS write, a log rotation),
# records past this bound are DROPPED rather than back-pressuring the caller.
# A `logger.*` call must never block the asyncio event loop — logs are
# best-effort, the run is not.
_LOG_QUEUE_MAXSIZE = 20000

_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}

# Live non-blocking sinks, drained best-effort at interpreter exit. Daemon writer
# threads are killed without draining their queues at shutdown; without the exit
# flush the tail — which on a crash holds the traceback — would be lost.
_ACTIVE_SINKS: list["_NonBlockingSink"] = []
_EXIT_FLUSH_TIMEOUT_S = 2.0


def _parse_size(rotation: str) -> int:
    """Parse a loguru-style size string (e.g. ``"50 MB"``) to bytes.

    Returns 0 for time-based policies (``"1 day"``) or unparseable input, which
    disables size rotation.
    """
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*([KMG]?B)\s*", rotation or "", re.IGNORECASE
    )
    if not match:
        return 0
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()])


class _NonBlockingSink:
    """A loguru sink whose ``__call__`` never blocks the calling thread.

    Formatted records are handed to a bounded queue and flushed to the real
    (possibly slow) destination by a dedicated daemon thread. A full queue drops
    records instead of blocking the producer — this is what stops a stalled NFS
    write from freezing the single asyncio event loop (the loguru ``enqueue=True``
    pipe would block the producer once its OS buffer filled).

    Records are queued as loguru hands them over, unchanged. A loguru ``Message``
    subclasses ``str``, so plain text writers work as-is while record-aware
    writers keep access to ``message.record`` (the exception sink needs it).

    Drops are never silent: once the queue recovers, the writer emits one marker
    accounting for the shed records. ``flush`` offers a BOUNDED best-effort drain
    so a clean shutdown does not discard the queued tail (e.g. a crash traceback).
    """

    def __init__(self, write, *, name: str, maxsize: int = _LOG_QUEUE_MAXSIZE) -> None:
        self._write = write
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._drop_lock = threading.Lock()
        self.dropped = 0
        self._thread = threading.Thread(target=self._drain, name=name, daemon=True)
        self._thread.start()
        _ACTIVE_SINKS.append(self)

    def __call__(self, message) -> None:
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            with self._drop_lock:
                self.dropped += 1

    def flush(self, timeout: float) -> None:
        """Block up to ``timeout`` seconds for the queue to fully drain.

        Bounded by design: an unbounded join would reintroduce the very
        event-loop hang the non-blocking sink exists to prevent. Best-effort — a
        destination still stalled at the deadline keeps its remaining tail.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self._queue.unfinished_tasks == 0:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.01, remaining))

    def _drain(self) -> None:
        reported = 0
        while True:
            text = self._queue.get()
            try:
                self._write(text)
                dropped = self.dropped
                if dropped > reported:
                    # A gap in the log must never be silent: once the queue has
                    # recovered, emit one marker for the records back-pressure shed.
                    self._write(
                        f"[logging] {dropped - reported} record(s) dropped "
                        "under back-pressure\n"
                    )
                    reported = dropped
            except Exception:  # a sink error must never kill the writer thread
                pass
            finally:
                self._queue.task_done()


def _flush_active_sinks_at_exit() -> None:
    """Bounded, best-effort drain of every live sink at interpreter exit.

    The total wait is capped so a stalled destination cannot hang the exit.
    """
    deadline = time.monotonic() + _EXIT_FLUSH_TIMEOUT_S
    for sink in list(_ACTIVE_SINKS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sink.flush(remaining)


atexit.register(_flush_active_sinks_at_exit)


class _RotatingFileWriter:
    """Size-rotating file writer used as a ``_NonBlockingSink`` destination.

    Appends UTF-8 text; when the file exceeds ``max_bytes`` it is renamed to
    ``<path>.1`` (one previous generation kept) and a fresh file is opened. No
    compression — synchronous zip on rotation was the dominant event-loop stall
    this replaces.
    """

    def __init__(self, path: str, *, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, text: str) -> None:
        try:
            self._fh.write(text)
        except (ValueError, OSError):
            # A closed handle (a prior rotation whose reopen failed) or a
            # transient NFS write fault must not silently kill file logging for
            # the rest of the run: reopen and retry the write once so records
            # resume the moment the mount recovers. Only the write is retried,
            # so a post-write flush/rotate fault can never duplicate a line.
            self._reopen()
            self._fh.write(text)
        self._fh.flush()
        if self._max_bytes and self._fh.tell() >= self._max_bytes:
            self._rotate()

    def _reopen(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = open(self._path, "a", encoding="utf-8")

    def _rotate(self) -> None:
        self._fh.close()
        try:
            os.replace(self._path, f"{self._path}.1")
        except OSError:
            # A rename failure (e.g. an NFS ESTALE/EIO — the exact fault this
            # sink hardens against) must not leave the writer holding a closed
            # handle for the rest of the run. The unrotated file is still
            # present, so reopening it keeps file logging alive; unrotated
            # growth is strictly safer than silently dead logs.
            pass
        try:
            self._fh = open(self._path, "a", encoding="utf-8")
        except OSError:
            # Reopen itself faulted (compound fault): __call__'s write-retry
            # reopens on the next record once the mount recovers.
            pass


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "30 days",
    enable_colors: bool = True,
) -> str:
    """
    Set up logging with file and console output.

    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Size-based rotation policy (e.g., "50 MB"). The non-blocking
            file sink supports byte sizes only; a time-based value ("1 day")
            disables rotation (a warning is logged).
        retention: Accepted for configuration compatibility. The file sink keeps
            a single previous generation (``<file>.1``); time-based retention is
            not implemented.
        enable_colors: Whether to enable colored console output

    Returns:
        Path to the main log file
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evolution_{timestamp}.log")

    # Remove any existing handlers to avoid duplicates
    logger.remove()

    # Enhanced console format with comprehensive coloring
    if enable_colors and sys.stdout.isatty():
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<blue>{function}</blue>:<yellow>{line}</yellow> | "
            "<level>{message}</level>"
        )
    else:
        # Fallback format for non-TTY environments
        console_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

    # Console and file sinks are non-blocking (see _NonBlockingSink): the caller
    # never waits on the sink's destination, so a stalled NFS write can no longer
    # freeze the event loop. enqueue=False is safe here because neither sink
    # re-enters logging — only the exception sink below does, so only it keeps
    # loguru's own background queue.
    console_sink = _NonBlockingSink(
        lambda text: (sys.stdout.write(text), sys.stdout.flush()), name="log-console"
    )
    logger.add(
        console_sink,
        level=level,
        format=console_format,
        colorize=enable_colors and sys.stdout.isatty(),
        backtrace=True,
        diagnose=True,
        enqueue=False,
    )

    max_bytes = _parse_size(rotation)
    file_sink = _NonBlockingSink(
        _RotatingFileWriter(log_file, max_bytes=max_bytes), name="log-file"
    )
    logger.add(
        file_sink,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        colorize=False,
        backtrace=True,
        diagnose=True,
        enqueue=False,
    )

    # Install the EXCEPTION canonical-event sink — emits one [EXCEPTION]
    # line for every logger.exception(...). The sink guards against
    # recursion via record["extra"]["canonical_event"].
    from gigaevo.monitoring.exception_sink import install_exception_sink

    install_exception_sink()

    if rotation and max_bytes <= 0:
        logger.warning(
            f"[LoggerSetup] rotation policy {rotation!r} is not a byte size; "
            "size-based rotation is disabled. The non-blocking file sink supports "
            "only size-based rotation (e.g. '50 MB')."
        )

    logger.info(f"[LoggerSetup] Logging to console and file: {log_file}")
    return log_file
