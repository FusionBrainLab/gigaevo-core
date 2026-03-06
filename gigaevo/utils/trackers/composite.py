from __future__ import annotations

from typing import Any

from gigaevo.utils.trackers.base import LogWriter


class CompositeLogger(LogWriter):
    """Logger that broadcasts writes to multiple underlying loggers.

    Example:
        >>> tb = init_tb(tb_config)
        >>> redis = init_redis(redis_config)
        >>> logger = CompositeLogger([tb, redis])
        >>> logger.scalar("loss", 0.5)  # writes to both TB and Redis
    """

    def __init__(self, loggers: list[LogWriter]):
        if not loggers:
            raise ValueError("CompositeLogger requires at least one logger")
        self._loggers = list(loggers)

    def bind(self, *, path: list[str] | None = None) -> "BoundComposite":
        return BoundComposite(self, path or [])

    def scalar(self, metric: str, value: float, **kwargs) -> None:
        for logger in self._loggers:
            try:
                logger.scalar(metric, value, **kwargs)
            except Exception:
                pass

    def hist(self, metric: str, values: Any, **kwargs) -> None:
        for logger in self._loggers:
            try:
                logger.hist(metric, values, **kwargs)
            except Exception:
                pass

    def text(self, tag: str, text: str, **kwargs) -> None:
        for logger in self._loggers:
            try:
                logger.text(tag, text, **kwargs)
            except Exception:
                pass

    def close(self) -> None:
        for logger in self._loggers:
            try:
                logger.close()
            except Exception:
                pass


class BoundComposite(LogWriter):
    """Bound version of CompositeLogger with a path prefix."""

    def __init__(self, base: CompositeLogger, path: list[str]):
        self._base = base
        self._path = list(path)

    def bind(self, *, path: list[str] | None = None) -> "BoundComposite":
        return BoundComposite(self._base, [*self._path, *(path or [])])

    def scalar(self, metric: str, value: float, **kwargs) -> None:
        path = [*self._path, *kwargs.pop("path", [])]
        self._base.scalar(metric, value, path=path, **kwargs)

    def hist(self, metric: str, values: Any, **kwargs) -> None:
        path = [*self._path, *kwargs.pop("path", [])]
        self._base.hist(metric, values, path=path, **kwargs)

    def text(self, tag: str, text: str, **kwargs) -> None:
        path = [*self._path, *kwargs.pop("path", [])]
        self._base.text(tag, text, path=path, **kwargs)

    def close(self) -> None:
        self._base.close()
