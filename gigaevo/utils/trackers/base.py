from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricsHistoryReader(Protocol):
    """Read side of a metrics backend's history store.

    Backends that persist time-series history (Redis, disk) expose it
    through this protocol so monitors can read metrics without knowing
    which backend the run writes to. Entries follow the backend schema
    ``{"s": step, "t": wall_time, "v": value, "k": kind}``; ``start`` /
    ``end`` use LRANGE semantics (inclusive, negative = from the end).
    """

    def get_history(
        self, tag: str, start: int = 0, end: int = -1
    ) -> list[dict[str, Any]]: ...


class LogWriter(ABC):
    @abstractmethod
    def bind(self, *, path: list[str] | None = None) -> "LogWriter":
        pass

    @abstractmethod
    def scalar(self, metric: str, value: float, **kwargs) -> None:
        pass

    @abstractmethod
    def hist(self, metric: str, values: list[float], **kwargs) -> None:
        pass

    @abstractmethod
    def text(self, tag: str, text: str, **kwargs) -> None:
        pass

    def clear_series(self, metric: str, **kwargs) -> None:
        """Delete all history for a metric series so it can be rewritten.

        Used by MetricsTracker to rewrite the frontier when NO_CACHE stages
        change program metrics retroactively.  Default is a no-op; the Redis
        backend implements the actual DELETE.
        """

    @abstractmethod
    def close(self) -> None:
        pass
