from abc import ABC, abstractmethod


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
