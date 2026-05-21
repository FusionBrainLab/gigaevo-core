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

        Default is a no-op; backends that persist history (e.g. Redis) override
        this to delete the stored series.
        """

    @abstractmethod
    def close(self) -> None:
        pass
