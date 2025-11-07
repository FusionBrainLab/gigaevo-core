from abc import ABC, abstractmethod


class LogWriter(ABC):
    @abstractmethod
    def bind(self, path: list[str], labels: dict[str, str]) -> "LogWriter":
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

    @abstractmethod
    def close(self) -> None:
        pass
