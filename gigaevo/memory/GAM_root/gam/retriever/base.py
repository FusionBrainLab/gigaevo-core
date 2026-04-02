from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from GAM_root.gam.schemas import Hit, InMemoryPageStore


class AbsRetriever(ABC):
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    @abstractmethod
    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]:
        pass

    @abstractmethod
    def build(self, page_store: InMemoryPageStore):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def update(self, page_store: InMemoryPageStore):
        pass
