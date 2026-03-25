from abc import ABC, abstractmethod
from GAM_root.gam.schemas import InMemoryPageStore, Hit
from typing import Any, List, Dict

class AbsRetriever(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.config = config

    @abstractmethod
    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
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