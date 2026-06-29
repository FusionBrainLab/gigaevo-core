from __future__ import annotations

import os
from typing import Any

from loguru import logger

from gigaevo.memory._vendor.GAM_root.gam.retriever.base import AbsRetriever
from gigaevo.memory._vendor.GAM_root.gam.schemas import Hit, InMemoryPageStore, Page


class IndexRetriever(AbsRetriever):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.pages: list[Page] = []

    def load(self):
        index_dir = self.config.get("index_dir")
        try:
            self.page_store = InMemoryPageStore(
                dir_path=os.path.join(index_dir, "pages")
            )
        except Exception as e:
            logger.error("[Memory][GAM][IndexRetriever] Cannot load index: {}", e)

    def build(self, page_store: InMemoryPageStore):
        target_path = os.path.join(self.config.get("index_dir"), "pages")
        new_store = InMemoryPageStore(dir_path=target_path)
        pages = (
            page_store._pages if hasattr(page_store, "_pages") else page_store.load()
        )
        new_store.save(pages)
        self.page_store = new_store

    def update(self, page_store: InMemoryPageStore):
        self.build(page_store)

    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]:
        hits: list[Hit] = []
        for query in query_list:
            try:
                page_index = [
                    int(idx.strip())
                    for idx in query.split(",")
                    if idx.strip().isdigit()
                ]
            except ValueError:
                continue

            for pid in page_index:
                p = self.page_store.get(pid)
                if not p:
                    continue
                amem_id = str((getattr(p, "meta", None) or {}).get("amem_id") or "")
                hits.append(
                    Hit(
                        page_id=amem_id.strip() or str(pid),
                        snippet=p.content,
                        source="page_index",
                        meta={"page_index": pid},
                    )
                )
        return [hits]
