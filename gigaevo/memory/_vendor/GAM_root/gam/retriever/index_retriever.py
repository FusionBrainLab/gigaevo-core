from __future__ import annotations

import os
from typing import Any

from gigaevo.memory._vendor.GAM_root.gam.retriever.base import AbsRetriever
from gigaevo.memory._vendor.GAM_root.gam.schemas import Hit, InMemoryPageStore, Page
from loguru import logger


class IndexRetriever(AbsRetriever):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.pages: list[Page] = []

    def load(self):
        index_dir = self.config.get("index_dir")
        try:
            # 正确创建 InMemoryPageStore 实例，会自动加载页面
            self.page_store = InMemoryPageStore(
                dir_path=os.path.join(index_dir, "pages")
            )
        except Exception as e:
            logger.error("cannot load index, error: ", e)

    def build(self, page_store: InMemoryPageStore):
        # 创建一个新的 InMemoryPageStore 实例用于保存
        target_path = os.path.join(self.config.get("index_dir"), "pages")
        new_store = InMemoryPageStore(dir_path=target_path)
        # 获取 page_store 中的所有页面并保存到新实例
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
            # 尝试将查询解析为页面索引
            try:
                page_index = [
                    int(idx.strip())
                    for idx in query.split(",")
                    if idx.strip().isdigit()
                ]
            except ValueError:
                # 如果解析失败，跳过这个查询
                continue

            for pid in page_index:
                p = self.page_store.get(pid)
                if not p:
                    continue
                hits.append(
                    Hit(
                        page_id=str(pid),  # 使用页面索引作为page_id
                        snippet=p.content,
                        source="page_index",
                        meta={},
                    )
                )
        return [hits]  # 包装成 List[List[Hit]] 格式
