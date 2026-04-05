"""Manages GAM ResearchAgent lifecycle: build, rebuild, invalidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_store import CardStore


class GamSearch:
    """Manages GAM ResearchAgent: build from records, hold agent reference.

    The orchestrator calls ``build()`` during rebuild cycles and reads
    ``agent`` for search dispatch.
    """

    def __init__(
        self,
        *,
        research_agent_cls: type[Any],
        generator: Any,
        card_store: CardStore,
        checkpoint_dir: Path,
        gam_store_dir: Path,
        export_file: Path,
        enable_bm25: bool,
        allowed_gam_tools: set[str],
        gam_top_k_by_tool: dict[str, int],
        gam_pipeline_mode: str,
    ):
        self._research_agent_cls = research_agent_cls
        self._generator = generator
        self._card_store = card_store
        self._checkpoint_dir = checkpoint_dir
        self._gam_store_dir = gam_store_dir
        self._export_file = export_file
        self._enable_bm25 = enable_bm25
        self._allowed_gam_tools = allowed_gam_tools
        self._gam_top_k_by_tool = gam_top_k_by_tool
        self._gam_pipeline_mode = gam_pipeline_mode
        self.agent: Any = None

    def build(self) -> None:
        """Build/rebuild the ResearchAgent from exported records."""
        try:
            from gigaevo.memory.shared_memory.amem_gam_retriever import (
                build_gam_store,
                build_retrievers,
                load_amem_records,
            )
        except Exception as exc:
            raise RuntimeError(f"GAM helper modules are unavailable: {exc}") from exc

        self._gam_store_dir.mkdir(parents=True, exist_ok=True)
        if self._export_file.exists():
            records = load_amem_records(self._export_file)
        else:
            records = [c.model_dump() for c in self._card_store.cards.values()]

        memory_store, page_store, added = build_gam_store(records, self._gam_store_dir)
        logger.info(
            "[Memory] Loaded {} cards, added {} new pages.", len(records), added
        )

        retrievers = build_retrievers(
            page_store,
            self._gam_store_dir / "indexes",
            self._checkpoint_dir / "chroma",
            enable_bm25=self._enable_bm25,
            allowed_tools=sorted(self._allowed_gam_tools),
        )
        retrievers = {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self._allowed_gam_tools
        }
        if not retrievers:
            logger.info(
                "[Memory] No GAM retrievers enabled after applying allowed_gam_tools. "
                "GAM agentic search is disabled."
            )
            self.agent = None
            return

        self.agent = self._research_agent_cls(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=self._generator,
            max_iters=3,
            allowed_tools=sorted(self._allowed_gam_tools),
            top_k_by_tool=self._gam_top_k_by_tool,
            pipeline_mode=self._gam_pipeline_mode,
        )

    def invalidate(self) -> None:
        """Clear the agent so it will be rebuilt on next build() call."""
        self.agent = None
