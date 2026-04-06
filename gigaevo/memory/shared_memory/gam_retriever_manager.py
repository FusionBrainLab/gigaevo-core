"""GAM ResearchAgent lifecycle management — build, rebuild, invalidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import AnyCard


class GAMRetrieverManager:
    """Manages GAM ResearchAgent lifecycle: build, rebuild, invalidate.

    Reads memory_cards via a shared reference (no copy).
    """

    def __init__(
        self,
        *,
        generator: Any | None,
        research_agent_cls: type | None,
        export_file: Path,
        gam_store_dir: Path,
        checkpoint_dir: Path,
        enable_bm25: bool,
        allowed_gam_tools: set[str],
        gam_top_k_by_tool: dict[str, int],
        gam_pipeline_mode: str,
        memory_cards_ref: dict[str, AnyCard],
    ):
        self._generator = generator
        self._ResearchAgentCls = research_agent_cls
        self.export_file = export_file
        self.gam_store_dir = gam_store_dir
        self.checkpoint_dir = checkpoint_dir
        self.enable_bm25 = enable_bm25
        self.allowed_gam_tools = allowed_gam_tools
        self.gam_top_k_by_tool = gam_top_k_by_tool
        self.gam_pipeline_mode = gam_pipeline_mode
        self._memory_cards_ref = memory_cards_ref

    def load_or_create_retriever(self) -> Any:
        """Build a GAM ResearchAgent from exported cards or in-memory cards."""
        if self._generator is None or self._ResearchAgentCls is None:
            raise RuntimeError(
                "Generator is not available. Cannot create GAM research agent."
            )
        try:
            from gigaevo.memory.shared_memory.amem_gam_retriever import (
                build_gam_store,
                build_retrievers,
                load_amem_records,
            )
        except Exception as exc:
            raise RuntimeError(f"GAM helper modules are unavailable: {exc}") from exc

        self.gam_store_dir.mkdir(parents=True, exist_ok=True)
        if self.export_file.exists():
            records = load_amem_records(self.export_file)
        else:
            records = [c.model_dump() for c in self._memory_cards_ref.values()]

        memory_store, page_store, added = build_gam_store(records, self.gam_store_dir)
        logger.info(
            "[Memory] Loaded {} cards, added {} new pages.", len(records), added
        )

        retrievers = build_retrievers(
            page_store,
            self.gam_store_dir / "indexes",
            self.checkpoint_dir / "chroma",
            enable_bm25=self.enable_bm25,
            allowed_tools=sorted(self.allowed_gam_tools),
        )
        retrievers = {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self.allowed_gam_tools
        }
        if not retrievers:
            logger.info(
                "[Memory] No GAM retrievers enabled after applying allowed_gam_tools. "
                "GAM agentic search is disabled."
            )
            return None
        return self._ResearchAgentCls(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=self._generator,
            max_iters=3,
            allowed_tools=sorted(self.allowed_gam_tools),
            top_k_by_tool=self.gam_top_k_by_tool,
            pipeline_mode=self.gam_pipeline_mode,
        )
