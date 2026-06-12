from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import AnyCard, MemoryCard, ProgramCard


class GamRetriever:
    """Thin seam over a GAM memory backend: ``research`` runs the red-agent
    retrieval+selection pass; ``get_card`` resolves a shortlisted id, fail-to-None.
    The retrieval knobs ride along for
    Hydra composition; numeric values live in config/memory/retriever/gam.yaml."""

    def __init__(
        self,
        backend: Any = None,
        *,
        enable_bm25: bool = False,
        pipeline_mode: str = "default",
        allowed_tools: Sequence[str] = (),
        top_k_by_tool: Mapping[str, int] | None = None,
        max_iters: int | None = None,
    ) -> None:
        self.backend = backend
        self.enable_bm25 = enable_bm25
        self.pipeline_mode = pipeline_mode
        self.allowed_tools = list(allowed_tools)
        self.top_k_by_tool = dict(top_k_by_tool or {})
        self.max_iters = max_iters

    def bind(self, backend: Any) -> GamRetriever:
        """Attach the lazily-built memory backend and return self.

        Hydra instantiates the retriever at config-resolution time, before any
        backend exists; the provider binds the backend on first card selection.
        """
        self.backend = backend
        return self

    def research(self, query: str, *, planning_request: str | None = None) -> Any:
        if self.backend is None:
            raise RuntimeError(
                "GamRetriever.research called before bind(); no backend attached"
            )
        return self.backend.research(query, planning_request=planning_request)

    def get_card(self, card_id: str) -> AnyCard | None:
        """Resolve a shortlisted card id to a typed card, fail-to-None.

        Typed backends pass through; legacy backends that still return raw
        dicts are normalized here — this is the typed boundary of the read
        pipeline, so a corrupt persisted card degrades to None instead of
        sinking the selection."""
        if self.backend is None:
            return None
        try:
            raw = self.backend.get_card(card_id)
            if raw is None or isinstance(raw, MemoryCard | ProgramCard):
                return raw
            return normalize_memory_card(raw, fallback_id=card_id)
        except Exception as exc:
            logger.warning("[Memory][Retriever] get_card({}) failed: {}", card_id, exc)
            return None
