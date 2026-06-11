"""Agentic runtime dependency resolution for the memory system.

Replaces the try/except lazy-import pattern in AmemGamMemory._load_agentic_classes()
with a clean factory that returns a typed bundle of resolved classes, or None.

Also provides the A-MEM storage init factory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.protocols import (
    AgenticMemoryProtocol,
    LLMServiceProtocol,
)


def configure_huggingface_timeouts() -> None:
    # HF hub's 10s defaults flake when pulling embedding models on shared NFS boxes.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")


class AgenticRuntime(BaseModel):
    """Resolved agentic dependencies (A-MEM + GAM classes).

    Passed to AmemGamMemory at construction time.
    In tests, use FakeAgenticRuntime with fake classes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    memory_system_cls: type[Any]
    memory_note_cls: type[Any]
    research_agent_cls: type[Any]
    generator_cls: type[Any]


def load_agentic_runtime() -> AgenticRuntime | None:
    """Try to import A-MEM + GAM dependencies.

    Returns ``AgenticRuntime`` if all deps are available, ``None`` otherwise.
    This is the single place where agentic imports are attempted.
    """
    try:
        from gigaevo.memory._vendor.A_mem.agentic_memory.memory_system import (
            AgenticMemorySystem as _AgenticMemorySystem,
        )
        from gigaevo.memory._vendor.A_mem.agentic_memory.memory_system import (
            MemoryNote as _MemoryNote,
        )
        from gigaevo.memory._vendor.GAM_root.gam import ResearchAgent as _ResearchAgent
        from gigaevo.memory._vendor.GAM_root.gam.generator import (
            AMemGenerator as _AMemGenerator,
        )
    except ImportError as exc:
        logger.info(
            "[Memory][Runtime] Agentic runtime dependencies unavailable: {}. "
            "Falling back to API full-text mode.",
            exc,
        )
        return None

    return AgenticRuntime(
        memory_system_cls=_AgenticMemorySystem,
        memory_note_cls=_MemoryNote,
        research_agent_cls=_ResearchAgent,
        generator_cls=_AMemGenerator,
    )


def init_agentic_storage(
    *,
    llm_service: LLMServiceProtocol | None,
    system_cls: type[Any] | None,
    checkpoint_dir: Path,
    enable_evolution: bool,
    embedding_model_name: str,
) -> AgenticMemoryProtocol | None:
    """Create the A-MEM agentic memory system (Chroma vector store).

    Returns ``None`` when deps are unavailable.
    """
    if llm_service is None or system_cls is None:
        return None
    configure_huggingface_timeouts()
    try:
        return system_cls(
            model_name=embedding_model_name,
            llm_backend="custom",
            llm_service=llm_service,
            chroma_persist_dir=checkpoint_dir / "chroma",
            chroma_collection_name="memories",
            use_gam_card_document=True,
            enable_evolution=enable_evolution,
        )
    except Exception as exc:
        logger.warning(
            "[Memory][Runtime] Could not initialize AgenticMemorySystem: {}", exc
        )
        return None
