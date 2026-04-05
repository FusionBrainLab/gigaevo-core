"""Agentic runtime dependency resolution for the memory system.

Replaces the try/except lazy-import pattern in AmemGamMemory._load_agentic_classes()
with a clean factory that returns a typed bundle of resolved classes, or None.

Also provides factory functions for LLM/generator and A-MEM storage init.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.protocols import (
    AgenticMemoryProtocol,
    GeneratorProtocol,
    LLMServiceProtocol,
)


class AgenticRuntime(BaseModel):
    """Resolved agentic dependencies (A-MEM + GAM classes).

    Passed to AmemGamMemory at construction time.
    In tests, use FakeAgenticRuntime with fake classes.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

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
        from gigaevo.memory.A_mem.agentic_memory.memory_system import (
            AgenticMemorySystem as _AgenticMemorySystem,
        )
        from gigaevo.memory.A_mem.agentic_memory.memory_system import (
            MemoryNote as _MemoryNote,
        )
        from gigaevo.memory.GAM_root.gam import ResearchAgent as _ResearchAgent
        from gigaevo.memory.GAM_root.gam.generator import (
            AMemGenerator as _AMemGenerator,
        )
    except Exception as exc:
        logger.info(
            "[Memory] Agentic runtime dependencies unavailable: {}. "
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


def init_llm_and_generator(
    *,
    generator_cls: type[Any] | None,
    dedup_enabled: bool,
) -> tuple[LLMServiceProtocol | None, GeneratorProtocol | None]:
    """Create LLM service + generator from environment config.

    Returns ``(None, None)`` when deps are unavailable.
    """
    import gigaevo.memory.config as env_config
    from gigaevo.memory.openai_inference import OpenAIInferenceService
    from gigaevo.memory.shared_memory.card_conversion import DEFAULT_MODEL_NAME

    if generator_cls is None and not dedup_enabled:
        return None, None

    api_key = env_config.OPENAI_API_KEY
    if not api_key and env_config.LLM_BASE_URL:
        api_key = "EMPTY"

    if not api_key:
        logger.info(
            "[Memory] OPENAI_API_KEY/OPENROUTER_API_KEY is not set. "
            "Agentic retrieval is disabled; API full-text fallback is available."
        )
        return None, None

    try:
        llm_service: LLMServiceProtocol = OpenAIInferenceService(
            model_name=env_config.OPENROUTER_MODEL_NAME or DEFAULT_MODEL_NAME,
            api_key=api_key,
            base_url=env_config.LLM_BASE_URL,
            temperature=0.0,
            max_tokens=0,
            reasoning=env_config.OPENROUTER_REASONING,
        )
        if generator_cls is None:
            return llm_service, None
        generator: GeneratorProtocol = generator_cls({"llm_service": llm_service})
        return llm_service, generator
    except Exception as exc:
        logger.warning("[Memory] Could not initialize LLM/generator: {}", exc)
        return None, None


def init_agentic_storage(
    *,
    llm_service: LLMServiceProtocol | None,
    system_cls: type[Any] | None,
    checkpoint_dir: Path,
    enable_evolution: bool,
) -> AgenticMemoryProtocol | None:
    """Create the A-MEM agentic memory system (Chroma vector store).

    Returns ``None`` when deps are unavailable.
    """
    import gigaevo.memory.config as env_config

    if llm_service is None or system_cls is None:
        return None
    try:
        return system_cls(
            model_name=env_config.AMEM_EMBEDDING_MODEL_NAME,
            llm_backend="custom",
            llm_service=llm_service,
            chroma_persist_dir=checkpoint_dir / "chroma",
            chroma_collection_name="memories",
            use_gam_card_document=True,
            enable_evolution=enable_evolution,
        )
    except Exception as exc:
        logger.warning("[Memory] Could not initialize AgenticMemorySystem: {}", exc)
        return None
