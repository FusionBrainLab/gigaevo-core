"""Agentic runtime dependency resolution for the memory system.

Replaces the try/except lazy-import pattern in AmemGamMemory._load_agentic_classes()
with a clean factory that returns a typed bundle of resolved classes, or None.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class AgenticRuntime:
    """Resolved agentic dependencies (A-MEM + GAM classes).

    Passed to AmemGamMemory at construction time.
    In tests, use FakeAgenticRuntime with fake classes.
    """

    memory_system_cls: type
    memory_note_cls: type
    research_agent_cls: type
    generator_cls: type


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
