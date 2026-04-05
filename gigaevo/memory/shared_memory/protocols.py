"""Structural types (protocols) for agentic dependencies (A-MEM, GAM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gigaevo.memory.shared_memory.card_conversion import MemoryNoteProtocol


class LLMServiceProtocol(Protocol):
    """Structural type for OpenAIInferenceService."""

    def generate(self, data: str) -> tuple[str, Any, int | None, float | None]: ...


class AgenticMemoryProtocol(Protocol):
    """Structural type for AgenticMemorySystem."""

    memories: dict[str, Any]
    retriever: Any

    def read(self, memory_id: str) -> MemoryNoteProtocol | None: ...
    def add_note(self, content: str, **kwargs: Any) -> str: ...
    def update(self, memory_id: str, **kwargs: Any) -> bool: ...
    def delete(self, memory_id: str) -> bool: ...
    def analyze_content(self, content: str) -> dict[str, Any]: ...
    def _document_for_note(self, note: MemoryNoteProtocol) -> str: ...


@dataclass
class ResearchOutput:
    """Return type of ResearchAgent.research()."""

    integrated_memory: str = ""
    raw_memory: dict[str, Any] | None = None


class ResearchAgentProtocol(Protocol):
    """Structural type for GAM ResearchAgent."""

    def research(
        self, request: str, memory_state: str | None = None
    ) -> ResearchOutput: ...


class GeneratorProtocol(Protocol):
    """Structural type for AMemGenerator."""

    def generate_single(
        self, prompt: str | None = None, **kwargs: Any
    ) -> dict[str, Any]: ...
