"""Card normalization, conversion, and export utilities.

Pure data-transformation functions with no dependency on AmemGamMemory
instance state. Extracted from memory.py for cleaner module structure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.memory.context import ContextualGain
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    MemoryCard,
    ProgramCard,
)
from gigaevo.memory.shared_memory.utils import (
    _str_or_empty,
    _to_list,
    dedupe_keep_order,
)
from gigaevo.memory.utils import to_float

_ENTITY_NAME_MAX_LENGTH = 255


def normalize_delta_best(value: Any, *, lower_is_better: bool) -> float:
    """Producer-normalized Δbest: positive ALWAYS = improvement.

    For lower-is-better metrics (loss, error), flips the raw child−parent
    delta so an improvement is reported as positive.
    """
    raw = to_float(value, default=0.0) or 0.0
    return -raw if lower_is_better else raw


class MemoryNoteProtocol(Protocol):
    """Structural type for A-MEM MemoryNote objects."""

    id: str
    content: str
    keywords: list[str]
    links: list[str]
    retrieval_count: int
    timestamp: str
    last_accessed: str
    context: str
    evolution_history: list[Any]
    category: str
    tags: list[str]
    strategy: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_GAM_TOOLS = {
    "vector",
    "vector_description",
    "vector_task_description",
    "vector_explanation_summary",
    "vector_description_explanation_summary",
    "vector_description_task_description_summary",
}

ALLOWED_GAM_TOOLS = {
    "keyword",
    "page_index",
    *VECTOR_GAM_TOOLS,
}

DEFAULT_GAM_TOP_K_BY_TOOL = {
    "keyword": 5,
    "vector": 5,
    "vector_description": 5,
    "vector_task_description": 5,
    "vector_explanation_summary": 5,
    "vector_description_explanation_summary": 5,
    "vector_description_task_description_summary": 5,
    "page_index": 5,
}


# ---------------------------------------------------------------------------
# Card normalization
# ---------------------------------------------------------------------------


class RawCardRecord(BaseModel):
    """Boundary envelope for one raw card payload (JSON dict or API concept).

    Validates and coerces every loosely-typed field once, at the boundary;
    `to_card()` is the only raw-payload → AnyCard path. Legacy alias keys
    (content/context/context_summary) are kept as separate fields and
    resolved against their canonical counterparts in `to_card()`.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(
        default="",
        description="Card id from the payload; empty defers to to_card()'s fallback_id.",
    )
    category: str = Field(
        default="general",
        description="Card category; 'program' dispatches to ProgramCard.",
    )
    program_id: str = Field(
        default="",
        description="Source program id; non-empty dispatches to ProgramCard.",
    )
    description: str = Field(
        default="", description="Card text — the idea or program summary."
    )
    explanation_summary: str = Field(
        default="", description="One-line condensed reason the card's lever works."
    )
    content: str = Field(
        default="", description="Legacy alias of description; loses ties to it."
    )
    task_description: str = Field(
        default="", description="Task description active when the card was produced."
    )
    context: str = Field(
        default="", description="Legacy alias of task_description; loses ties to it."
    )
    task_description_summary: str = Field(
        default="", description="Condensed form of the task description."
    )
    context_summary: str = Field(
        default="",
        description="Legacy alias of task_description_summary; loses ties to it.",
    )
    code: str = Field(
        default="", description="Program source code (program cards only)."
    )
    fitness: float | None = Field(
        default=None,
        description="Program fitness; None when absent or unparsable.",
    )
    programs: list[Any] = Field(
        default_factory=list,
        description="Ids of programs that exhibited the idea.",
    )
    keywords: list[Any] = Field(default_factory=list, description="Search keywords.")
    gain_events: list[ContextualGain] | None = Field(
        default=None,
        description="Use-attributed, base-relative gain events the card earned.",
    )

    @field_validator(
        "id",
        "description",
        "explanation_summary",
        "content",
        "task_description",
        "context",
        "task_description_summary",
        "context_summary",
        "code",
        mode="before",
    )
    @classmethod
    def coerce_text(cls, value: Any) -> str:
        return str(value or "")

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, value: Any) -> str:
        return str(value or "general")

    @field_validator("program_id", mode="before")
    @classmethod
    def coerce_program_id(cls, value: Any) -> str:
        return _str_or_empty(value)

    @field_validator("fitness", mode="before")
    @classmethod
    def coerce_fitness(cls, value: Any) -> float | None:
        return to_float(value, default=None)

    @field_validator("programs", "keywords", mode="before")
    @classmethod
    def coerce_list(cls, value: Any) -> list[Any]:
        return _to_list(value)

    @field_validator("gain_events", mode="before")
    @classmethod
    def coerce_gain_events(cls, value: Any) -> Any:
        return value if isinstance(value, list) else None

    def to_card(self, fallback_id: str | None = None) -> AnyCard:
        """Build the typed card, resolving alias keys and program dispatch."""
        card_id = self.id or fallback_id or ""
        description = self.description or self.content
        task_description = self.task_description or self.context
        task_description_summary = self.task_description_summary or self.context_summary

        if self.category == "program" or self.program_id:
            # coerce_category turns a missing label into "general"; a program
            # card with no explicit category is canonically "program".
            category = self.category if self.category != "general" else "program"
            return ProgramCard(
                id=card_id,
                category=category,
                program_id=self.program_id,
                task_description=task_description,
                task_description_summary=task_description_summary,
                description=description,
                explanation_summary=self.explanation_summary,
                fitness=self.fitness,
                code=self.code,
                keywords=self.keywords,
                gain_events=self.gain_events,
            )

        return MemoryCard(
            id=card_id,
            category=self.category,
            description=description,
            explanation_summary=self.explanation_summary,
            task_description=task_description,
            task_description_summary=task_description_summary,
            programs=self.programs,
            keywords=self.keywords,
            gain_events=self.gain_events,
        )


def normalize_memory_card(
    card: dict[str, Any] | AnyCard | None = None,
    fallback_id: str | None = None,
) -> AnyCard:
    """Normalize raw input into a typed Pydantic card model.

    Returns:
        ProgramCard if category="program" or program_id is truthy.
        MemoryCard otherwise.
    """
    if isinstance(card, (MemoryCard, ProgramCard)):
        return card
    return RawCardRecord.model_validate(card or {}).to_card(fallback_id)


# ---------------------------------------------------------------------------
# Memory note ↔ card conversion
# ---------------------------------------------------------------------------


def memory_note_to_card(
    memory_note: MemoryNoteProtocol | None,
    base_card: dict[str, Any] | None = None,
    memory_id: str | None = None,
) -> AnyCard:
    """Convert an A-MEM MemoryNote into a normalized card model.

    The base_card payload wins field-by-field; the note only fills gaps.
    """
    mem_id = (memory_note.id if memory_note is not None else None) or memory_id
    card = normalize_memory_card(base_card, fallback_id=mem_id)
    if memory_note is None:
        return card

    updates: dict[str, Any] = {
        "id": str(mem_id or card.id),
        "category": str(card.category or memory_note.category or "general"),
        "description": str(card.description or memory_note.content or ""),
        "task_description": str(card.task_description or memory_note.context or ""),
    }

    if isinstance(card, ProgramCard):
        return card.model_copy(update=updates)

    updates["keywords"] = _to_list(memory_note.keywords or [])

    return card.model_copy(update=updates)


def export_memories_jsonl(
    memory_system: Any,
    memory_ids: list[str],
    out_path: Path,
    card_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Export A-MEM memories to JSONL for GAM retriever consumption."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card_overrides = card_overrides or {}

    unique_ids = list(dict.fromkeys(memory_ids))
    tmp_path = out_path.with_suffix(f".{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        for memory_id in unique_ids:
            memory_note = memory_system.read(memory_id)
            base_card = card_overrides.get(memory_id)
            if memory_note is None and base_card is None:
                continue
            record = memory_note_to_card(
                memory_note, base_card=base_card, memory_id=memory_id
            )
            file_obj.write(json.dumps(record.model_dump(), ensure_ascii=True) + "\n")
    os.replace(str(tmp_path), str(out_path))


# ---------------------------------------------------------------------------
# Static classification helpers
# ---------------------------------------------------------------------------


def card_to_concept_content(card: AnyCard) -> dict[str, Any]:
    """Convert a Pydantic card model to the API concept content format."""
    if isinstance(card, ProgramCard):
        return {
            "id": card.id,
            "category": "program",
            "program_id": card.program_id,
            "task_description": card.task_description,
            "task_description_summary": card.task_description_summary,
            "description": card.description,
            "explanation_summary": card.explanation_summary,
            "fitness": card.fitness,
            "code": card.code,
            "keywords": dedupe_keep_order(list(card.keywords)),
            "gain_events": (
                [g.model_dump() for g in card.gain_events] if card.gain_events else None
            ),
        }

    return {
        "id": card.id,
        "category": card.category,
        "program_id": "",
        "fitness": None,
        "task_description": card.task_description,
        "task_description_summary": card.task_description_summary,
        "description": card.description,
        "explanation_summary": card.explanation_summary,
        "code": "",
        "programs": dedupe_keep_order(list(card.programs)),
        "keywords": dedupe_keep_order(list(card.keywords)),
        "gain_events": (
            [g.model_dump() for g in card.gain_events] if card.gain_events else None
        ),
    }


def build_entity_meta(card: AnyCard) -> tuple[str, list[str], str]:
    """Build API entity metadata (name, tags, when_to_use) from a card."""
    description = card.description.strip()
    task_description = card.task_description.strip()
    task_description_summary = card.task_description_summary.strip()

    name_seed = (
        description or task_description_summary or task_description or "memory card"
    )
    name = f"{card.id}: {name_seed}" if card.id else name_seed
    name = name[:_ENTITY_NAME_MAX_LENGTH]

    tags = dedupe_keep_order(
        [
            card.category.strip(),
            *[str(x).strip() for x in card.keywords],
        ]
    )

    when_to_use_parts = dedupe_keep_order(
        [
            task_description_summary,
            task_description,
            description,
            " ".join(str(x) for x in card.keywords).strip(),
        ]
    )
    when_to_use = " | ".join(when_to_use_parts)

    return name, tags, when_to_use


def is_program_card(card: AnyCard) -> bool:
    """Check if a card is a program card."""
    return isinstance(card, ProgramCard)


def normalize_allowed_gam_tools(allowed_gam_tools: list[str] | None) -> set[str]:
    """Normalize GAM tool list, expanding 'vector' to all vector variants."""
    if not allowed_gam_tools:
        return set(ALLOWED_GAM_TOOLS)

    normalized = {str(tool).strip() for tool in allowed_gam_tools if str(tool).strip()}
    valid = {tool for tool in normalized if tool in ALLOWED_GAM_TOOLS}
    if "vector" in valid:
        valid.update(VECTOR_GAM_TOOLS)
    return valid or set(ALLOWED_GAM_TOOLS)


def normalize_gam_top_k_by_tool(
    gam_top_k_by_tool: dict[str, int] | None,
) -> dict[str, int]:
    """Normalize per-tool top_k limits, falling back to defaults."""
    normalized = dict(DEFAULT_GAM_TOP_K_BY_TOOL)
    if not isinstance(gam_top_k_by_tool, dict):
        return normalized

    for tool_name, raw_value in gam_top_k_by_tool.items():
        tool = str(tool_name).strip()
        if tool not in normalized:
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            normalized[tool] = value
    return normalized


# ---------------------------------------------------------------------------
# API concept ↔ card conversion
# ---------------------------------------------------------------------------


def concept_to_card(concept_content: dict[str, Any], fallback_id: str) -> AnyCard:
    """Convert an API concept content dict to a normalized memory card.

    The concept payload uses the same field names as raw card payloads, so
    the RawCardRecord envelope handles it directly.
    """
    return normalize_memory_card(concept_content, fallback_id=fallback_id)


def note_metadata(note: MemoryNoteProtocol) -> dict[str, Any]:
    """Extract metadata dict from an A-MEM MemoryNote."""
    return {
        "id": note.id,
        "content": note.content,
        "keywords": note.keywords,
        "links": note.links,
        "retrieval_count": note.retrieval_count,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "context": note.context,
        "evolution_history": note.evolution_history,
        "category": note.category,
        "tags": note.tags,
        "strategy": note.strategy,
    }
