"""Card normalization, conversion, and export utilities.

Pure data-transformation functions with no dependency on AmemGamMemory
instance state. Extracted from memory.py for cleaner module structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typing import Protocol


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


from gigaevo.memory.shared_memory.utils import (
    _safe_get,
    _str_or_empty,
    _to_float,
    _to_int,
    _to_list,
    dedupe_keep_order,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_STRATEGIES = {"exploration", "exploitation", "hybrid"}

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

ALLOWED_GAM_PIPELINE_MODES = {"default", "experimental"}

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


def normalize_memory_card(
    card: dict[str, Any] | None = None,
    fallback_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a raw card dict into the canonical memory card shape.

    Two output shapes:
    - Program cards (category="program" or program_id truthy): 9 keys
    - General cards: 15 keys including explanation, keywords, etc.
    """
    raw = dict(card or {})
    category = str(raw.get("category") or "general")
    program_id = _str_or_empty(raw.get("program_id"))
    if category == "program" or program_id:
        return {
            "id": str(raw.get("id") or fallback_id or ""),
            "category": "program",
            "program_id": program_id,
            "task_description": str(
                raw.get("task_description") or raw.get("context") or ""
            ),
            "task_description_summary": str(
                raw.get("task_description_summary") or raw.get("context_summary") or ""
            ),
            "description": str(raw.get("description") or raw.get("content") or ""),
            "fitness": _to_float(raw.get("fitness"), default=None),
            "code": str(raw.get("code") or ""),
            "connected_ideas": _to_list(raw.get("connected_ideas")),
        }

    explanation = raw.get("explanation")
    if not isinstance(explanation, dict):
        explanation = {}

    return {
        "id": str(raw.get("id") or fallback_id or ""),
        "category": category,
        "description": str(raw.get("description") or raw.get("content") or ""),
        "task_description": str(
            raw.get("task_description") or raw.get("context") or ""
        ),
        "task_description_summary": str(
            raw.get("task_description_summary") or raw.get("context_summary") or ""
        ),
        "strategy": str(raw.get("strategy") or ""),
        "last_generation": _to_int(raw.get("last_generation"), default=0),
        "programs": _to_list(raw.get("programs")),
        "aliases": _to_list(raw.get("aliases")),
        "keywords": _to_list(raw.get("keywords")),
        "evolution_statistics": (
            raw.get("evolution_statistics")
            if isinstance(raw.get("evolution_statistics"), dict)
            else {}
        ),
        "explanation": {
            "explanations": _to_list(explanation.get("explanations")),
            "summary": str(explanation.get("summary") or ""),
        },
        "works_with": _to_list(raw.get("works_with")),
        "links": _to_list(raw.get("links")),
        "usage": raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
    }


# ---------------------------------------------------------------------------
# Memory note ↔ card conversion
# ---------------------------------------------------------------------------


def memory_to_card(
    memory_note: MemoryNoteProtocol | None,
    base_card: dict[str, Any] | None = None,
    memory_id: str | None = None,
) -> dict[str, Any]:
    """Convert an A-MEM MemoryNote into a normalized card dict."""
    mem_id = _safe_get(memory_note, "id", None) or memory_id
    card = normalize_memory_card(base_card, fallback_id=mem_id)
    if memory_note is None:
        return card

    card["id"] = str(mem_id or card["id"])
    card["category"] = str(
        card.get("category") or _safe_get(memory_note, "category", None) or "general"
    )
    card["description"] = str(
        card.get("description") or _safe_get(memory_note, "content", "")
    )
    card["task_description"] = str(
        card.get("task_description") or _safe_get(memory_note, "context", "")
    )
    if str(card.get("category") or "").strip().lower() == "program":
        return card
    card["strategy"] = str(
        card.get("strategy") or _safe_get(memory_note, "strategy", "")
    )
    card["keywords"] = _to_list(_safe_get(memory_note, "keywords", []) or [])

    if not card.get("links"):
        card["links"] = (
            _safe_get(memory_note, "links", None)
            or _safe_get(memory_note, "linked_memories", None)
            or _safe_get(memory_note, "linked_ids", None)
            or _safe_get(memory_note, "relations", None)
            or []
        )
    card["links"] = _to_list(card["links"])

    return card


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
    with out_path.open("w", encoding="utf-8") as file_obj:
        for memory_id in unique_ids:
            memory_note = memory_system.read(memory_id)
            base_card = card_overrides.get(memory_id)
            if memory_note is None and base_card is None:
                continue
            record = memory_to_card(
                memory_note, base_card=base_card, memory_id=memory_id
            )
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Static classification helpers
# ---------------------------------------------------------------------------


def card_to_concept_content(card: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized card dict to the API concept content format."""
    if is_program_card(card):
        return {
            "id": str(card.get("id") or ""),
            "category": "program",
            "program_id": _str_or_empty(card.get("program_id")),
            "task_description": str(card.get("task_description") or ""),
            "task_description_summary": str(
                card.get("task_description_summary") or ""
            ),
            "description": str(card.get("description") or ""),
            "fitness": _to_float(card.get("fitness"), default=None),
            "code": str(card.get("code") or ""),
            "connected_ideas": _to_list(card.get("connected_ideas")),
        }

    explanation = card.get("explanation")
    if isinstance(explanation, dict):
        explanation_text = str(explanation.get("summary") or "")
    else:
        explanation_text = str(explanation or "")

    strategy = str(card.get("strategy") or "").strip().lower() or None
    if strategy not in ALLOWED_STRATEGIES:
        strategy = None

    evolution_statistics = card.get("evolution_statistics")
    if not isinstance(evolution_statistics, dict):
        evolution_statistics = None

    usage = card.get("usage")
    if not isinstance(usage, dict):
        usage = None

    return {
        "id": str(card.get("id") or ""),
        "category": str(card.get("category") or "general"),
        "program_id": _str_or_empty(card.get("program_id")),
        "fitness": _to_float(card.get("fitness"), default=None),
        "task_description": str(card.get("task_description") or ""),
        "task_description_summary": str(card.get("task_description_summary") or ""),
        "description": str(card.get("description") or ""),
        "code": str(card.get("code") or ""),
        "connected_ideas": _to_list(card.get("connected_ideas")),
        "explanation": explanation_text,
        "strategy": strategy,
        "keywords": dedupe_keep_order(list(card.get("keywords") or [])),
        "evolution_statistics": evolution_statistics,
        "works_with": dedupe_keep_order(list(card.get("works_with") or [])),
        "links": dedupe_keep_order(list(card.get("links") or [])),
        "usage": usage,
    }


def build_entity_meta(card: dict[str, Any]) -> tuple[str, list[str], str]:
    """Build API entity metadata (name, tags, when_to_use) from a card."""
    card_id = str(card.get("id") or "")
    description = str(card.get("description") or "").strip()
    task_description = str(card.get("task_description") or "").strip()
    task_description_summary = str(
        card.get("task_description_summary") or ""
    ).strip()

    explanation = card.get("explanation")
    if isinstance(explanation, dict):
        explanation_summary = str(explanation.get("summary") or "").strip()
    else:
        explanation_summary = str(explanation or "").strip()

    name_seed = (
        description or task_description_summary or task_description or "memory card"
    )
    name = f"{card_id}: {name_seed}" if card_id else name_seed
    name = name[:255]

    tags = dedupe_keep_order(
        [
            str(card.get("category") or "").strip(),
            str(card.get("strategy") or "").strip(),
            *[str(x).strip() for x in (card.get("keywords") or [])],
        ]
    )

    when_to_use_parts = dedupe_keep_order(
        [
            task_description_summary,
            task_description,
            description,
            explanation_summary,
            " ".join([str(x) for x in (card.get("keywords") or [])]).strip(),
        ]
    )
    when_to_use = " | ".join(when_to_use_parts)

    return name, tags, when_to_use


def is_program_card(card: dict[str, Any]) -> bool:
    """Check if a card dict represents a program card."""
    if str(card.get("category") or "").strip().lower() == "program":
        return True
    return bool(_str_or_empty(card.get("program_id")).strip())


def normalize_allowed_gam_tools(allowed_gam_tools: list[str] | None) -> set[str]:
    """Normalize GAM tool list, expanding 'vector' to all vector variants."""
    if not allowed_gam_tools:
        return set(ALLOWED_GAM_TOOLS)

    normalized = {
        str(tool).strip() for tool in allowed_gam_tools if str(tool).strip()
    }
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
        if value > 0:
            normalized[tool] = value
    return normalized


def normalize_gam_pipeline_mode(gam_pipeline_mode: str | None) -> str:
    """Normalize pipeline mode to 'default' or 'experimental'."""
    mode = str(gam_pipeline_mode or "default").strip().lower()
    if mode in ALLOWED_GAM_PIPELINE_MODES:
        return mode
    return "default"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def concept_to_card(
    concept_content: dict[str, Any], fallback_id: str
) -> dict[str, Any]:
    """Convert an API concept content dict to a normalized memory card."""
    return normalize_memory_card(
        {
            "id": concept_content.get("id") or fallback_id,
            "category": concept_content.get("category") or "general",
            "program_id": concept_content.get("program_id") or "",
            "fitness": concept_content.get("fitness"),
            "description": concept_content.get("description") or "",
            "task_description": concept_content.get("task_description") or "",
            "task_description_summary": concept_content.get(
                "task_description_summary"
            )
            or "",
            "code": concept_content.get("code") or "",
            "connected_ideas": concept_content.get("connected_ideas") or [],
            "strategy": concept_content.get("strategy") or "",
            "keywords": concept_content.get("keywords") or [],
            "evolution_statistics": concept_content.get("evolution_statistics") or {},
            "explanation": {
                "explanations": [],
                "summary": concept_content.get("explanation") or "",
            },
            "works_with": concept_content.get("works_with") or [],
            "links": concept_content.get("links") or [],
            "usage": concept_content.get("usage") or {},
        },
        fallback_id=fallback_id,
    )


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


def format_search_results(query: str, cards: list[dict[str, Any]]) -> str:
    """Format search results as numbered card list for MemorySelectorAgent parsing."""
    lines = [f"Query: {query}", "", "Top relevant memory cards:"]
    for idx, card in enumerate(cards, start=1):
        card_id = str(card.get("id") or "")
        category = str(card.get("category") or "general")
        description = str(card.get("description") or "").strip()
        lines.append(f"{idx}. {card_id} [{category}] {description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GigaEvoMemoryBase:
    """Abstract base for memory backends."""

    def save(self, data: str) -> str:
        raise NotImplementedError

    def search(self, query: str) -> str:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError
