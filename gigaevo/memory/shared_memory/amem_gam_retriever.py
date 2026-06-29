"""GAM store/retriever builders: load A-mem JSONL exports into GAM stores and
Chroma/index retrievers (consumed by ``gam_search``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.memory._vendor.GAM_root.gam import (
    ChromaRetriever,
    IndexRetriever,
    InMemoryMemoryStore,
    InMemoryPageStore,
)
from gigaevo.memory._vendor.GAM_root.gam.schemas import MemoryState, Page
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_search import (
    format_card_efficacy,
    topical_keywords,
)
from gigaevo.memory.shared_memory.models import AnyCard, ProgramCard


class GamPageMeta(BaseModel):
    """Our slice of the vendor ``Page.meta`` payload; extra keys stay vendor-owned."""

    model_config = ConfigDict(extra="ignore")

    amem_id: str = Field(
        default="", description="Bank card id the vendor page was built from."
    )

    @field_validator("amem_id", mode="before")
    @classmethod
    def coerce_to_string(cls, value: Any) -> str:
        return str(value or "")


def load_amem_records(path: Path) -> list[dict[str, Any]]:
    """Load A-MEM exported records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_card_text(record: dict[str, Any]) -> str:
    """Format one exported card record as multi-line text for the GAM page store."""
    return render_card_text(normalize_memory_card(record))


def render_card_text(card: AnyCard) -> str:
    """Render a typed card as the GAM corpus text (one ``field: value`` per line)."""
    parts = [
        f"description: {card.description}",
        f"task_description_summary: {card.task_description_summary}",
        f"task_description: {card.task_description}",
        f"category: {card.category}",
        f"keywords: {', '.join(topical_keywords(card.keywords))}",
    ]
    if isinstance(card, ProgramCard):
        parts += [
            f"program_id: {card.program_id}",
            f"fitness: {card.fitness if card.fitness is not None else ''}",
        ]
    else:
        parts.append(f"explanation_summary: {card.explanation_summary}")
        parts.append(f"programs: {card.programs}")
    efficacy = format_card_efficacy(card)
    if efficacy:
        parts.append(efficacy)
    return "\n".join(parts)


def render_card_abstract(card: AnyCard) -> str:
    """GAM long-term-memory abstract: the always-on planning-context summary.

    Idea cards carry the why-text so the planner's remembered memory is not
    description-only; program exemplars stay description-only, mirroring
    ``render_card_text`` (their explanation rides page metadata)."""
    base = (card.description or "").strip()
    if not base:
        return render_card_text(card)
    if isinstance(card, ProgramCard):
        return base
    why = (card.explanation_summary or "").strip()
    return f"{base} — {why}" if why else base


def build_gam_store(
    records: list[dict[str, Any]], store_dir: Path
) -> tuple[InMemoryMemoryStore, InMemoryPageStore, int]:
    """Build GAM memory and page stores from exported records.

    Returns:
        Tuple of (memory_store, page_store, newly_added_count).
    """
    memory_store = InMemoryMemoryStore(dir_path=str(store_dir))
    page_store = InMemoryPageStore(dir_path=str(store_dir))

    existing_pages = page_store.load()
    existing_ids = {
        GamPageMeta.model_validate(p.meta).amem_id.strip()
        for p in existing_pages
        if isinstance(p.meta, dict)
    }
    existing_ids.discard("")

    added = 0
    next_pages: list[Page] = []
    next_abstracts: list[str] = []
    seen_ids: set[str] = set()
    seen_abstracts: set[str] = set()
    for rec in records:
        typed = normalize_memory_card(rec)
        rid = typed.id.strip()
        if rid and rid in seen_ids:
            continue
        if rid:
            seen_ids.add(rid)
        card = render_card_text(typed)
        abstract = render_card_abstract(typed)
        if abstract and abstract not in seen_abstracts:
            seen_abstracts.add(abstract)
            next_abstracts.append(abstract)
        header = f"[A-MEM] {rid}" if rid else "[A-MEM]"
        # Page meta keeps the raw record: the vendor GAM page contract.
        next_pages.append(
            Page(header=header, content=card, meta={"amem_id": rid, "amem": rec})
        )
        if rid and rid not in existing_ids:
            added += 1

    # Overwrite (not append) so edited/removed cards leave no stale planning
    # abstract behind, mirroring the page store's full rebuild below.
    memory_store.save(MemoryState(abstracts=next_abstracts))
    page_store.save(next_pages)
    return memory_store, page_store, added


def build_retrievers(
    page_store: InMemoryPageStore,
    index_dir: Path,
    chroma_dir: Path,
    chroma_collection: str = "memories",
    allowed_tools: list[str] | set[str] | tuple[str, ...] | None = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Build retriever index from a page store.

    Returns:
        Mapping of tool name to retriever instance.
    """
    retrievers: dict[str, Any] = {}

    vector_tool_configs = {
        "vector": {
            "active_collections": [
                "description",
                "task_description",
                "explanation_summary",
                "description_explanation_summary",
                "description_task_description_summary",
            ],
            "source_label": "vector",
        },
        "vector_description": {
            "active_collections": ["description"],
            "source_label": "vector_description",
        },
        "vector_task_description": {
            "active_collections": ["task_description"],
            "source_label": "vector_task_description",
        },
        "vector_explanation_summary": {
            "active_collections": ["explanation_summary"],
            "source_label": "vector_explanation_summary",
        },
        "vector_description_explanation_summary": {
            "active_collections": ["description_explanation_summary"],
            "source_label": "vector_description_explanation_summary",
        },
        "vector_description_task_description_summary": {
            "active_collections": ["description_task_description_summary"],
            "source_label": "vector_description_task_description_summary",
        },
    }
    allowed = {str(tool).strip() for tool in (allowed_tools or []) if str(tool).strip()}
    if not allowed:
        allowed = {"page_index", *vector_tool_configs.keys()}

    if "page_index" in allowed:
        try:
            index_retriever = IndexRetriever(
                {"index_dir": str(index_dir / "page_index")}
            )
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            logger.debug("[Memory][AmemGamRetriever] Index retriever ready")
        except Exception as exc:
            logger.warning(
                "[Memory][AmemGamRetriever] Index retriever init failed: {}", exc
            )

    for tool_name, extra in vector_tool_configs.items():
        if tool_name not in allowed:
            continue
        try:
            chroma_config = {
                "persist_dir": str(chroma_dir),
                "collection_name": chroma_collection,
                "model_name": embedding_model_name,
                **extra,
            }
            retrievers[tool_name] = ChromaRetriever(chroma_config)
            logger.debug(
                "[Memory][AmemGamRetriever] Chroma retriever ready: {}", tool_name
            )
        except Exception as exc:
            logger.warning(
                "[Memory][AmemGamRetriever] Chroma retriever init for '{}' failed: {}",
                tool_name,
                exc,
            )

    return retrievers
