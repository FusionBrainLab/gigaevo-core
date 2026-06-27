"""GAM retriever script that loads A-mem exports and runs research via LangChain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.memory._vendor.GAM_root.gam import (
    ChromaRetriever,
    IndexRetriever,
    InMemoryMemoryStore,
    InMemoryPageStore,
    ResearchAgent,
)
from gigaevo.memory._vendor.GAM_root.gam.generator import AMemGenerator
from gigaevo.memory._vendor.GAM_root.gam.schemas import Page
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
        parts.append(f"programs: {card.programs}")
    efficacy = format_card_efficacy(card)
    if efficacy:
        parts.append(efficacy)
    return "\n".join(parts)


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
    seen_ids: set[str] = set()
    for rec in records:
        typed = normalize_memory_card(rec)
        rid = typed.id.strip()
        if rid and rid in seen_ids:
            continue
        if rid:
            seen_ids.add(rid)
        card = render_card_text(typed)
        abstract = typed.description or card
        memory_store.add(abstract)
        header = f"[A-MEM] {rid}" if rid else "[A-MEM]"
        # Page meta keeps the raw record: the vendor GAM page contract.
        next_pages.append(
            Page(header=header, content=card, meta={"amem_id": rid, "amem": rec})
        )
        if rid and rid not in existing_ids:
            added += 1

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


def main():
    from langchain_openai import ChatOpenAI

    from gigaevo.llm.models import MultiModelRouter

    parser = argparse.ArgumentParser(
        description="Run a GAM research query over an A-mem export."
    )
    parser.add_argument(
        "--export-file", type=Path, default=Path("amem_exports/amem_memories.jsonl")
    )
    parser.add_argument("--model", default="google/gemini-3-flash-preview")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="Defaults to $OPENROUTER_API_KEY.",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--question",
        default="What changes improved the primary fitness metric the most and why?",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key not given and OPENROUTER_API_KEY is not set")
    if not args.export_file.exists():
        raise FileNotFoundError(f"A-mem export not found: {args.export_file}")

    records = load_amem_records(args.export_file)
    if not records:
        raise RuntimeError("A-mem export is empty.")

    store_dir = Path(__file__).resolve().parents[1] / "gam_shared" / "amem_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    memory_store, page_store, added = build_gam_store(records, store_dir)
    logger.info(
        "[Memory][AmemGamRetriever] Loaded {} A-mem records, added {} new pages.",
        len(records),
        added,
    )
    logger.info("[Memory][AmemGamRetriever] LLM: {} at {}", args.model, args.base_url)

    llm_service = MultiModelRouter(
        models=[
            ChatOpenAI(
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=0.0,
            )
        ],
        probabilities=[1.0],
        name="memory",
    )
    generator = AMemGenerator({"llm_service": llm_service})

    chroma_dir = Path(__file__).resolve().parents[1] / "chroma"
    retrievers = build_retrievers(
        page_store,
        store_dir / "indexes",
        chroma_dir,
        embedding_model_name=args.embedding_model,
    )
    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=generator,
        max_iters=3,
    )

    logger.info("[Memory][AmemGamRetriever] Research question: {}", args.question)
    result = research_agent.research(args.question)
    logger.info(
        "[Memory][AmemGamRetriever] Research result:\n{}", result.integrated_memory
    )


if __name__ == "__main__":
    main()
