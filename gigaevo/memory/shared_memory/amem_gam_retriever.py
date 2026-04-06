"""
GAM retriever script that loads A-mem exports and uses OpenAI-style inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from gigaevo.memory._vendor.GAM_root.gam import (
    ChromaRetriever,
    IndexRetriever,
    InMemoryMemoryStore,
    InMemoryPageStore,
    ResearchAgent,
)
from gigaevo.memory._vendor.GAM_root.gam.generator import AMemGenerator
from gigaevo.memory._vendor.GAM_root.gam.schemas import Page
import gigaevo.memory.config as config
from gigaevo.memory.openai_inference import OpenAIInferenceService


def load_amem_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_card_text(record: dict[str, Any]) -> str:
    # Support both new and legacy A-mem export schemas.
    description = record.get("description") or record.get("content") or ""
    task_description = record.get("task_description") or record.get("context") or ""
    task_description_summary = record.get("task_description_summary") or ""
    category = record.get("category") or ""
    strategy = record.get("strategy") or ""
    keywords = ", ".join(record.get("keywords", []) or [])
    links = record.get("links", []) or []
    program_id = record.get("program_id") or ""
    fitness = record.get("fitness", "")
    connected_ideas = record.get("connected_ideas", []) or []
    last_generation = record.get("last_generation", "")
    programs = record.get("programs", []) or []
    aliases = record.get("aliases", []) or []
    works_with = record.get("works_with", []) or []
    explanation = record.get("explanation", {}) or {}
    explanation_summary = (
        explanation.get("summary", "") if isinstance(explanation, dict) else ""
    )
    evolution_statistics = record.get("evolution_statistics", {}) or {}
    usage = record.get("usage", {}) or {}
    parts = [
        f"description: {description}",
        f"task_description_summary: {task_description_summary}",
        f"task_description: {task_description}",
        f"category: {category}",
        f"program_id: {program_id}",
        f"fitness: {fitness}",
        f"strategy: {strategy}",
        f"last_generation: {last_generation}",
        f"programs: {programs}",
        f"aliases: {aliases}",
        f"keywords: {keywords}",
        f"evolution_statistics: {evolution_statistics}",
        f"explanation_summary: {explanation_summary}",
        f"works_with: {works_with}",
        f"links: {links}",
        f"connected_ideas: {connected_ideas}",
        f"usage: {usage}",
    ]
    return "\n".join(parts)


def build_gam_store(records: list[dict[str, Any]], store_dir: Path):
    memory_store = InMemoryMemoryStore(dir_path=str(store_dir))
    page_store = InMemoryPageStore(dir_path=str(store_dir))

    existing_pages = page_store.load()
    existing_ids = {
        str((p.meta or {}).get("amem_id") or "").strip()
        for p in existing_pages
        if isinstance(p.meta, dict)
    }
    existing_ids.discard("")

    added = 0
    next_pages: list[Page] = []
    seen_ids: set[str] = set()
    for rec in records:
        rid = str(rec.get("id") or "").strip()
        if rid and rid in seen_ids:
            continue
        if rid:
            seen_ids.add(rid)
        card = make_card_text(rec)
        abstract = rec.get("description") or rec.get("content") or card
        memory_store.add(abstract)
        header = f"[A-MEM] {rid}" if rid else "[A-MEM]"
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
    enable_bm25: bool = False,
    allowed_tools: list[str] | set[str] | tuple[str, ...] | None = None,
):
    retrievers = {}

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
        allowed = {"page_index", "keyword", *vector_tool_configs.keys()}

    if "page_index" in allowed:
        try:
            index_retriever = IndexRetriever(
                {"index_dir": str(index_dir / "page_index")}
            )
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            logger.debug("[Memory] Index retriever ready")
        except Exception as e:
            logger.warning("[Memory] Index retriever init failed: {}", e)

    for tool_name, extra in vector_tool_configs.items():
        if tool_name not in allowed:
            continue
        try:
            chroma_config = {
                "persist_dir": str(chroma_dir),
                "collection_name": chroma_collection,
                "model_name": config.AMEM_EMBEDDING_MODEL_NAME,
                **extra,
            }
            retrievers[tool_name] = ChromaRetriever(chroma_config)
            logger.debug("[Memory] Chroma retriever ready: {}", tool_name)
        except Exception as e:
            logger.warning(
                "[Memory] Chroma retriever init for '{}' failed: {}", tool_name, e
            )

    if enable_bm25 and "keyword" in allowed:
        try:
            from GAM_root.gam.retriever.bm25 import BM25Retriever

            bm25_config = {"index_dir": str(index_dir / "bm25")}
            bm25_retriever = BM25Retriever(bm25_config)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            logger.debug("[Memory] BM25 retriever ready")
        except Exception as e:
            logger.warning("[Memory] BM25 retriever init failed: {}", e)

    return retrievers


def main():
    export_file = Path("amem_exports/amem_memories.jsonl")
    # if export_path:
    #     export_file = Path(export_path)
    # else:
    #     export_file = Path(__file__).resolve().parents[1]  / "amem_memories.jsonl"

    if not export_file.exists():
        raise FileNotFoundError(f"A-mem export not found: {export_file}")

    records = load_amem_records(export_file)
    if not records:
        raise RuntimeError("A-mem export is empty.")

    store_dir = Path(__file__).resolve().parents[1] / "gam_shared" / "amem_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    memory_store, page_store, added = build_gam_store(records, store_dir)
    logger.info("Loaded {} A-mem records, added {} new pages.", len(records), added)

    api_key = config.OPENAI_API_KEY
    if not api_key and config.LLM_BASE_URL:
        api_key = "EMPTY"

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY/OPENROUTER_API_KEY env var. "
            "Set one before running this retriever."
        )

    base_url = config.LLM_BASE_URL

    llm_service = OpenAIInferenceService(
        model_name=config.OPENROUTER_MODEL_NAME or "openai/gpt-4.1-mini",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=2048,
        reasoning=config.OPENROUTER_REASONING,
    )
    generator = AMemGenerator({"llm_service": llm_service})

    chroma_dir = Path(__file__).resolve().parents[1] / "chroma"
    retrievers = build_retrievers(page_store, store_dir / "indexes", chroma_dir)
    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=generator,
        max_iters=3,
    )

    question = os.getenv(
        "AMEM_QUESTION", "What changes improved min_area the most and why?"
    )
    logger.info("Research question: {}", question)
    result = research_agent.research(question)
    logger.info("Research result:\n{}", result.integrated_memory)


if __name__ == "__main__":
    main()
