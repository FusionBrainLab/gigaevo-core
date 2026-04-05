"""Fake implementations of A-MEM and GAM classes for testing.

These fakes implement the exact interfaces that AmemGamMemory depends on,
allowing full-loop tests without real Chroma, embeddings, or LLM calls.

Usage:
    from tests.fakes.agentic_memory import (
        FakeMemoryNote,
        FakeAgenticMemorySystem,
        FakeResearchAgent,
        FakeAMemGenerator,
        inject_fakes_into_memory,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import re
from typing import Any
import uuid

# ---------------------------------------------------------------------------
# FakeMemoryNote — mirrors A_mem.agentic_memory.memory_system.MemoryNote
# ---------------------------------------------------------------------------


@dataclass
class FakeMemoryNote:
    """In-memory MemoryNote with all fields AmemGamMemory accesses."""

    content: str = ""
    id: str = ""
    keywords: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    timestamp: str = ""
    last_accessed: str = ""
    context: str = "General"
    evolution_history: list[Any] = field(default_factory=list)
    category: str = "Uncategorized"
    tags: list[str] = field(default_factory=list)
    strategy: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M")
        if not self.last_accessed:
            self.last_accessed = self.timestamp


# ---------------------------------------------------------------------------
# FakeRetriever — mirrors ChromaRetriever interface
# ---------------------------------------------------------------------------


class FakeRetriever:
    """In-memory retriever that stores documents by ID."""

    def __init__(self):
        self._docs: dict[str, tuple[str, dict]] = {}  # id → (document, metadata)

    def add_document(self, document: str, metadata: dict, doc_id: str) -> None:
        self._docs[doc_id] = (document, metadata)

    def delete_document(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def search(self, queries: list[str], top_k: int = 5) -> list[list[Any]]:
        """Jaccard similarity search — returns normalized [0,1] scores."""
        results = []
        for query in queries:
            tokens = set(re.split(r"\W+", query.lower())) - {""}
            scored = []
            for doc_id, (doc, meta) in self._docs.items():
                doc_tokens = set(re.split(r"\W+", doc.lower())) - {""}
                union = tokens | doc_tokens
                if not union:
                    continue
                jaccard = len(tokens & doc_tokens) / len(union)
                if jaccard > 0:
                    # Put score in meta so _score_retrieved_candidates can read it
                    hit_meta = {**meta, "score": jaccard}
                    scored.append(
                        FakeSearchResult(
                            page_id=doc_id,
                            score=jaccard,
                            meta=hit_meta,
                        )
                    )
            scored.sort(key=lambda r: r.score, reverse=True)
            results.append(scored[:top_k])
        return results


@dataclass
class FakeSearchResult:
    """Mimics the GAM search result object."""

    page_id: str = ""
    score: float = 0.0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FakeAgenticMemorySystem — mirrors AgenticMemorySystem
# ---------------------------------------------------------------------------


class FakeAgenticMemorySystem:
    """In-memory agentic memory system — no Chroma, no embeddings."""

    def __init__(
        self,
        model_name: str = "fake-model",
        llm_backend: str = "custom",
        llm_service: Any = None,
        chroma_persist_dir: Any = None,
        chroma_collection_name: str = "memories",
        use_gam_card_document: bool = False,
        enable_evolution: bool = False,
        **kwargs,
    ):
        self.memories: dict[str, FakeMemoryNote] = {}
        self.retriever = FakeRetriever()
        self._use_gam_card_document = use_gam_card_document
        self._llm_service = llm_service

    def read(self, memory_id: str) -> FakeMemoryNote | None:
        return self.memories.get(memory_id)

    def add_note(
        self,
        content: str,
        id: str | None = None,
        time: str | None = None,
        **kwargs,
    ) -> str:
        note_id = id or str(uuid.uuid4())
        note = FakeMemoryNote(
            content=content,
            id=note_id,
            keywords=kwargs.get("keywords", []),
            links=kwargs.get("links", []),
            context=kwargs.get("context", "General"),
            category=kwargs.get("category", "Uncategorized"),
            tags=kwargs.get("tags", []),
            strategy=kwargs.get("strategy", ""),
        )
        self.memories[note_id] = note
        doc = self._document_for_note(note)
        self.retriever.add_document(doc, self._note_metadata(note), note_id)
        return note_id

    def update(self, memory_id: str, content: str | None = None, **kwargs) -> bool:
        note = self.memories.get(memory_id)
        if note is None:
            return False
        if content is not None:
            note.content = content
        for key in ("keywords", "links", "context", "category", "tags", "strategy"):
            if key in kwargs:
                setattr(note, key, kwargs[key])
        note.last_accessed = datetime.now(UTC).strftime("%Y%m%d%H%M")
        doc = self._document_for_note(note)
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(doc, self._note_metadata(note), memory_id)
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self.memories:
            return False
        del self.memories[memory_id]
        self.retriever.delete_document(memory_id)
        return True

    def analyze_content(self, content: str) -> dict[str, Any]:
        """Extract keywords from content (simple word extraction)."""
        words = re.findall(r"\b[a-z]{4,}\b", content.lower())
        unique = list(dict.fromkeys(words))[:10]
        return {
            "keywords": unique,
            "context": "General",
            "tags": [],
        }

    def _document_for_note(self, note: FakeMemoryNote) -> str:
        if self._use_gam_card_document:
            return f"id: {note.id}\ncontent: {note.content}\nkeywords: {note.keywords}"
        return note.content

    def _note_metadata(self, note: FakeMemoryNote) -> dict[str, Any]:
        return {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "category": note.category,
            "context": note.context,
            "strategy": note.strategy,
        }


# ---------------------------------------------------------------------------
# FakeAMemGenerator — mirrors GAM_root.gam.generator.AMemGenerator
# ---------------------------------------------------------------------------


class FakeAMemGenerator:
    """Generator that returns canned LLM responses."""

    def __init__(self, config: dict[str, Any]):
        self._llm_service = config.get("llm_service")
        if self._llm_service is None:
            raise ValueError("llm_service is required in config")

    def generate_single(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        schema: dict | None = None,
        extra_params: dict | None = None,
    ) -> dict[str, Any]:
        return {"text": "Generated response", "json": None, "response": None}

    def generate_batch(
        self,
        prompts: list[str] | None = None,
        messages_list: list | None = None,
        schema: dict | None = None,
        extra_params: dict | None = None,
    ) -> list[dict[str, Any]]:
        count = len(prompts or messages_list or [])
        return [self.generate_single() for _ in range(count)]


# ---------------------------------------------------------------------------
# FakeResearchAgent — mirrors GAM_root.gam.ResearchAgent
# ---------------------------------------------------------------------------


@dataclass
class FakeResearchOutput:
    integrated_memory: str = ""
    raw_memory: dict = field(default_factory=dict)


class FakeResearchAgent:
    """Research agent that searches the fake retriever and formats results."""

    def __init__(
        self,
        page_store: Any = None,
        memory_store: Any = None,
        retrievers: dict | None = None,
        generator: Any = None,
        max_iters: int = 3,
        allowed_tools: list[str] | None = None,
        top_k_by_tool: dict[str, int] | None = None,
        pipeline_mode: str = "default",
        **kwargs,
    ):
        self._retrievers = retrievers or {}
        self._generator = generator

    def research(
        self, request: str, memory_state: str | None = None
    ) -> FakeResearchOutput:
        """Search all retrievers and format results."""
        all_results = []
        for name, retriever in self._retrievers.items():
            if hasattr(retriever, "search"):
                hits = retriever.search([request], top_k=5)
                if hits and hits[0]:
                    all_results.extend(hits[0])

        if not all_results:
            return FakeResearchOutput(
                integrated_memory="No relevant memories found.",
                raw_memory={},
            )

        lines = []
        card_ids = []
        for i, hit in enumerate(all_results[:5], 1):
            lines.append(f"{i}. {hit.page_id} [general] {hit.meta.get('content', '')}")
            card_ids.append(hit.page_id)

        return FakeResearchOutput(
            integrated_memory="\n".join(lines),
            raw_memory={
                "final_decision": {
                    "top_ideas": [{"card_id": cid} for cid in card_ids],
                },
            },
        )


# ---------------------------------------------------------------------------
# Fake GAM stores — mirrors InMemoryMemoryStore, InMemoryPageStore, Page
# ---------------------------------------------------------------------------


@dataclass
class FakePage:
    """Mirrors GAM_root.gam.schemas.Page."""

    header: str = ""
    content: str = ""
    meta: dict = field(default_factory=dict)


class FakeMemoryStore:
    """Mirrors InMemoryMemoryStore."""

    def __init__(self, dir_path: str = ""):
        self._items: list[str] = []

    def add(self, abstract: str) -> None:
        self._items.append(abstract)

    def load(self) -> list[str]:
        return list(self._items)


class FakePageStore:
    """Mirrors InMemoryPageStore."""

    def __init__(self, dir_path: str = ""):
        self._pages: list[FakePage] = []

    def save(self, pages: list[FakePage]) -> None:
        self._pages = list(pages)

    def load(self) -> list[FakePage]:
        return list(self._pages)


# ---------------------------------------------------------------------------
# Fake build_gam_store / build_retrievers / load_amem_records
# ---------------------------------------------------------------------------


def fake_load_amem_records(path: Any) -> list[dict[str, Any]]:
    """Load JSONL records — same as real implementation, no external deps."""
    import json as _json
    from pathlib import Path as _Path

    records: list[dict[str, Any]] = []
    p = _Path(path)
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(_json.loads(line))
    return records


def fake_build_gam_store(
    records: list[dict[str, Any]], store_dir: Any
) -> tuple[FakeMemoryStore, FakePageStore, int]:
    """Build fake GAM stores from records."""
    memory_store = FakeMemoryStore()
    page_store = FakePageStore()

    pages: list[FakePage] = []
    for rec in records:
        rid = str(rec.get("id") or "").strip()
        desc = rec.get("description") or rec.get("content") or ""
        memory_store.add(desc)
        pages.append(
            FakePage(
                header=f"[A-MEM] {rid}" if rid else "[A-MEM]",
                content=desc,
                meta={"amem_id": rid, "amem": rec},
            )
        )

    page_store.save(pages)
    return memory_store, page_store, len(pages)


def fake_build_retrievers(
    page_store: FakePageStore,
    index_dir: Any,
    chroma_dir: Any,
    chroma_collection: str = "memories",
    enable_bm25: bool = False,
    allowed_tools: list[str] | None = None,
) -> dict[str, FakeRetriever]:
    """Build fake retrievers from page store content.

    Creates a FakeRetriever per allowed tool, all sharing the same
    in-memory keyword search index built from page content.
    """
    allowed = set(
        allowed_tools
        or [
            "page_index",
            "keyword",
            "vector",
            "vector_description",
            "vector_task_description",
            "vector_explanation_summary",
            "vector_description_explanation_summary",
            "vector_description_task_description_summary",
        ]
    )

    # Build a single index from all pages
    base_retriever = FakeRetriever()
    for page in page_store.load():
        amem_id = (page.meta or {}).get("amem_id", "")
        base_retriever.add_document(
            page.content,
            {**(page.meta or {})},
            amem_id or page.header,
        )

    return {tool: base_retriever for tool in allowed}


# ---------------------------------------------------------------------------
# inject_fakes_into_memory — wire fakes into AmemGamMemory
# ---------------------------------------------------------------------------


def inject_fakes_into_memory(mem: Any) -> FakeAgenticMemorySystem:
    """Replace agentic classes in an AmemGamMemory instance with fakes.

    Returns the FakeAgenticMemorySystem so tests can inspect its state.
    """
    from gigaevo.memory.shared_memory.note_sync import NoteSync

    fake_system = FakeAgenticMemorySystem(
        use_gam_card_document=True,
        llm_service=mem.llm_service,
    )
    mem.memory_system = fake_system
    mem._AgenticMemorySystemCls = FakeAgenticMemorySystem
    mem._MemoryNoteCls = FakeMemoryNote
    mem._ResearchAgentCls = FakeResearchAgent
    mem._AMemGeneratorCls = FakeAMemGenerator
    mem.note_sync = NoteSync(
        memory_system=fake_system, note_cls=FakeMemoryNote, card_store=mem.card_store
    )
    return fake_system


def patch_gam_imports():
    """Return a dict suitable for unittest.mock.patch on the amem_gam_retriever import.

    Usage:
        with patch.dict('sys.modules', patch_gam_imports()):
            ...
    Or more commonly, patch the specific import site in memory.py.
    """
    import types

    fake_module = types.ModuleType("shared_memory.amem_gam_retriever")
    fake_module.build_gam_store = fake_build_gam_store  # type: ignore[attr-defined]
    fake_module.build_retrievers = fake_build_retrievers  # type: ignore[attr-defined]
    fake_module.load_amem_records = fake_load_amem_records  # type: ignore[attr-defined]
    return {"shared_memory.amem_gam_retriever": fake_module}
