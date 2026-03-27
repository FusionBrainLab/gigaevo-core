"""GAM retrievers backed by gigaevo-memory search APIs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _ensure_memory_client_path() -> None:
    workspace_root = Path(__file__).resolve().parents[4]
    client_src = workspace_root / "gigaevo-memory" / "client" / "python" / "src"
    if client_src.exists() and str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))
    legacy_memory_root = Path(__file__).resolve().parents[2] / "memory"
    if legacy_memory_root.exists() and str(legacy_memory_root) not in sys.path:
        sys.path.insert(0, str(legacy_memory_root))


_ensure_memory_client_path()

from gigaevo_memory.embeddings import MemoryApiProvider
from gigaevo_memory.models import SearchHitData
from gigaevo_memory.platform_client import PlatformMemoryClient
from gigaevo_memory.search_types import SearchType

from GAM_root.gam import InMemoryMemoryStore, InMemoryPageStore
from GAM_root.gam.retriever.base import AbsRetriever
from GAM_root.gam.schemas import Hit, Page


DOCUMENT_KIND_FULL_CARD = "full_card"
DOCUMENT_KIND_DESCRIPTION = "description"
DOCUMENT_KIND_TASK_DESCRIPTION = "task_description"
DOCUMENT_KIND_EXPLANATION_SUMMARY = "explanation_summary"
DOCUMENT_KIND_DESCRIPTION_EXPLANATION_SUMMARY = "description_explanation_summary"
DOCUMENT_KIND_DESCRIPTION_TASK_DESCRIPTION_SUMMARY = (
    "description_task_description_summary"
)


def make_card_text(record: dict[str, Any]) -> str:
    description = record.get("description") or ""
    task_description = record.get("task_description") or ""
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
    explanation_summary = explanation.get("summary", "") if isinstance(explanation, dict) else str(explanation or "")
    evolution_statistics = record.get("evolution_statistics", {}) or {}
    usage = record.get("usage", {}) or {}
    code = record.get("code") or ""
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
        f"code: {code}",
    ]
    return "\n".join(parts)


def build_gam_store(records: list[dict[str, Any]], store_dir: Path):
    memory_store = InMemoryMemoryStore(dir_path=str(store_dir))
    page_store = InMemoryPageStore(dir_path=str(store_dir))

    pages: list[Page] = []
    seen_ids: set[str] = set()
    for record in records:
        card_id = str(record.get("id") or "").strip()
        if card_id and card_id in seen_ids:
            continue
        if card_id:
            seen_ids.add(card_id)
        card_text = make_card_text(record)
        abstract = record.get("description") or card_text
        memory_store.add(str(abstract))
        header = f"[MEMORY_PLATFORM] {card_id}" if card_id else "[MEMORY_PLATFORM]"
        pages.append(
            Page(
                header=header,
                content=card_text,
                meta={"card_id": card_id, "memory_card": record},
            )
        )

    page_store.save(pages)
    return memory_store, page_store, len(pages)


class RemoteSearchRetriever(AbsRetriever):
    """Retriever that delegates search to gigaevo-memory."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        client: PlatformMemoryClient,
        search_type: SearchType,
        source_label: str,
        namespace: str | None = None,
        channel: str = "latest",
        document_kind: str | None = None,
        hybrid_weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        super().__init__(config)
        self.client = client
        self.search_type = search_type
        self.source_label = source_label
        self.namespace = namespace
        self.channel = channel
        self.document_kind = document_kind
        self.hybrid_weights = hybrid_weights
        self.page_store = None
        self._page_index_by_card_id: dict[str, int] = {}
        self.name = source_label

    def build(self, page_store: InMemoryPageStore):
        self.update(page_store)

    def load(self):
        return None

    def update(self, page_store: InMemoryPageStore):
        self.page_store = page_store
        self._page_index_by_card_id = {}
        for index, page in enumerate(page_store.load()):
            if isinstance(page.meta, dict):
                card_id = str(page.meta.get("card_id") or "").strip()
                if card_id:
                    self._page_index_by_card_id[card_id] = index

    def _hit_to_gam_hit(self, hit: SearchHitData) -> Hit:
        content = hit.content or {}
        card_id = str((content.get("id") if isinstance(content, dict) else None) or "").strip()
        page_index = self._page_index_by_card_id.get(card_id)
        page_id = str(page_index) if page_index is not None else card_id or str(hit.entity_id)
        snippet = str(hit.snippet or (content.get("description") if isinstance(content, dict) else "") or "")
        return Hit(
            page_id=page_id,
            snippet=snippet,
            source=self.source_label,
            meta={
                "score": float(hit.score),
                "entity_id": hit.entity_id,
                "version_id": hit.version_id,
                "document_id": hit.document_id,
                "document_kind": hit.document_kind,
                "card_id": card_id,
            },
        )

    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]:
        hit_batches = self.client.batch_search_hits(
            queries=query_list,
            search_type=self.search_type,
            top_k=top_k,
            entity_type="memory_card",
            namespace=self.namespace,
            channel=self.channel,
            document_kind=self.document_kind,
            hybrid_weights=self.hybrid_weights,
        )
        return [
            [self._hit_to_gam_hit(hit) for hit in hits]
            for hits in hit_batches
        ]


class RemotePageIndexRetriever(AbsRetriever):
    """Simple page-index retriever over the locally mirrored page store."""

    name = "page_index"

    def __init__(self, config: dict[str, Any], page_store: InMemoryPageStore):
        super().__init__(config)
        self.page_store = page_store

    def build(self, page_store: InMemoryPageStore):
        self.page_store = page_store

    def load(self):
        return None

    def update(self, page_store: InMemoryPageStore):
        self.page_store = page_store

    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]:
        results: list[list[Hit]] = []
        for query in query_list:
            page_hits: list[Hit] = []
            for raw_part in str(query).split(","):
                raw_part = raw_part.strip()
                if not raw_part:
                    continue
                try:
                    page_index = int(raw_part)
                except ValueError:
                    continue
                page = self.page_store.get(page_index)
                if page is None:
                    continue
                page_hits.append(
                    Hit(
                        page_id=str(page_index),
                        snippet=page.content,
                        source="page_index",
                        meta=page.meta if isinstance(page.meta, dict) else {},
                    )
                )
                if len(page_hits) >= top_k:
                    break
            results.append(page_hits)
        return results


def build_retrievers(
    page_store: InMemoryPageStore,
    client: PlatformMemoryClient,
    *,
    vector_search_type: str = "vector",
    namespace: str | None = None,
    channel: str = "latest",
    hybrid_weights: tuple[float, float] = (0.5, 0.5),
    enable_keyword: bool = True,
) -> dict[str, AbsRetriever]:
    if vector_search_type == "hybrid":
        semantic_search_type = SearchType.HYBRID
    else:
        semantic_search_type = SearchType.VECTOR

    retrievers: dict[str, AbsRetriever] = {
        "page_index": RemotePageIndexRetriever({}, page_store),
        "vector": RemoteSearchRetriever(
            {},
            client=client,
            search_type=semantic_search_type,
            source_label="vector",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_FULL_CARD,
            hybrid_weights=hybrid_weights,
        ),
        "vector_description": RemoteSearchRetriever(
            {},
            client=client,
            search_type=semantic_search_type,
            source_label="vector_description",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_DESCRIPTION,
            hybrid_weights=hybrid_weights,
        ),
        "vector_task_description": RemoteSearchRetriever(
            {},
            client=client,
            search_type=semantic_search_type,
            source_label="vector_task_description",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_TASK_DESCRIPTION,
            hybrid_weights=hybrid_weights,
        ),
        "vector_explanation_summary": RemoteSearchRetriever(
            {},
            client=client,
            search_type=semantic_search_type,
            source_label="vector_explanation_summary",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_EXPLANATION_SUMMARY,
            hybrid_weights=hybrid_weights,
        ),
        "vector_description_explanation_summary": RemoteSearchRetriever(
            {},
            client=client,
            search_type=semantic_search_type,
            source_label="vector_description_explanation_summary",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_DESCRIPTION_EXPLANATION_SUMMARY,
            hybrid_weights=hybrid_weights,
        ),
        "vector_description_task_description_summary": RemoteSearchRetriever(
            {},
            client=client,
            search_type=semantic_search_type,
            source_label="vector_description_task_description_summary",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_DESCRIPTION_TASK_DESCRIPTION_SUMMARY,
            hybrid_weights=hybrid_weights,
        ),
    }

    if enable_keyword:
        retrievers["keyword"] = RemoteSearchRetriever(
            {},
            client=client,
            search_type=SearchType.BM25,
            source_label="keyword",
            namespace=namespace,
            channel=channel,
            document_kind=DOCUMENT_KIND_FULL_CARD,
        )

    for retriever in retrievers.values():
        retriever.build(page_store)

    return retrievers
def build_memory_client(base_url: str) -> PlatformMemoryClient:
    return PlatformMemoryClient(
        base_url=base_url,
        embedding_provider=MemoryApiProvider(base_url=base_url),
    )
