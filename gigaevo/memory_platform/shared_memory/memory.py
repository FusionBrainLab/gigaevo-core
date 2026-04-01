from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import uuid

from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
_LEGACY_MEMORY_ROOT = _THIS_DIR.parents[2] / "memory"
if str(_LEGACY_MEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_MEMORY_ROOT))
_WORKSPACE_ROOT = _THIS_DIR.parents[4]
_MEMORY_CLIENT_SRC = _WORKSPACE_ROOT / "gigaevo-memory" / "client" / "python" / "src"
if _MEMORY_CLIENT_SRC.exists() and str(_MEMORY_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_MEMORY_CLIENT_SRC))

from gigaevo.memory import config
from gigaevo.memory.openai_inference import OpenAIInferenceService
from gigaevo.memory.shared_memory.card_update_dedup import (
    QUERY_DESCRIPTION,
    QUERY_DESCRIPTION_EXPLANATION_SUMMARY,
    QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY,
    QUERY_EXPLANATION_SUMMARY,
    CardUpdateDedupConfig,
    build_dedup_queries,
    compute_weighted_candidates,
    get_explanation_summary,
    get_full_explanations,
    merge_updated_card,
    parse_llm_card_decision,
)

load_dotenv()

from gigaevo_memory.embeddings import MemoryApiProvider
from gigaevo_memory.platform_client import PlatformMemoryClient
from gigaevo_memory.search_types import SearchType

_ALLOWED_STRATEGIES = {"exploration", "exploitation", "hybrid"}
_VECTOR_GAM_TOOLS = {
    "vector",
    "vector_description",
    "vector_task_description",
    "vector_explanation_summary",
    "vector_description_explanation_summary",
    "vector_description_task_description_summary",
}
_ALLOWED_GAM_TOOLS = {"keyword", "page_index", *_VECTOR_GAM_TOOLS}
_ALLOWED_GAM_PIPELINE_MODES = {"default", "experimental"}
_DEFAULT_GAM_TOP_K_BY_TOOL = {
    "keyword": 5,
    "vector": 5,
    "vector_description": 5,
    "vector_task_description": 5,
    "vector_explanation_summary": 5,
    "vector_description_explanation_summary": 5,
    "vector_description_task_description_summary": 5,
    "page_index": 5,
}
DOCUMENT_KIND_FULL_CARD = "full_card"


def build_memory_client(base_url: str) -> PlatformMemoryClient:
    return PlatformMemoryClient(
        base_url=base_url,
        embedding_provider=MemoryApiProvider(base_url=base_url),
    )


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_memory_card(
    card: dict[str, Any] | None = None,
    fallback_id: str | None = None,
) -> dict[str, Any]:
    raw = dict(card or {})
    category = str(raw.get("category") or "general")
    program_id = str(raw.get("program_id") or "")
    if category == "program" or program_id:
        return {
            "id": str(raw.get("id") or fallback_id or ""),
            "category": "program",
            "program_id": program_id,
            "task_description": str(raw.get("task_description") or raw.get("context") or ""),
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
        "task_description": str(raw.get("task_description") or raw.get("context") or ""),
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


class GigaEvoMemoryBase:
    def save(self, data: str) -> str:
        raise NotImplementedError

    def search(self, query: str) -> str:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError


class AmemGamMemory(GigaEvoMemoryBase):
    """Platform-backed memory implementation using gigaevo-memory APIs."""

    def __init__(
        self,
        checkpoint_path: str,
        base_url: str = "http://localhost:8000",
        use_api: bool = True,
        namespace: str = "default",
        author: str | None = None,
        channel: str = "latest",
        search_limit: int = 5,
        enable_llm_synthesis: bool = True,
        enable_memory_evolution: bool = True,
        enable_llm_card_enrichment: bool = True,
        rebuild_interval: int = 10,
        enable_bm25: bool = False,
        sync_batch_size: int = 100,
        sync_on_init: bool = True,
        allowed_gam_tools: list[str] | None = None,
        gam_top_k_by_tool: dict[str, int] | None = None,
        gam_pipeline_mode: str = "default",
        card_update_dedup_config: dict[str, Any] | None = None,
        remote_vector_search_type: str = "vector",
        remote_hybrid_weights: tuple[float, float] = (0.4, 0.6),
    ):
        self.checkpoint_dir = Path(checkpoint_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.checkpoint_dir / "platform_index.json"
        self.gam_store_dir = self.checkpoint_dir / "gam_shared" / "platform_store"

        self.base_url = base_url.rstrip("/")
        self.use_api = True
        self.namespace = namespace
        self.author = author
        self.channel = channel
        self.search_limit = search_limit
        self.enable_llm_synthesis = enable_llm_synthesis
        self.enable_memory_evolution = bool(enable_memory_evolution)
        self.enable_llm_card_enrichment = bool(enable_llm_card_enrichment)
        self.rebuild_interval = max(1, int(rebuild_interval))
        self.enable_bm25 = bool(enable_bm25)
        self.sync_batch_size = max(10, int(sync_batch_size))
        self.allowed_gam_tools = self._normalize_allowed_gam_tools(allowed_gam_tools)
        self.gam_top_k_by_tool = self._normalize_gam_top_k_by_tool(gam_top_k_by_tool)
        self.gam_pipeline_mode = self._normalize_gam_pipeline_mode(gam_pipeline_mode)
        self.card_update_dedup_config = CardUpdateDedupConfig.from_mapping(
            card_update_dedup_config or {}
        )
        self.remote_vector_search_type = str(remote_vector_search_type or "vector").strip().lower()
        self.remote_hybrid_weights = remote_hybrid_weights

        if not use_api:
            print("[MemoryPlatform] use_api=False was requested, but memory_platform always uses the backend API.")

        self.client = build_memory_client(self.base_url)
        self._AgenticMemorySystemCls: type[Any] | None = None
        self._ResearchAgentCls: type[Any] | None = None
        self._AMemGeneratorCls: type[Any] | None = None
        self._build_gam_store_fn: Any | None = None
        self._build_retrievers_fn: Any | None = None
        self._agentic_import_error: Exception | None = None
        self._load_agentic_classes()
        self.llm_service, self.generator = self._init_llm_service_and_generator()
        self.memory_system = self._init_storage()

        self.memory_cards: dict[str, dict[str, Any]] = {}
        self.entity_by_card_id: dict[str, str] = {}
        self.card_id_by_entity: dict[str, str] = {}
        self.entity_version_by_entity: dict[str, str] = {}
        self.memory_ids: set[str] = set()
        self.card_write_stats: dict[str, int] = {
            "processed": 0,
            "added": 0,
            "rejected": 0,
            "updated": 0,
            "updated_target_cards": 0,
        }
        self._dedup_retrievers: dict[str, Any] | None = None
        self.research_agent: Any | None = None
        self._warned_missing_card_update_llm = False
        self._iters_after_rebuild = 0

        self._load_index()
        if sync_on_init:
            self.refresh_from_backend()
            self.rebuild()

    @staticmethod
    def _normalize_allowed_gam_tools(allowed_gam_tools: list[str] | None) -> set[str]:
        if not allowed_gam_tools:
            return set(_ALLOWED_GAM_TOOLS)
        valid = {str(tool).strip() for tool in allowed_gam_tools if str(tool).strip() in _ALLOWED_GAM_TOOLS}
        if "vector" in valid:
            valid.update(_VECTOR_GAM_TOOLS)
        return valid or set(_ALLOWED_GAM_TOOLS)

    @staticmethod
    def _normalize_gam_top_k_by_tool(gam_top_k_by_tool: dict[str, int] | None) -> dict[str, int]:
        normalized = dict(_DEFAULT_GAM_TOP_K_BY_TOOL)
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

    @staticmethod
    def _normalize_gam_pipeline_mode(gam_pipeline_mode: str | None) -> str:
        mode = str(gam_pipeline_mode or "default").strip().lower()
        return mode if mode in _ALLOWED_GAM_PIPELINE_MODES else "default"

    def _load_agentic_classes(self) -> None:
        try:
            from A_mem.agentic_memory.memory_system import (
                AgenticMemorySystem as _AgenticMemorySystem,
            )
            from GAM_root.gam import ResearchAgent as _ResearchAgent
            from GAM_root.gam.generator import AMemGenerator as _AMemGenerator

            from .remote_gam_retriever import (
                build_gam_store as _build_gam_store,
            )
            from .remote_gam_retriever import (
                build_retrievers as _build_retrievers,
            )
        except Exception as exc:
            self._agentic_import_error = exc
            return
        self._AgenticMemorySystemCls = _AgenticMemorySystem
        self._ResearchAgentCls = _ResearchAgent
        self._AMemGeneratorCls = _AMemGenerator
        self._build_gam_store_fn = _build_gam_store
        self._build_retrievers_fn = _build_retrievers

    def _init_llm_service_and_generator(self) -> tuple[Any | None, Any | None]:
        if self._AMemGeneratorCls is None and not self.card_update_dedup_config.enabled:
            return None, None
        api_key = config.OPENAI_API_KEY
        if not api_key and config.LLM_BASE_URL:
            api_key = "EMPTY"
        if not api_key:
            return None, None
        try:
            llm_service = OpenAIInferenceService(
                model_name=config.OPENROUTER_MODEL_NAME,
                api_key=api_key,
                base_url=config.LLM_BASE_URL,
                temperature=0.0,
                max_tokens=0,
                reasoning=config.OPENROUTER_REASONING,
            )
            if self._AMemGeneratorCls is None:
                return llm_service, None
            generator = self._AMemGeneratorCls({"llm_service": llm_service})
            return llm_service, generator
        except Exception as exc:
            print(f"[MemoryPlatform] Could not initialize LLM/generator: {exc}")
            return None, None

    def _init_storage(self) -> Any | None:
        if self.llm_service is None or self._AgenticMemorySystemCls is None:
            return None
        try:
            return self._AgenticMemorySystemCls(
                model_name=config.AMEM_EMBEDDING_MODEL_NAME,
                llm_backend="custom",
                llm_service=self.llm_service,
                chroma_persist_dir=self.checkpoint_dir / "chroma",
                chroma_collection_name="platform_memories",
                use_gam_card_document=True,
                enable_evolution=self.enable_memory_evolution,
            )
        except Exception as exc:
            print(f"[MemoryPlatform] Could not initialize AgenticMemorySystem: {exc}")
            return None

    def _load_index(self) -> None:
        if not self.index_file.exists():
            return
        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[MemoryPlatform] Could not parse index file {self.index_file}: {exc}")
            return

        raw_cards = payload.get("memory_cards", {})
        raw_map = payload.get("entity_by_card_id", {})
        raw_versions = payload.get("entity_version_by_entity", {})
        if isinstance(raw_cards, dict):
            for card_id, card in raw_cards.items():
                cid = str(card_id)
                self.memory_cards[cid] = normalize_memory_card(card, fallback_id=cid)
                self.memory_ids.add(cid)
        if isinstance(raw_map, dict):
            for card_id, entity_id in raw_map.items():
                cid = str(card_id)
                eid = str(entity_id)
                if cid and eid:
                    self.entity_by_card_id[cid] = eid
                    self.card_id_by_entity[eid] = cid
        if isinstance(raw_versions, dict):
            for entity_id, version_id in raw_versions.items():
                eid = str(entity_id)
                if eid:
                    self.entity_version_by_entity[eid] = str(version_id or "")

    def _persist_index(self) -> None:
        payload = {
            "entity_by_card_id": self.entity_by_card_id,
            "entity_version_by_entity": self.entity_version_by_entity,
            "memory_cards": self.memory_cards,
        }
        self.index_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def refresh_from_backend(self) -> None:
        memory_cards: dict[str, dict[str, Any]] = {}
        entity_by_card_id: dict[str, str] = {}
        card_id_by_entity: dict[str, str] = {}
        entity_version_by_entity: dict[str, str] = {}
        memory_ids: set[str] = set()

        offset = 0
        while True:
            batch = self.client.list_memory_cards(
                limit=self.sync_batch_size,
                offset=offset,
                channel=self.channel,
            )
            if not batch:
                break

            for entity in batch:
                meta = entity.meta or {}
                row_namespace = meta.get("namespace") if isinstance(meta, dict) else None
                if self.namespace and row_namespace not in (None, "", self.namespace):
                    continue
                card = normalize_memory_card(entity.content or {}, fallback_id=str(entity.entity_id))
                card_id = self._ensure_card_id(card)
                memory_cards[card_id] = card
                memory_ids.add(card_id)
                entity_id = str(entity.entity_id)
                entity_by_card_id[card_id] = entity_id
                card_id_by_entity[entity_id] = card_id
                entity_version_by_entity[entity_id] = str(entity.version_id or "")

            offset += len(batch)
            if len(batch) < self.sync_batch_size:
                break

        self.memory_cards = memory_cards
        self.entity_by_card_id = entity_by_card_id
        self.card_id_by_entity = card_id_by_entity
        self.entity_version_by_entity = entity_version_by_entity
        self.memory_ids = memory_ids
        self._persist_index()
        self._dedup_retrievers = None

    @staticmethod
    def _looks_like_uuid(value: str) -> bool:
        try:
            uuid.UUID(value)
            return True
        except Exception:
            return False

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def _ensure_card_id(self, card: dict[str, Any]) -> str:
        card_id = str(card.get("id") or "").strip()
        if not card_id:
            card_id = f"mem-{uuid.uuid4().hex[:12]}"
            card["id"] = card_id
        return card_id

    @staticmethod
    def _truncate_text(value: Any, max_chars: int = 1200) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _is_program_card(card: dict[str, Any]) -> bool:
        if str(card.get("category") or "").strip().lower() == "program":
            return True
        return bool(str(card.get("program_id") or "").strip())

    def _card_to_backend_content(self, card: dict[str, Any]) -> dict[str, Any]:
        if self._is_program_card(card):
            return {
                "id": str(card.get("id") or ""),
                "category": "program",
                "program_id": str(card.get("program_id") or ""),
                "task_description": str(card.get("task_description") or ""),
                "task_description_summary": str(card.get("task_description_summary") or ""),
                "description": str(card.get("description") or ""),
                "fitness": _to_float(card.get("fitness"), default=None),
                "code": str(card.get("code") or ""),
                "connected_ideas": _to_list(card.get("connected_ideas")),
            }

        explanation = card.get("explanation")
        strategy = str(card.get("strategy") or "").strip().lower() or None
        if strategy not in _ALLOWED_STRATEGIES:
            strategy = None

        return {
            "id": str(card.get("id") or ""),
            "category": str(card.get("category") or "general"),
            "task_description": str(card.get("task_description") or ""),
            "task_description_summary": str(card.get("task_description_summary") or ""),
            "description": str(card.get("description") or ""),
            "explanation": explanation if isinstance(explanation, dict) else str(explanation or ""),
            "strategy": strategy,
            "keywords": self._dedupe_keep_order(list(card.get("keywords") or [])),
            "evolution_statistics": (
                card.get("evolution_statistics")
                if isinstance(card.get("evolution_statistics"), dict)
                else None
            ),
            "works_with": self._dedupe_keep_order(list(card.get("works_with") or [])),
            "links": self._dedupe_keep_order(list(card.get("links") or [])),
            "usage": card.get("usage") if isinstance(card.get("usage"), dict) else None,
            "last_generation": _to_int(card.get("last_generation"), default=0),
            "programs": self._dedupe_keep_order(list(card.get("programs") or [])),
            "aliases": self._dedupe_keep_order(list(card.get("aliases") or [])),
        }

    def _build_entity_meta(self, card: dict[str, Any]) -> tuple[str, list[str], str]:
        card_id = str(card.get("id") or "")
        description = str(card.get("description") or "").strip()
        task_description = str(card.get("task_description") or "").strip()
        task_description_summary = str(card.get("task_description_summary") or "").strip()

        explanation = card.get("explanation")
        explanation_summary = ""
        if isinstance(explanation, dict):
            explanation_summary = str(explanation.get("summary") or "").strip()
        else:
            explanation_summary = str(explanation or "").strip()

        name_seed = description or task_description_summary or task_description or "memory card"
        name = f"{card_id}: {name_seed}" if card_id else name_seed
        name = name[:255]

        tags = self._dedupe_keep_order(
            [
                str(card.get("category") or "").strip(),
                str(card.get("strategy") or "").strip(),
                *[str(x).strip() for x in (card.get("keywords") or [])],
            ]
        )

        when_to_use = " | ".join(
            self._dedupe_keep_order(
                [
                    task_description_summary,
                    task_description,
                    description,
                    explanation_summary,
                    " ".join([str(x) for x in (card.get("keywords") or [])]).strip(),
                ]
            )
        )
        return name, tags, when_to_use

    def rebuild(self) -> None:
        self._persist_index()
        if (
            self._build_gam_store_fn is None
            or self._build_retrievers_fn is None
            or self._ResearchAgentCls is None
        ):
            self.research_agent = None
            self._dedup_retrievers = None
            self._iters_after_rebuild = 0
            return
        records = list(self.memory_cards.values())
        memory_store, page_store, _ = self._build_gam_store_fn(records, self.gam_store_dir)
        self._dedup_retrievers = None
        self._iters_after_rebuild = 0

        if self.generator is None:
            self.research_agent = None
            return

        retrievers = self._build_retrievers_fn(
            page_store,
            self.client,
            vector_search_type=self.remote_vector_search_type,
            namespace=self.namespace,
            channel=self.channel,
            hybrid_weights=self.remote_hybrid_weights,
            enable_keyword=self.enable_bm25,
        )
        retrievers = {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self.allowed_gam_tools
        }
        self.research_agent = self._ResearchAgentCls(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=self.generator,
            max_iters=3,
            allowed_tools=sorted(self.allowed_gam_tools),
            top_k_by_tool=self.gam_top_k_by_tool,
            pipeline_mode=self.gam_pipeline_mode,
        )

    def _build_dedup_retrievers(self) -> dict[str, Any]:
        if self.research_agent is None:
            self.rebuild()
        if self.research_agent is None:
            return {}
        return {
            name: retriever
            for name, retriever in self.research_agent.retrievers.items()
            if name in self.allowed_gam_tools
        }

    def _resolve_vector_retriever(self, tool_name: str) -> Any | None:
        if self._dedup_retrievers is None:
            self._dedup_retrievers = self._build_dedup_retrievers()
        retrievers = self._dedup_retrievers or {}
        retriever = retrievers.get(tool_name)
        if retriever is None and tool_name != "vector":
            retriever = retrievers.get("vector")
        return retriever

    def _score_retrieved_candidates(self, card: dict[str, Any]) -> list[dict[str, Any]]:
        cfg = self.card_update_dedup_config
        if not cfg.enabled or not self.memory_cards:
            return []

        query_by_key = build_dedup_queries(card)
        tool_by_key = {
            QUERY_DESCRIPTION: "vector_description",
            QUERY_EXPLANATION_SUMMARY: "vector_explanation_summary",
            QUERY_DESCRIPTION_EXPLANATION_SUMMARY: "vector_description_explanation_summary",
            QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY: "vector_description_task_description_summary",
        }
        scores_by_query: dict[str, dict[str, float]] = {}

        for query_key, query_text in query_by_key.items():
            text_value = str(query_text or "").strip()
            if not text_value:
                continue
            retriever = self._resolve_vector_retriever(tool_by_key[query_key])
            if retriever is None:
                continue
            try:
                hits_by_query = retriever.search([text_value], top_k=cfg.top_k_per_query)
            except Exception as exc:
                print(f"[MemoryPlatform] Dedup retrieval failed for query '{query_key}': {exc}")
                continue

            hits = hits_by_query[0] if hits_by_query and isinstance(hits_by_query[0], list) else hits_by_query
            query_scores: dict[str, float] = {}
            for hit in hits or []:
                meta = getattr(hit, "meta", {}) or {}
                card_id = str(meta.get("card_id") or getattr(hit, "page_id", "") or "").strip()
                if not card_id or card_id not in self.memory_cards:
                    continue
                if self._is_program_card(self.memory_cards[card_id]):
                    continue
                try:
                    score = float(meta.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                if score > query_scores.get(card_id, 0.0):
                    query_scores[card_id] = score
            scores_by_query[query_key] = query_scores

        return compute_weighted_candidates(
            scores_by_query,
            weights=cfg.weights,
            final_top_n=cfg.final_top_n,
            min_final_score=cfg.min_final_score,
        )

    def _dedup_candidates_for_llm(self, scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in scored_candidates:
            card_id = str(item.get("card_id") or "").strip()
            card = self.memory_cards.get(card_id)
            if not isinstance(card, dict):
                continue
            explanations = get_full_explanations(card)
            payload.append(
                {
                    "card_id": card_id,
                    "final_score": float(item.get("final_score", 0.0)),
                    "scores": item.get("scores", {}),
                    "task_description_summary": self._truncate_text(card.get("task_description_summary"), 600),
                    "description": self._truncate_text(card.get("description"), 1200),
                    "explanation_summary": self._truncate_text(get_explanation_summary(card), 600),
                    "explanation_full": [self._truncate_text(explanation, 1200) for explanation in explanations],
                }
            )
        return payload

    def _decide_card_action(
        self,
        incoming_card: dict[str, Any],
        candidates_for_llm: list[dict[str, Any]],
    ) -> dict[str, Any]:
        default_decision = {"action": "add", "reason": "", "duplicate_of": "", "updates": []}
        if self.llm_service is None or not candidates_for_llm:
            return default_decision

        candidate_ids = {
            str(item.get("card_id") or "").strip()
            for item in candidates_for_llm
            if str(item.get("card_id") or "").strip()
        }
        if not candidate_ids:
            return default_decision

        incoming_payload = {
            "id": str(incoming_card.get("id") or "").strip(),
            "task_description_summary": self._truncate_text(incoming_card.get("task_description_summary"), 600),
            "description": self._truncate_text(incoming_card.get("description"), 1200),
            "explanation_summary": self._truncate_text(get_explanation_summary(incoming_card), 600),
            "explanation_full": [self._truncate_text(explanation, 1200) for explanation in get_full_explanations(incoming_card)],
        }
        prompt = (
            "You are a memory-card deduplication and update policy agent.\n"
            "For NEW_CARD choose exactly one action: add, discard, or update.\n"
            "Return only JSON with keys action, reason, duplicate_of, updates.\n\n"
            "Expected JSON schema:\n"
            "{\n"
            '  "action": "add" | "discard" | "update",\n'
            '  "reason": "brief explanation",\n'
            '  "duplicate_of": "candidate card_id or empty string",\n'
            '  "updates": [\n'
            "    {\n"
            '      "card_id": "candidate card_id",\n'
            '      "description_append": "optional text to append",\n'
            '      "explanation_summary_append": "optional text to append",\n'
            '      "explanation_full_append": ["optional bullet", "..."],\n'
            '      "links_append": ["optional link ids"],\n'
            '      "works_with_append": ["optional related systems"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Use add when NEW_CARD is a genuinely new idea and should become its own card.\n"
            "- Use update when one candidate already contains the same core idea, but NEW_CARD contributes a new use-case, sharper wording, extra mechanism detail, or additional explanation that should be merged into that existing card.\n"
            "- Use discard when one candidate already expresses the same idea with no meaningful new information.\n"
            "- Do not choose update or discard just because cards share the same broad task, benchmark, or domain.\n"
            "- Compare the actual idea/mechanism/intervention described in DESCRIPTION and EXPLANATION.\n"
            "- If the core idea in NEW_CARD is meaningfully different from every candidate, action must be add.\n"
            "- If action=discard, set duplicate_of to one candidate card_id.\n"
            "- If action=update, include one or more update objects with candidate card_ids.\n"
            "- Never invent card ids outside the candidate list.\n\n"
            f"NEW_CARD:\n{json.dumps(incoming_payload, ensure_ascii=True, indent=2)}\n\n"
            f"CANDIDATE_CARDS:\n{json.dumps(candidates_for_llm, ensure_ascii=True, indent=2)}"
        )
        decision = default_decision
        for _ in range(self.card_update_dedup_config.llm_max_retries):
            try:
                response_text, _, _, _ = self.llm_service.generate(prompt)
            except Exception as exc:
                print(f"[MemoryPlatform] Dedup LLM decision call failed: {exc}")
                continue
            parsed = parse_llm_card_decision(response_text, candidate_ids=candidate_ids)
            if isinstance(parsed, dict):
                decision = parsed
                break
        return decision

    def _apply_update_actions(self, incoming_card: dict[str, Any], updates: list[dict[str, Any]]) -> list[str]:
        updated_ids: list[str] = []
        seen_ids: set[str] = set()
        for update in updates:
            card_id = str(update.get("card_id") or "").strip()
            existing_card = self.memory_cards.get(card_id)
            if not card_id or card_id in seen_ids or not isinstance(existing_card, dict):
                continue
            merged_card = merge_updated_card(existing_card, incoming_card, update)
            merged_card["id"] = card_id
            self._save_card_core(merged_card)
            seen_ids.add(card_id)
            updated_ids.append(card_id)
        return updated_ids

    def _save_card_core(self, card: dict[str, Any]) -> str:
        card = normalize_memory_card(card)
        card_id = self._ensure_card_id(card)

        if self.enable_llm_card_enrichment and self.memory_system is not None:
            analysis = self.memory_system.analyze_content(card["description"])
            if not card.get("keywords"):
                card["keywords"] = analysis.get("keywords") or []
            if not card.get("task_description"):
                card["task_description"] = analysis.get("context") or ""

        content = self._card_to_backend_content(card)
        name, tags, when_to_use = self._build_entity_meta(card)

        current_entity_id = self.entity_by_card_id.get(card_id)
        ref = self.client.save_memory_card(
            content,
            name=name,
            tags=tags,
            when_to_use=when_to_use,
            namespace=self.namespace,
            author=self.author,
            entity_id=current_entity_id,
            channel=self.channel,
        )

        entity_id = str(ref.entity_id)
        if current_entity_id and current_entity_id != entity_id:
            self.card_id_by_entity.pop(current_entity_id, None)
            self.entity_version_by_entity.pop(current_entity_id, None)
        self.entity_by_card_id[card_id] = entity_id
        self.card_id_by_entity[entity_id] = card_id
        self.entity_version_by_entity[entity_id] = str(ref.version_id or "")
        self.memory_cards[card_id] = normalize_memory_card(card, fallback_id=card_id)
        self.memory_ids.add(card_id)
        self._persist_index()
        self._dedup_retrievers = None

        self._iters_after_rebuild += 1
        if self._iters_after_rebuild >= self.rebuild_interval:
            self.rebuild()
        return card_id

    def save_card(self, card: dict[str, Any]) -> str:
        normalized_card = normalize_memory_card(card)
        self.card_write_stats["processed"] += 1
        incoming_card_id = str(normalized_card.get("id") or "").strip()
        if incoming_card_id and incoming_card_id in self.memory_cards:
            self.card_write_stats["updated"] += 1
            return self._save_card_core(normalized_card)

        if self._is_program_card(normalized_card):
            self.card_write_stats["added"] += 1
            return self._save_card_core(normalized_card)

        if self.card_update_dedup_config.enabled and self.memory_cards and self.llm_service is None and not self._warned_missing_card_update_llm:
            print(
                "[MemoryPlatform] card_update_dedup is enabled but LLM service is unavailable. "
                "Falling back to regular save_card behavior."
            )
            self._warned_missing_card_update_llm = True

        if self.card_update_dedup_config.enabled and self.memory_cards and self.llm_service is not None:
            scored_candidates = self._score_retrieved_candidates(normalized_card)
            candidates_for_llm = self._dedup_candidates_for_llm(scored_candidates)
            decision = self._decide_card_action(normalized_card, candidates_for_llm)
            action = str(decision.get("action") or "add").strip().lower()
            if action == "discard":
                duplicate_id = str(decision.get("duplicate_of") or "").strip()
                if duplicate_id in self.memory_cards:
                    self.card_write_stats["rejected"] += 1
                    return duplicate_id
            elif action == "update":
                updates = decision.get("updates")
                updated_ids = self._apply_update_actions(
                    normalized_card,
                    updates if isinstance(updates, list) else [],
                )
                if updated_ids:
                    self.card_write_stats["updated"] += 1
                    self.card_write_stats["updated_target_cards"] += len(updated_ids)
                    return updated_ids[0]

        self.card_write_stats["added"] += 1
        return self._save_card_core(normalized_card)

    def save(self, data: str, category: str = "general") -> str:
        return self.save_card(
            {
                "category": category,
                "description": data,
                "task_description": "",
                "task_description_summary": "",
                "strategy": "",
                "last_generation": 0,
                "programs": [],
                "aliases": [],
                "keywords": [],
                "evolution_statistics": {},
                "explanation": {"explanations": [], "summary": ""},
                "works_with": [],
                "links": [],
                "usage": {},
            }
        )

    def _format_search_results(self, query: str, cards: list[dict[str, Any]]) -> str:
        lines = [f"Query: {query}", "", "Top relevant memory cards:"]
        for idx, card in enumerate(cards, start=1):
            card_id = str(card.get("id") or "")
            category = str(card.get("category") or "general")
            description = str(card.get("description") or "").strip()
            lines.append(f"{idx}. {card_id} [{category}] {description}")
        return "\n".join(lines)

    def _synthesize_results(self, query: str, memory_state: str | None, cards: list[dict[str, Any]]) -> str:
        if self.llm_service is None:
            return self._format_search_results(query, cards)

        cards_blob = []
        for card in cards:
            explanation = card.get("explanation")
            explanation_summary = explanation.get("summary", "") if isinstance(explanation, dict) else str(explanation or "")
            cards_blob.append(
                "\n".join(
                    [
                        f"id: {card.get('id', '')}",
                        f"category: {card.get('category', '')}",
                        f"task_description_summary: {card.get('task_description_summary', '')}",
                        f"task_description: {card.get('task_description', '')}",
                        f"description: {card.get('description', '')}",
                        f"keywords: {card.get('keywords', [])}",
                        f"explanation: {explanation_summary}",
                    ]
                )
            )

        prompt = (
            "You are a memory retrieval assistant.\n"
            "Use only the provided memory cards to answer the user query.\n"
            "Always cite card ids explicitly.\n"
            f"Memory state:\n{memory_state or '(empty)'}\n\n"
            f"User query:\n{query}\n\n"
            "Retrieved cards:\n"
            + "\n\n".join(cards_blob)
            + "\n\nAnswer:"
        )
        try:
            text, _, _, _ = self.llm_service.generate(prompt)
            text = str(text or "").strip()
            if text:
                return text
        except Exception as exc:
            print(f"[MemoryPlatform] LLM synthesis failed, fallback to plain output: {exc}")
        return self._format_search_results(query, cards)

    def _search_via_backend(self, query: str, memory_state: str | None = None) -> str:
        effective_query = query.strip()
        if memory_state:
            effective_query = f"{effective_query}\n{memory_state.strip()}"
        remote_search_type = str(self.remote_vector_search_type or "vector").strip().lower()
        if remote_search_type == "hybrid":
            search_type = SearchType.HYBRID
        elif remote_search_type == "bm25":
            search_type = SearchType.BM25
        else:
            search_type = SearchType.VECTOR
        hits = self.client.search_hits(
            query=effective_query,
            search_type=search_type,
            top_k=self.search_limit,
            entity_type="memory_card",
            namespace=self.namespace,
            channel=self.channel,
            document_kind=DOCUMENT_KIND_FULL_CARD,
            hybrid_weights=self.remote_hybrid_weights,
        )
        cards = []
        for hit in hits:
            content = hit.content or {}
            card = normalize_memory_card(content, fallback_id=hit.entity_id)
            card_id = self._ensure_card_id(card)
            entity_id = str(hit.entity_id)
            self.memory_cards[card_id] = card
            self.entity_by_card_id[card_id] = entity_id
            self.card_id_by_entity[entity_id] = card_id
            self.entity_version_by_entity[entity_id] = str(hit.version_id or "")
            cards.append(card)

        self._persist_index()
        if not cards:
            return f"Query: {query}\n\nNo relevant memories found."
        if self.enable_llm_synthesis:
            return self._synthesize_results(query, memory_state, cards)
        return self._format_search_results(query, cards)

    def search(self, query: str, memory_state: str | None = None) -> str:
        self.refresh_from_backend()
        self.rebuild()
        if self.research_agent is not None:
            try:
                return self.research_agent.research(query, memory_state=memory_state).integrated_memory
            except Exception as exc:
                print(f"[MemoryPlatform] GAM search failed, falling back to backend search: {exc}")
        return self._search_via_backend(query, memory_state=memory_state)

    def get_card(self, card_id: str) -> dict[str, Any] | None:
        return self.memory_cards.get(card_id)

    def get_card_write_stats(self) -> dict[str, int]:
        return dict(self.card_write_stats)

    def delete(self, memory_id: str) -> bool:
        key = str(memory_id).strip()
        entity_id = self.entity_by_card_id.get(key)
        if not entity_id and self._looks_like_uuid(key):
            entity_id = key
        if not entity_id:
            return False

        self.client.delete_memory_card(entity_id)
        card_id = self.card_id_by_entity.pop(entity_id, key)
        self.entity_version_by_entity.pop(entity_id, None)
        self.entity_by_card_id.pop(card_id, None)
        self.memory_cards.pop(card_id, None)
        self.memory_ids.discard(card_id)
        self._persist_index()
        if self.generator is not None:
            self.rebuild()
        return True

    def close(self) -> None:
        self.client.close()

    def __del__(self) -> None:
        try:
            if self._iters_after_rebuild > 0:
                self.rebuild()
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass
