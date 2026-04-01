from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any
import uuid

from dotenv import load_dotenv
import httpx

_THIS_DIR = Path(__file__).resolve().parent
_AGENT_ROOT = _THIS_DIR.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from openai_inference import OpenAIInferenceService

import config

try:
    from .card_update_dedup import (
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
except ImportError:  # pragma: no cover - direct script execution fallback
    from shared_memory.card_update_dedup import (
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

if TYPE_CHECKING:
    pass


_ALLOWED_STRATEGIES = {"exploration", "exploitation", "hybrid"}
_VECTOR_GAM_TOOLS = {
    "vector",
    "vector_description",
    "vector_task_description",
    "vector_explanation_summary",
    "vector_description_explanation_summary",
    "vector_description_task_description_summary",
}
_ALLOWED_GAM_TOOLS = {
    "keyword",
    "page_index",
    *_VECTOR_GAM_TOOLS,
}
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


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _memory_to_card(
    memory_note: Any,
    base_card: dict[str, Any] | None = None,
    memory_id: str | None = None,
) -> dict[str, Any]:
    mem_id = _safe_get(memory_note, "id", None) or memory_id
    card = normalize_memory_card(base_card, fallback_id=mem_id)
    if memory_note is None:
        return card

    card["id"] = str(mem_id or card["id"])
    card["category"] = str(card.get("category") or _safe_get(memory_note, "category", None) or "general")
    card["description"] = str(card.get("description") or _safe_get(memory_note, "content", ""))
    card["task_description"] = str(card.get("task_description") or _safe_get(memory_note, "context", ""))
    if str(card.get("category") or "").strip().lower() == "program":
        return card
    card["strategy"] = str(card.get("strategy") or _safe_get(memory_note, "strategy", ""))
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


def _export_memories_jsonl(
    memory_system: Any,
    memory_ids: list[str],
    out_path: Path,
    card_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
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
            record = _memory_to_card(memory_note, base_card=base_card, memory_id=memory_id)
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")


class GigaEvoMemoryBase:
    def save(self, data: str) -> str:
        raise NotImplementedError

    def search(self, query: str) -> str:
        raise NotImplementedError

    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError


class _ConceptApiClient:
    """Small HTTP client around Memory API entity endpoints.

    NOTE: despite the class name, this client now targets the newer Memory API
    entity model:
    - memory cards: `/v1/memory-cards`
    - search: `/v1/search/batch`
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        base = base_url.rstrip("/")
        # Accept both "http://host:port" and "http://host:port/..." inputs.
        # If the caller accidentally points at the service root (e.g. :15010),
        # we keep it; if they point at an older reverse-proxy path, the relative
        # URLs below still work.
        self._http = httpx.Client(base_url=base, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            host = str(self._http.base_url).rstrip("/")
            raise RuntimeError(
                f"Cannot connect to Memory API at {host}. "
                "Start the API service or set MEMORY_API_URL to a reachable endpoint."
            ) from exc
        except httpx.TimeoutException as exc:
            host = str(self._http.base_url).rstrip("/")
            raise RuntimeError(
                f"Memory API request timed out for {host}. "
                "Check service health and network connectivity."
            ) from exc
        if response.status_code == 204:
            return None
        if response.status_code >= 400:
            raise RuntimeError(
                f"Memory API request failed ({method} {path}): "
                f"{response.status_code} {response.text}"
            )
        return response.json()

    def save_concept(
        self,
        *,
        content: dict[str, Any],
        name: str,
        tags: list[str],
        when_to_use: str,
        channel: str,
        namespace: str | None,
        author: str | None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "meta": {
                "name": name,
                "tags": tags,
                "when_to_use": when_to_use,
                "namespace": namespace,
                "author": author,
            },
            "channel": channel,
            "content": content,
        }
        if entity_id:
            result = self._request("PUT", f"/v1/memory-cards/{entity_id}", json=body)
        else:
            result = self._request("POST", "/v1/memory-cards", json=body)
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected empty response from concept save")
        return result

    def get_concept(self, entity_id: str, channel: str = "latest") -> dict[str, Any]:
        result = self._request(
            "GET",
            f"/v1/memory-cards/{entity_id}",
            params={"channel": channel},
        )
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected empty response from concept get")
        return result

    def list_memory_cards(
        self,
        *,
        limit: int,
        offset: int = 0,
        channel: str = "latest",
    ) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            "/v1/memory-cards",
            params={"limit": int(limit), "offset": int(offset), "channel": channel},
        )
        if not isinstance(result, list):
            return []
        items: list[dict[str, Any]] = []
        for row in result:
            if isinstance(row, dict):
                items.append(row)
        return items

    def search_concepts(
        self,
        *,
        query: str | None,
        limit: int,
        namespace: str | None,
        offset: int = 0,
    ) -> dict[str, Any]:
        query_text = str(query or "").strip()
        if not query_text:
            # The new API doesn't support "list everything" via search; use list_memory_cards().
            return {"hits": [], "total": 0}

        # New API: POST /v1/search/batch
        payload: dict[str, Any] = {
            "queries": [query_text],
            "top_k": int(limit),
            "entity_type": "memory_card",
            "channel": "latest",
            "search_type": "bm25",
        }
        if namespace:
            payload["namespace"] = namespace

        result = self._request("POST", "/v1/search/batch", json=payload) or {}
        if not isinstance(result, dict):
            return {"hits": [], "total": 0}

        results = result.get("results")
        if not isinstance(results, list) or not results:
            return {"hits": [], "total": 0}
        first = results[0]
        if not isinstance(first, list):
            return {"hits": [], "total": 0}

        hits: list[dict[str, Any]] = [h for h in first if isinstance(h, dict)]
        return {"hits": hits, "total": len(hits)}

    def delete_concept(self, entity_id: str) -> None:
        self._request("DELETE", f"/v1/memory-cards/{entity_id}")


class AmemGamMemory(GigaEvoMemoryBase):
    """API-backed memory where API is the source of truth and local GAM is retrieval runtime."""

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
    ):
        self.checkpoint_dir = Path(checkpoint_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.checkpoint_dir / "api_index.json"
        self.export_file = self.checkpoint_dir / "amem_exports" / "amem_memories.jsonl"
        self.gam_store_dir = self.checkpoint_dir / "gam_shared" / "amem_store"

        self.use_api = bool(use_api)
        self.namespace = namespace
        self.author = author
        self.channel = channel
        self.search_limit = search_limit
        self.rebuild_interval = rebuild_interval
        self.enable_bm25 = enable_bm25
        self.sync_batch_size = max(10, int(sync_batch_size))
        self.enable_llm_synthesis = enable_llm_synthesis
        self.enable_memory_evolution = bool(enable_memory_evolution)
        self.enable_llm_card_enrichment = bool(enable_llm_card_enrichment)
        self.allowed_gam_tools = self._normalize_allowed_gam_tools(allowed_gam_tools)
        self.gam_top_k_by_tool = self._normalize_gam_top_k_by_tool(gam_top_k_by_tool)
        self.gam_pipeline_mode = self._normalize_gam_pipeline_mode(gam_pipeline_mode)
        self.card_update_dedup_config = CardUpdateDedupConfig.from_mapping(
            card_update_dedup_config or {}
        )
        self._warned_missing_card_update_llm = False
        self._iters_after_rebuild = 0

        self.api: _ConceptApiClient | None = None
        if self.use_api:
            self.api = _ConceptApiClient(base_url=base_url)
        else:
            print("[Memory] API mode disabled. Running in local-only mode.")

        self._AgenticMemorySystemCls: type[Any] | None = None
        self._MemoryNoteCls: type[Any] | None = None
        self._ResearchAgentCls: type[Any] | None = None
        self._AMemGeneratorCls: type[Any] | None = None
        self._agentic_import_error: Exception | None = None
        self._load_agentic_classes()

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
        self._load_index()

        self.llm_service, self.generator = self._init_llm_service_and_generator()
        self.memory_system = self._init_storage()
        self.research_agent: Any | None = None
        self._dedup_retrievers: dict[str, Any] | None = None

        if self.memory_system is not None and self.generator is not None and self.export_file.exists():
            try:
                self.research_agent = self._load_or_create_retriever()
            except Exception as exc:
                print(f"[Memory] Initial retriever load skipped: {exc}")

        if sync_on_init and self.use_api:
            self._sync_from_api(force_full=True)

    @staticmethod
    def _normalize_allowed_gam_tools(allowed_gam_tools: list[str] | None) -> set[str]:
        if not allowed_gam_tools:
            return set(_ALLOWED_GAM_TOOLS)

        normalized = {
            str(tool).strip()
            for tool in allowed_gam_tools
            if str(tool).strip()
        }
        valid = {tool for tool in normalized if tool in _ALLOWED_GAM_TOOLS}
        if "vector" in valid:
            # Backward compatibility: opting into "vector" enables all vector-backed tools.
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
        if mode in _ALLOWED_GAM_PIPELINE_MODES:
            return mode
        return "default"

    def _load_agentic_classes(self) -> None:
        try:
            from A_mem.agentic_memory.memory_system import (
                AgenticMemorySystem as _AgenticMemorySystem,
            )
            from A_mem.agentic_memory.memory_system import (
                MemoryNote as _MemoryNote,
            )
            from GAM_root.gam import ResearchAgent as _ResearchAgent
            from GAM_root.gam.generator import AMemGenerator as _AMemGenerator
        except Exception as exc:
            self._agentic_import_error = exc
            print(
                "[Memory] Agentic runtime dependencies are unavailable. "
                f"Reason: {exc}. Falling back to API full-text mode."
            )
            return

        self._AgenticMemorySystemCls = _AgenticMemorySystem
        self._MemoryNoteCls = _MemoryNote
        self._ResearchAgentCls = _ResearchAgent
        self._AMemGeneratorCls = _AMemGenerator

    def _init_llm_service_and_generator(self) -> tuple[Any | None, Any | None]:
        if self._AMemGeneratorCls is None and not self.card_update_dedup_config.enabled:
            return None, None
        api_key = config.OPENAI_API_KEY
        if not api_key and config.LLM_BASE_URL:
            # Local OpenAI-compatible servers (vLLM/LM Studio/Ollama OpenAI mode)
            # often accept any non-empty bearer token.
            api_key = "EMPTY"

        if not api_key:
            print(
                "[Memory] OPENAI_API_KEY/OPENROUTER_API_KEY is not set. "
                "Agentic retrieval is disabled; API full-text fallback is available."
            )
            return None, None

        try:
            base_url = config.LLM_BASE_URL

            llm_service = OpenAIInferenceService(
                model_name=config.OPENROUTER_MODEL_NAME,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                max_tokens=0,
                reasoning=config.OPENROUTER_REASONING,
            )
            if self._AMemGeneratorCls is None:
                return llm_service, None
            generator = self._AMemGeneratorCls({"llm_service": llm_service})
            return llm_service, generator
        except Exception as exc:
            print(f"[Memory] Could not initialize LLM/generator: {exc}")
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
                chroma_collection_name="memories",
                use_gam_card_document=True,
                enable_evolution=self.enable_memory_evolution,
            )
        except Exception as exc:
            print(f"[Memory] Could not initialize AgenticMemorySystem: {exc}")
            return None

    def _load_index(self) -> None:
        if not self.index_file.exists():
            return

        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Memory] Could not parse index file {self.index_file}: {exc}")
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
                if not cid or not eid:
                    continue
                self.entity_by_card_id[cid] = eid
                self.card_id_by_entity[eid] = cid

        if isinstance(raw_versions, dict):
            for entity_id, version_id in raw_versions.items():
                eid = str(entity_id)
                vid = str(version_id or "")
                if eid:
                    self.entity_version_by_entity[eid] = vid

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

    def _card_to_concept_content(self, card: dict[str, Any]) -> dict[str, Any]:
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
        if isinstance(explanation, dict):
            explanation_text = str(explanation.get("summary") or "")
        else:
            explanation_text = str(explanation or "")

        strategy = str(card.get("strategy") or "").strip().lower() or None
        if strategy not in _ALLOWED_STRATEGIES:
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
            "program_id": str(card.get("program_id") or ""),
            "fitness": _to_float(card.get("fitness"), default=None),
            "task_description": str(card.get("task_description") or ""),
            "task_description_summary": str(card.get("task_description_summary") or ""),
            "description": str(card.get("description") or ""),
            "code": str(card.get("code") or ""),
            "connected_ideas": _to_list(card.get("connected_ideas")),
            "explanation": explanation_text,
            "strategy": strategy,
            "keywords": self._dedupe_keep_order(list(card.get("keywords") or [])),
            "evolution_statistics": evolution_statistics,
            "works_with": self._dedupe_keep_order(list(card.get("works_with") or [])),
            "links": self._dedupe_keep_order(list(card.get("links") or [])),
            "usage": usage,
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

        when_to_use_parts = self._dedupe_keep_order(
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

    def _concept_to_card(self, concept_content: dict[str, Any], fallback_id: str) -> dict[str, Any]:
        return normalize_memory_card(
            {
                "id": concept_content.get("id") or fallback_id,
                "category": concept_content.get("category") or "general",
                "program_id": concept_content.get("program_id") or "",
                "fitness": concept_content.get("fitness"),
                "description": concept_content.get("description") or "",
                "task_description": concept_content.get("task_description") or "",
                "task_description_summary": concept_content.get("task_description_summary")
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

    def _note_metadata(self, note: Any) -> dict[str, Any]:
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

    def _build_note_from_card(self, card: dict[str, Any]) -> Any:
        if self._MemoryNoteCls is None:
            raise RuntimeError("MemoryNote class is unavailable")
        card_id = str(card.get("id") or "")
        description = str(card.get("description") or "")
        context = str(
            card.get("task_description")
            or card.get("task_description_summary")
            or "General"
        )
        category = str(card.get("category") or "general")
        strategy = str(card.get("strategy") or "")
        keywords = list(card.get("keywords") or [])
        links = list(card.get("links") or [])
        existing = self.memory_system.read(card_id) if self.memory_system is not None else None

        return self._MemoryNoteCls(
            content=description,
            id=card_id,
            keywords=keywords,
            links=links,
            retrieval_count=(existing.retrieval_count if existing is not None else 0),
            timestamp=(existing.timestamp if existing is not None else None),
            last_accessed=(existing.last_accessed if existing is not None else None),
            context=context or "General",
            evolution_history=(existing.evolution_history if existing is not None else None),
            category=category,
            tags=(existing.tags if existing is not None else []),
            strategy=strategy,
        )

    def _upsert_local_note_fast(self, card: dict[str, Any]) -> bool:
        """Synchronize card into local A-MEM/Chroma without running LLM evolution."""
        if self.memory_system is None:
            return False

        note = self._build_note_from_card(card)
        existing = self.memory_system.read(note.id)
        changed = (
            existing is None
            or existing.content != note.content
            or existing.category != note.category
            or existing.context != note.context
            or existing.strategy != note.strategy
            or existing.keywords != note.keywords
            or existing.links != note.links
        )
        if not changed:
            self.memory_ids.add(note.id)
            return False

        self.memory_system.memories[note.id] = note
        try:
            self.memory_system.retriever.delete_document(note.id)
        except Exception:
            pass
        self.memory_system.retriever.add_document(
            self.memory_system._document_for_note(note),
            self._note_metadata(note),
            note.id,
        )
        self.memory_ids.add(note.id)
        return True

    def _upsert_local_note_agentic(self, card: dict[str, Any]) -> bool:
        """Add/update card in local A-MEM using regular add/update path for local writes."""
        if self.memory_system is None:
            return False

        card_id = str(card.get("id") or "").strip()
        if not card_id:
            return False

        description = str(card.get("description") or "")
        kwargs = {
            "category": str(card.get("category") or "general"),
            "keywords": list(card.get("keywords") or []),
            "context": str(
                card.get("task_description")
                or card.get("task_description_summary")
                or "General"
            ),
            "strategy": str(card.get("strategy") or ""),
            "links": list(card.get("links") or []),
            "tags": [],
        }

        existing = self.memory_system.read(card_id)
        if existing is None:
            self.memory_system.add_note(id=card_id, content=description, **kwargs)
        else:
            changed = (
                existing.content != description
                or existing.category != kwargs["category"]
                or existing.context != kwargs["context"]
                or existing.strategy != kwargs["strategy"]
                or existing.keywords != kwargs["keywords"]
                or existing.links != kwargs["links"]
            )
            if not changed:
                self.memory_ids.add(card_id)
                return False
            self.memory_system.update(card_id, content=description, **kwargs)

        self.memory_ids.add(card_id)
        return True

    def _remove_local_note(self, card_id: str) -> bool:
        if self.memory_system is None:
            self.memory_ids.discard(card_id)
            return False
        deleted = self.memory_system.delete(card_id)
        self.memory_ids.discard(card_id)
        return deleted

    def _fetch_all_concept_hits(self) -> list[dict[str, Any]]:
        if not self.use_api or self.api is None:
            return []
        hits: list[dict[str, Any]] = []
        offset = 0
        while True:
            # New API: list memory cards via `/v1/memory-cards` and filter by namespace client-side.
            rows = self.api.list_memory_cards(
                limit=self.sync_batch_size,
                offset=offset,
                channel=self.channel,
            )
            if not rows:
                break

            page_hits: list[dict[str, Any]] = []
            for row in rows:
                meta = row.get("meta") if isinstance(row, dict) else None
                row_namespace = None
                if isinstance(meta, dict):
                    row_namespace = meta.get("namespace")
                if self.namespace and row_namespace not in (None, "", self.namespace):
                    continue
                page_hits.append(row)

            # Advance offset by raw rows count to preserve pagination even with filtering.
            offset += len(rows)
            hits.extend(page_hits)

            if len(rows) < self.sync_batch_size:
                break
        return hits

    def _sync_from_api(self, force_full: bool = False) -> bool:
        if not self.use_api or self.api is None:
            return False
        remote_hits = self._fetch_all_concept_hits()
        remote_entity_ids: set[str] = set()
        changed = False

        for hit in remote_hits:
            entity_id = str(hit.get("entity_id") or "").strip()
            if not entity_id:
                continue
            remote_entity_ids.add(entity_id)
            remote_version = str(hit.get("version_id") or "").strip()

            known_card_id = self.card_id_by_entity.get(entity_id)
            known_version = self.entity_version_by_entity.get(entity_id, "")
            if (
                not force_full
                and known_card_id
                and remote_version
                and known_version == remote_version
            ):
                if (
                    self.memory_system is not None
                    and self.memory_system.read(known_card_id) is None
                    and known_card_id in self.memory_cards
                ):
                    if self._upsert_local_note_fast(self.memory_cards[known_card_id]):
                        changed = True
                self.memory_ids.add(known_card_id)
                continue

            concept = self.api.get_concept(entity_id, channel=self.channel)
            content = concept.get("content") or {}
            fallback_id = self.card_id_by_entity.get(entity_id) or str(content.get("id") or entity_id)
            card = self._concept_to_card(content, fallback_id=fallback_id)
            card_id = self._ensure_card_id(card)

            previous_card_id = self.card_id_by_entity.get(entity_id)
            if previous_card_id and previous_card_id != card_id:
                self.entity_by_card_id.pop(previous_card_id, None)
                self.memory_cards.pop(previous_card_id, None)
                self._remove_local_note(previous_card_id)
                changed = True

            self.card_id_by_entity[entity_id] = card_id
            self.entity_by_card_id[card_id] = entity_id
            self.entity_version_by_entity[entity_id] = str(
                concept.get("version_id") or remote_version or ""
            )

            old_card = self.memory_cards.get(card_id)
            if old_card != card:
                changed = True
            self.memory_cards[card_id] = card

            if self._upsert_local_note_fast(card):
                changed = True

        stale_entities = [eid for eid in self.card_id_by_entity if eid not in remote_entity_ids]
        for entity_id in stale_entities:
            card_id = self.card_id_by_entity.pop(entity_id, None)
            self.entity_version_by_entity.pop(entity_id, None)
            if card_id:
                self.entity_by_card_id.pop(card_id, None)
                self.memory_cards.pop(card_id, None)
                self._remove_local_note(card_id)
            changed = True

        if changed:
            self.rebuild()
        else:
            self._persist_index()
            if self.research_agent is None and self.memory_system is not None and self.generator is not None:
                self.rebuild()

        return changed

    def _load_or_create_retriever(self) -> Any | None:
        if self.generator is None or self._ResearchAgentCls is None:
            raise RuntimeError("Generator is not available. Cannot create GAM research agent.")
        try:
            from shared_memory.amem_gam_retriever import (
                build_gam_store,
                build_retrievers,
                load_amem_records,
            )
        except Exception as exc:
            raise RuntimeError(f"GAM helper modules are unavailable: {exc}") from exc

        self.gam_store_dir.mkdir(parents=True, exist_ok=True)
        if self.export_file.exists():
            records = load_amem_records(self.export_file)
        else:
            records = list(self.memory_cards.values())

        memory_store, page_store, added = build_gam_store(records, self.gam_store_dir)
        print(f"[Memory] Loaded {len(records)} cards, added {added} new pages.")

        retrievers = build_retrievers(
            page_store,
            self.gam_store_dir / "indexes",
            self.checkpoint_dir / "chroma",
            enable_bm25=self.enable_bm25,
            allowed_tools=sorted(self.allowed_gam_tools),
        )
        retrievers = {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self.allowed_gam_tools
        }
        if not retrievers:
            print(
                "[Memory] No GAM retrievers enabled after applying allowed_gam_tools. "
                "GAM agentic search is disabled."
            )
            return None
        return self._ResearchAgentCls(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=self.generator,
            max_iters=3,
            allowed_tools=sorted(self.allowed_gam_tools),
            top_k_by_tool=self.gam_top_k_by_tool,
            pipeline_mode=self.gam_pipeline_mode,
        )

    def _dump_memory(self) -> None:
        if self.memory_system is None:
            return
        all_ids = sorted(set(self.memory_ids) | set(self.memory_cards.keys()))
        _export_memories_jsonl(
            self.memory_system,
            all_ids,
            self.export_file,
            card_overrides=self.memory_cards,
        )

    @staticmethod
    def _truncate_text(value: Any, max_chars: int = 1200) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _build_dedup_retrievers(self) -> dict[str, Any]:
        try:
            from .amem_gam_retriever import (
                build_gam_store,
                build_retrievers,
                load_amem_records,
            )
        except Exception as exc:
            try:
                from shared_memory.amem_gam_retriever import (
                    build_gam_store,
                    build_retrievers,
                    load_amem_records,
                )
            except Exception:
                print(f"[Memory] Dedup retriever import failed: {exc}")
                return {}

        self.gam_store_dir.mkdir(parents=True, exist_ok=True)
        if self.export_file.exists():
            try:
                records = load_amem_records(self.export_file)
            except Exception:
                records = list(self.memory_cards.values())
        else:
            records = list(self.memory_cards.values())
        records = [record for record in records if not self._is_program_card(record)]
        if not records:
            return {}

        try:
            _, page_store, _ = build_gam_store(records, self.gam_store_dir)
            retrievers = build_retrievers(
                page_store,
                self.gam_store_dir / "indexes",
                self.checkpoint_dir / "chroma",
                enable_bm25=False,
                allowed_tools=[
                    "vector_description",
                    "vector_explanation_summary",
                    "vector_description_explanation_summary",
                    "vector_description_task_description_summary",
                ],
            )
        except Exception as exc:
            print(f"[Memory] Dedup retriever build failed: {exc}")
            return {}

        return {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self.allowed_gam_tools
        }

    def _resolve_vector_retriever(self, tool_name: str) -> Any | None:
        if self._dedup_retrievers is None:
            self._dedup_retrievers = self._build_dedup_retrievers()
        retrievers = self._dedup_retrievers or {}
        if not retrievers:
            return None

        retriever = retrievers.get(tool_name)
        if retriever is None and tool_name != "vector":
            retriever = retrievers.get("vector")
        return retriever

    def _score_retrieved_candidates(
        self,
        card: dict[str, Any],
    ) -> list[dict[str, Any]]:
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
            text = str(query_text or "").strip()
            if not text:
                continue

            retriever = self._resolve_vector_retriever(tool_by_key[query_key])
            if retriever is None:
                continue

            try:
                hits_by_query = retriever.search([text], top_k=cfg.top_k_per_query)
            except Exception as exc:
                print(
                    f"[Memory] Dedup retrieval failed for query '{query_key}': {exc}"
                )
                continue

            hits = []
            if isinstance(hits_by_query, list) and hits_by_query:
                first = hits_by_query[0]
                if isinstance(first, list):
                    hits = first
                else:
                    hits = hits_by_query

            query_scores: dict[str, float] = {}
            for hit in hits:
                card_id = str(getattr(hit, "page_id", "") or "").strip()
                if not card_id or card_id not in self.memory_cards:
                    continue
                if self._is_program_card(self.memory_cards[card_id]):
                    continue
                meta = getattr(hit, "meta", {}) or {}
                try:
                    score = float(meta.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                if score <= 0:
                    continue
                previous_score = query_scores.get(card_id, 0.0)
                if score > previous_score:
                    query_scores[card_id] = score
            scores_by_query[query_key] = query_scores

        return compute_weighted_candidates(
            scores_by_query,
            weights=cfg.weights,
            final_top_n=cfg.final_top_n,
            min_final_score=cfg.min_final_score,
        )

    def _dedup_candidates_for_llm(
        self,
        scored_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in scored_candidates:
            card_id = str(item.get("card_id") or "").strip()
            if not card_id:
                continue
            card = self.memory_cards.get(card_id)
            if not isinstance(card, dict):
                continue

            explanations = get_full_explanations(card)
            payload.append(
                {
                    "card_id": card_id,
                    "final_score": float(item.get("final_score", 0.0)),
                    "scores": item.get("scores", {}),
                    "task_description_summary": self._truncate_text(
                        card.get("task_description_summary"), 600
                    ),
                    "description": self._truncate_text(card.get("description"), 1200),
                    "explanation_summary": self._truncate_text(
                        get_explanation_summary(card), 600
                    ),
                    "explanation_full": [
                        self._truncate_text(explanation, 1200)
                        for explanation in explanations
                    ],
                }
            )
        return payload

    def _decide_card_action(
        self,
        incoming_card: dict[str, Any],
        candidates_for_llm: list[dict[str, Any]],
    ) -> dict[str, Any]:
        default_decision = {
            "action": "add",
            "reason": "",
            "duplicate_of": "",
            "updates": [],
        }
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
            "task_description_summary": self._truncate_text(
                incoming_card.get("task_description_summary"), 600
            ),
            "description": self._truncate_text(incoming_card.get("description"), 1200),
            "explanation_summary": self._truncate_text(
                get_explanation_summary(incoming_card), 600
            ),
            "explanation_full": [
                self._truncate_text(explanation, 1200)
                for explanation in get_full_explanations(incoming_card)
            ],
        }
        prompt = (
            "You are a memory-card deduplication and update policy agent.\n"
            "For NEW_CARD choose exactly one action:\n"
            "- add: NEW_CARD is genuinely new and should be saved as a new memory card.\n"
            "- discard: one existing card already represents the same idea.\n"
            "- update: idea exists, but NEW_CARD adds a new task/use-case and/or new explanation details.\n\n"
            "Return only JSON with this schema:\n"
            "{\n"
            '  "action": "add|discard|update",\n'
            '  "reason": "short reason",\n'
            '  "duplicate_of": "card_id or empty",\n'
            '  "updates": [\n'
            "    {\n"
            '      "card_id": "candidate card id",\n'
            '      "update_task_description": true|false,\n'
            '      "task_description_append": "text to append or empty",\n'
            '      "task_description_summary": "updated summary or empty",\n'
            '      "update_explanation": true|false,\n'
            '      "explanation_append": "full explanation text to append or empty",\n'
            '      "explanation_summary": "updated summary or empty"\n'
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
                print(f"[Memory] Dedup LLM decision call failed: {exc}")
                continue
            parsed = parse_llm_card_decision(
                response_text,
                candidate_ids=candidate_ids,
            )
            if isinstance(parsed, dict):
                decision = parsed
                break
        return decision

    def _apply_update_actions(
        self,
        incoming_card: dict[str, Any],
        updates: list[dict[str, Any]],
    ) -> list[str]:
        updated_ids: list[str] = []
        seen_ids: set[str] = set()
        for update in updates:
            if not isinstance(update, dict):
                continue
            card_id = str(update.get("card_id") or "").strip()
            if not card_id or card_id in seen_ids:
                continue
            existing_card = self.memory_cards.get(card_id)
            if not isinstance(existing_card, dict):
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

        content = self._card_to_concept_content(card)
        name, tags, when_to_use = self._build_entity_meta(card)

        if self.use_api and self.api is not None:
            current_entity_id = self.entity_by_card_id.get(card_id)
            response = self.api.save_concept(
                content=content,
                name=name,
                tags=tags,
                when_to_use=when_to_use,
                channel=self.channel,
                namespace=self.namespace,
                author=self.author,
                entity_id=current_entity_id,
            )

            saved_entity_id = str(response["entity_id"])
            if current_entity_id and current_entity_id != saved_entity_id:
                self.card_id_by_entity.pop(current_entity_id, None)
                self.entity_version_by_entity.pop(current_entity_id, None)

            self.entity_by_card_id[card_id] = saved_entity_id
            self.card_id_by_entity[saved_entity_id] = card_id
            self.entity_version_by_entity[saved_entity_id] = str(response.get("version_id") or "")
        else:
            stale_entity_id = self.entity_by_card_id.pop(card_id, None)
            if stale_entity_id:
                self.card_id_by_entity.pop(stale_entity_id, None)
                self.entity_version_by_entity.pop(stale_entity_id, None)
        self.memory_cards[card_id] = normalize_memory_card(card, fallback_id=card_id)

        self._upsert_local_note_agentic(self.memory_cards[card_id])
        self._persist_index()
        self._dedup_retrievers = None

        self._iters_after_rebuild += 1
        if self._iters_after_rebuild >= self.rebuild_interval:
            self.rebuild()

        return card_id

    @staticmethod
    def _is_program_card(card: dict[str, Any]) -> bool:
        if str(card.get("category") or "").strip().lower() == "program":
            return True
        return bool(str(card.get("program_id") or "").strip())

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

        if (
            self.card_update_dedup_config.enabled
            and self.memory_cards
            and self.llm_service is None
            and not self._warned_missing_card_update_llm
        ):
            print(
                "[Memory] card_update_dedup is enabled but LLM service is unavailable. "
                "Falling back to regular save_card behavior."
            )
            self._warned_missing_card_update_llm = True

        if (
            self.card_update_dedup_config.enabled
            and self.memory_cards
            and self.llm_service is not None
        ):
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

    def _synthesize_results(
        self,
        query: str,
        memory_state: str | None,
        cards: list[dict[str, Any]],
    ) -> str:
        if self.llm_service is None:
            return self._format_search_results(query, cards)

        cards_blob = []
        for card in cards:
            cards_blob.append(
                "\n".join(
                    [
                        f"id: {card.get('id', '')}",
                        f"category: {card.get('category', '')}",
                        f"task_description_summary: {card.get('task_description_summary', '')}",
                        f"task_description: {card.get('task_description', '')}",
                        f"description: {card.get('description', '')}",
                        f"keywords: {card.get('keywords', [])}",
                        f"explanation: {card.get('explanation', {}).get('summary', '') if isinstance(card.get('explanation'), dict) else card.get('explanation', '')}",
                    ]
                )
            )

        prompt = (
            "You are a memory retrieval assistant.\n"
            "Use only the provided memory cards to answer the user query.\n"
            "Always cite card ids explicitly (example: mem-029).\n"
            "If evidence is insufficient, say so clearly.\n\n"
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
            print(f"[Memory] LLM synthesis failed, fallback to plain output: {exc}")

        return self._format_search_results(query, cards)

    def _search_via_api(self, query: str, memory_state: str | None = None) -> str:
        if not self.use_api or self.api is None:
            return self._search_local_cards(query, memory_state)
        effective_query = query.strip()
        if memory_state:
            effective_query = f"{effective_query}\n{memory_state.strip()}"

        payload = self.api.search_concepts(
            query=effective_query,
            limit=self.search_limit,
            namespace=self.namespace,
            offset=0,
        )
        hits = payload.get("hits", [])
        if not hits:
            return f"Query: {query}\n\nNo relevant memories found."

        cards: list[dict[str, Any]] = []
        local_changed = False

        for hit in hits:
            entity_id = str(hit.get("entity_id") or "").strip()
            if not entity_id:
                continue

            concept = self.api.get_concept(entity_id, channel=self.channel)
            content = concept.get("content") or {}

            card_id = str(content.get("id") or self.card_id_by_entity.get(entity_id) or entity_id)
            card = self._concept_to_card(content, fallback_id=card_id)
            card_id = self._ensure_card_id(card)

            self.card_id_by_entity[entity_id] = card_id
            self.entity_by_card_id[card_id] = entity_id
            self.entity_version_by_entity[entity_id] = str(
                concept.get("version_id") or hit.get("version_id") or ""
            )
            self.memory_cards[card_id] = card
            cards.append(card)

            if self._upsert_local_note_fast(card):
                local_changed = True

        self._persist_index()
        if local_changed and self.memory_system is not None and self.generator is not None:
            self.rebuild()

        if not cards:
            return f"Query: {query}\n\nNo relevant memories found."

        if self.enable_llm_synthesis:
            return self._synthesize_results(query, memory_state, cards)
        return self._format_search_results(query, cards)

    def _search_local_cards(self, query: str, memory_state: str | None = None) -> str:
        if not self.memory_cards:
            return f"Query: {query}\n\nNo relevant memories found."

        query_text = f"{query} {memory_state or ''}".strip().lower()
        tokens = [tok for tok in re.split(r"\W+", query_text) if tok]
        if not tokens:
            tokens = [query.strip().lower()] if query.strip() else []

        scored: list[tuple[int, dict[str, Any]]] = []
        for card in self.memory_cards.values():
            haystack = " ".join(
                [
                    str(card.get("description") or ""),
                    str(card.get("task_description_summary") or ""),
                    str(card.get("task_description") or ""),
                    " ".join([str(x) for x in (card.get("keywords") or [])]),
                    str(card.get("category") or ""),
                ]
            ).lower()
            score = sum(1 for tok in tokens if tok and tok in haystack)
            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_cards = [card for _, card in scored[: self.search_limit]]

        if not top_cards:
            return f"Query: {query}\n\nNo relevant memories found."

        if self.enable_llm_synthesis:
            return self._synthesize_results(query, memory_state, top_cards)
        return self._format_search_results(query, top_cards)

    def search(self, query: str, memory_state: str | None = None) -> str:
        if self.use_api and self.api is not None:
            self._sync_from_api(force_full=False)

        if self.research_agent is not None:
            try:
                return self.research_agent.research(query, memory_state=memory_state).integrated_memory
            except Exception as exc:
                print(f"[Memory] GAM search failed, falling back to non-agentic search: {exc}")

        if self.use_api and self.api is not None:
            return self._search_via_api(query, memory_state=memory_state)
        return self._search_local_cards(query, memory_state=memory_state)

    def get_card(self, card_id: str) -> dict[str, Any] | None:
        return self.memory_cards.get(card_id)

    def get_card_write_stats(self) -> dict[str, int]:
        return dict(self.card_write_stats)

    def rebuild(self) -> None:
        self._persist_index()
        if self.memory_system is None or self.generator is None:
            return
        self._dump_memory()
        self.research_agent = self._load_or_create_retriever()
        self._dedup_retrievers = None
        self._iters_after_rebuild = 0

    def delete(self, memory_id: str) -> bool:
        key = str(memory_id).strip()
        if self.use_api and self.api is not None:
            entity_id = self.entity_by_card_id.get(key)
            if not entity_id and self._looks_like_uuid(key):
                entity_id = key
            if not entity_id:
                return False
            self.api.delete_concept(entity_id)
            card_id = self.card_id_by_entity.pop(entity_id, key)
            self.entity_version_by_entity.pop(entity_id, None)
        else:
            card_id = key
            if card_id not in self.memory_cards and key in self.card_id_by_entity:
                card_id = self.card_id_by_entity[key]
            if card_id not in self.memory_cards:
                return False
            entity_id = self.entity_by_card_id.get(card_id)
            if entity_id:
                self.card_id_by_entity.pop(entity_id, None)
                self.entity_version_by_entity.pop(entity_id, None)

        self.entity_by_card_id.pop(card_id, None)
        self.memory_cards.pop(card_id, None)
        self._remove_local_note(card_id)
        self._persist_index()

        if self.memory_system is not None and self.generator is not None:
            self.rebuild()

        return True

    def close(self) -> None:
        if self.api is not None:
            self.api.close()

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
