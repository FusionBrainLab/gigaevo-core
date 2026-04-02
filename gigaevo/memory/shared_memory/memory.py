from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
import uuid

from dotenv import load_dotenv
from loguru import logger

import gigaevo.memory.config as config
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

from gigaevo.memory.shared_memory.card_conversion import (
    DEFAULT_MODEL_NAME,
    AnyCard,
    GigaEvoMemoryBase,
    MemoryCard,
    MemoryNoteProtocol,
    build_entity_meta,
    card_to_concept_content,
    concept_to_card,
    export_memories_jsonl,
    format_search_results,
    is_program_card,
    normalize_allowed_gam_tools,
    normalize_gam_pipeline_mode,
    normalize_gam_top_k_by_tool,
    normalize_memory_card,
    note_metadata,
)

# Re-export for backward compatibility (extracted to concept_api.py)
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.utils import (
    looks_like_uuid,
    truncate_text,
)

# ---------------------------------------------------------------------------
# Protocols for agentic dependencies (A-MEM, GAM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Card memory type alias
# ---------------------------------------------------------------------------


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
        self.allowed_gam_tools = normalize_allowed_gam_tools(allowed_gam_tools)
        self.gam_top_k_by_tool = normalize_gam_top_k_by_tool(gam_top_k_by_tool)
        self.gam_pipeline_mode = normalize_gam_pipeline_mode(gam_pipeline_mode)
        self.card_update_dedup_config = CardUpdateDedupConfig.from_mapping(
            card_update_dedup_config or {}
        )
        self._warned_missing_card_update_llm = False
        self._iters_after_rebuild = 0

        self.api: _ConceptApiClient | None = None
        if self.use_api:
            self.api = _ConceptApiClient(base_url=base_url)
        else:
            logger.info("[Memory] API mode disabled. Running in local-only mode.")

        self._AgenticMemorySystemCls: type[Any] | None = None
        self._MemoryNoteCls: type[Any] | None = None
        self._ResearchAgentCls: type[Any] | None = None
        self._AMemGeneratorCls: type[Any] | None = None
        self._agentic_import_error: Exception | None = None
        self._load_agentic_classes()

        self.memory_cards: dict[str, AnyCard] = {}
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

        self.llm_service: LLMServiceProtocol | None
        self.generator: GeneratorProtocol | None
        self.llm_service, self.generator = self._init_llm_service_and_generator()
        self.memory_system: AgenticMemoryProtocol | None = self._init_storage()
        self.research_agent: ResearchAgentProtocol | None = None
        self._dedup_retrievers: dict[str, Any] | None = None

        if (
            self.memory_system is not None
            and self.generator is not None
            and self.export_file.exists()
        ):
            try:
                self.research_agent = self._load_or_create_retriever()
            except Exception as exc:
                logger.debug("[Memory] Initial retriever load skipped: {}", exc)

        if sync_on_init and self.use_api:
            self._sync_from_api(force_full=True)

    def _load_agentic_classes(self) -> None:
        try:
            from gigaevo.memory.A_mem.agentic_memory.memory_system import (
                AgenticMemorySystem as _AgenticMemorySystem,
            )
            from gigaevo.memory.A_mem.agentic_memory.memory_system import (
                MemoryNote as _MemoryNote,
            )
            from gigaevo.memory.GAM_root.gam import ResearchAgent as _ResearchAgent
            from gigaevo.memory.GAM_root.gam.generator import (
                AMemGenerator as _AMemGenerator,
            )
        except Exception as exc:
            self._agentic_import_error = exc
            logger.info(
                "[Memory] Agentic runtime dependencies are unavailable. "
                "Reason: {}. Falling back to API full-text mode.",
                exc,
            )
            return

        self._AgenticMemorySystemCls = _AgenticMemorySystem
        self._MemoryNoteCls = _MemoryNote
        self._ResearchAgentCls = _ResearchAgent
        self._AMemGeneratorCls = _AMemGenerator

    def _init_llm_service_and_generator(
        self,
    ) -> tuple[LLMServiceProtocol | None, GeneratorProtocol | None]:
        if self._AMemGeneratorCls is None and not self.card_update_dedup_config.enabled:
            return None, None
        api_key = config.OPENAI_API_KEY
        if not api_key and config.LLM_BASE_URL:
            # Local OpenAI-compatible servers (vLLM/LM Studio/Ollama OpenAI mode)
            # often accept any non-empty bearer token.
            api_key = "EMPTY"

        if not api_key:
            logger.info(
                "[Memory] OPENAI_API_KEY/OPENROUTER_API_KEY is not set. "
                "Agentic retrieval is disabled; API full-text fallback is available."
            )
            return None, None

        try:
            base_url = config.LLM_BASE_URL

            llm_service = OpenAIInferenceService(
                model_name=config.OPENROUTER_MODEL_NAME or DEFAULT_MODEL_NAME,
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
            logger.warning("[Memory] Could not initialize LLM/generator: {}", exc)
            return None, None

    def _init_storage(self) -> AgenticMemoryProtocol | None:
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
            logger.warning("[Memory] Could not initialize AgenticMemorySystem: {}", exc)
            return None

    def _load_index(self) -> None:
        if not self.index_file.exists():
            return

        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[Memory] Could not parse index file {}: {}", self.index_file, exc
            )
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
        serialized_cards = {cid: c.model_dump() for cid, c in self.memory_cards.items()}
        payload = {
            "entity_by_card_id": self.entity_by_card_id,
            "entity_version_by_entity": self.entity_version_by_entity,
            "memory_cards": serialized_cards,
        }
        tmp_file = self.index_file.with_suffix(f".{os.getpid()}.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_file), str(self.index_file))

    def _ensure_card_id(self, card: AnyCard) -> str:
        card_id = str(card.id or "").strip()
        if not card_id:
            card_id = f"mem-{uuid.uuid4().hex[:12]}"
            card.id = card_id
        return card_id

    def _build_note_from_card(self, card: AnyCard) -> MemoryNoteProtocol:
        if self._MemoryNoteCls is None:
            raise RuntimeError("MemoryNote class is unavailable")
        card_id = str(card.id or "")
        description = str(card.description or "")
        context = str(
            card.task_description or card.task_description_summary or "General"
        )
        category = str(card.category or "general")
        strategy = str(card.strategy or "")
        keywords = list(card.keywords or [])
        links = list(card.links or [])
        existing = (
            self.memory_system.read(card_id) if self.memory_system is not None else None
        )

        return self._MemoryNoteCls(
            content=description,
            id=card_id,
            keywords=keywords,
            links=links,
            retrieval_count=(existing.retrieval_count if existing is not None else 0),
            timestamp=(existing.timestamp if existing is not None else None),
            last_accessed=(existing.last_accessed if existing is not None else None),
            context=context or "General",
            evolution_history=(
                existing.evolution_history if existing is not None else None
            ),
            category=category,
            tags=(existing.tags if existing is not None else []),
            strategy=strategy,
        )

    @staticmethod
    def _note_fields_changed(
        existing: MemoryNoteProtocol,
        content: Any,
        category: Any,
        context: Any,
        strategy: Any,
        keywords: Any,
        links: Any,
    ) -> bool:
        return (
            existing.content != content
            or existing.category != category
            or existing.context != context
            or existing.strategy != strategy
            or existing.keywords != keywords
            or existing.links != links
        )

    def _upsert_local_note_fast(self, card: AnyCard) -> bool:
        """Synchronize card into local A-MEM/Chroma without running LLM evolution."""
        if self.memory_system is None:
            return False

        note = self._build_note_from_card(card)
        existing = self.memory_system.read(note.id)
        changed = existing is None or self._note_fields_changed(
            existing,
            note.content,
            note.category,
            note.context,
            note.strategy,
            note.keywords,
            note.links,
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
            note_metadata(note),
            note.id,
        )
        self.memory_ids.add(note.id)
        return True

    def _upsert_local_note_agentic(self, card: AnyCard) -> bool:
        """Add/update card in local A-MEM using regular add/update path for local writes."""
        if self.memory_system is None:
            return False

        card_id = str(card.id or "").strip()
        if not card_id:
            return False

        description = str(card.description or "")
        kwargs = {
            "category": str(card.category or "general"),
            "keywords": list(card.keywords or []),
            "context": str(
                card.task_description or card.task_description_summary or "General"
            ),
            "strategy": str(card.strategy or ""),
            "links": list(card.links or []),
            "tags": [],
        }

        existing = self.memory_system.read(card_id)
        if existing is None:
            self.memory_system.add_note(id=card_id, content=description, **kwargs)
        else:
            changed = self._note_fields_changed(
                existing,
                description,
                kwargs["category"],
                kwargs["context"],
                kwargs["strategy"],
                kwargs["keywords"],
                kwargs["links"],
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
            fallback_id = self.card_id_by_entity.get(entity_id) or str(
                content.get("id") or entity_id
            )
            card = concept_to_card(content, fallback_id=fallback_id)
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

        stale_entities = [
            eid for eid in self.card_id_by_entity if eid not in remote_entity_ids
        ]
        for entity_id in stale_entities:
            card_id = self.card_id_by_entity.pop(entity_id, "")
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
            if (
                self.research_agent is None
                and self.memory_system is not None
                and self.generator is not None
            ):
                self.rebuild()

        return changed

    def _load_or_create_retriever(self) -> ResearchAgentProtocol | None:
        if self.generator is None or self._ResearchAgentCls is None:
            raise RuntimeError(
                "Generator is not available. Cannot create GAM research agent."
            )
        try:
            from gigaevo.memory.shared_memory.amem_gam_retriever import (
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
        logger.info(
            "[Memory] Loaded {} cards, added {} new pages.", len(records), added
        )

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
            logger.info(
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
        export_memories_jsonl(
            self.memory_system,
            all_ids,
            self.export_file,
            card_overrides=self.memory_cards,
        )

    def _build_dedup_retrievers(self) -> dict[str, Any]:
        try:
            from gigaevo.memory.shared_memory.amem_gam_retriever import (
                build_gam_store,
                build_retrievers,
                load_amem_records,
            )
        except Exception as exc:
            logger.warning("[Memory] Dedup retriever import failed: {}", exc)
            return {}

        self.gam_store_dir.mkdir(parents=True, exist_ok=True)
        if self.export_file.exists():
            try:
                records = load_amem_records(self.export_file)
            except Exception:
                records = [c.model_dump() for c in self.memory_cards.values()]
        else:
            records = [c.model_dump() for c in self.memory_cards.values()]
        records = [
            r
            for r in records
            if str(r.get("category", "")).strip().lower() != "program"
        ]
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
            logger.warning("[Memory] Dedup retriever build failed: {}", exc)
            return {}

        return {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self.allowed_gam_tools
        }

    def _resolve_vector_retriever(self, tool_name: str) -> Any:
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
        card: AnyCard,
    ) -> list[dict[str, Any]]:
        cfg = self.card_update_dedup_config
        if not cfg.enabled or not self.memory_cards:
            return []

        query_by_key = build_dedup_queries(card.model_dump())
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
                logger.warning(
                    "[Memory] Dedup retrieval failed for query '{}': {}", query_key, exc
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
                if is_program_card(self.memory_cards[card_id]):
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
            if card is None:
                continue

            card_dict = card.model_dump()
            explanations = get_full_explanations(card_dict)
            payload.append(
                {
                    "card_id": card_id,
                    "final_score": float(item.get("final_score", 0.0)),
                    "scores": item.get("scores", {}),
                    "task_description_summary": truncate_text(
                        card.task_description_summary, 600
                    ),
                    "description": truncate_text(card.description, 1200),
                    "explanation_summary": truncate_text(
                        get_explanation_summary(card_dict), 600
                    ),
                    "explanation_full": [
                        truncate_text(explanation, 1200) for explanation in explanations
                    ],
                }
            )
        return payload

    def _decide_card_action(
        self,
        incoming_card: AnyCard,
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

        incoming_dict = incoming_card.model_dump()
        incoming_payload = {
            "id": str(incoming_card.id or "").strip(),
            "task_description_summary": truncate_text(
                incoming_card.task_description_summary, 600
            ),
            "description": truncate_text(incoming_card.description, 1200),
            "explanation_summary": truncate_text(
                get_explanation_summary(incoming_dict), 600
            ),
            "explanation_full": [
                truncate_text(explanation, 1200)
                for explanation in get_full_explanations(incoming_dict)
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
        for attempt in range(self.card_update_dedup_config.llm_max_retries):
            try:
                response_text, _, _, _ = self.llm_service.generate(prompt)
            except Exception as exc:
                logger.warning("[Memory] Dedup LLM decision call failed: {}", exc)
                continue
            parsed = parse_llm_card_decision(
                response_text,
                candidate_ids=candidate_ids,
            )
            if parsed is not None:
                decision = parsed
                break
            logger.warning(
                "[Memory] Dedup LLM returned no valid JSON (attempt {}/{})",
                attempt + 1,
                self.card_update_dedup_config.llm_max_retries,
            )
        return decision

    def _apply_update_actions(
        self,
        incoming_card: AnyCard,
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
            if existing_card is None:
                continue

            existing_dict = existing_card.model_dump()
            incoming_dict = incoming_card.model_dump()
            merged_dict = merge_updated_card(existing_dict, incoming_dict, update)
            merged_dict["id"] = card_id
            merged_card = normalize_memory_card(merged_dict)
            self._save_card_core(merged_card)
            seen_ids.add(card_id)
            updated_ids.append(card_id)
        return updated_ids

    def _save_card_core(self, card: AnyCard) -> str:
        card_id = self._ensure_card_id(card)

        if self.enable_llm_card_enrichment and self.memory_system is not None:
            analysis = self.memory_system.analyze_content(card.description)
            if not card.keywords:
                card.keywords = analysis.get("keywords") or []
            if not card.task_description:
                card.task_description = analysis.get("context") or ""

        content = card_to_concept_content(card)
        name, tags, when_to_use = build_entity_meta(card)

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
            self.entity_version_by_entity[saved_entity_id] = str(
                response.get("version_id") or ""
            )
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

    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        normalized_card = normalize_memory_card(card)
        self.card_write_stats["processed"] += 1
        incoming_card_id = str(normalized_card.id or "").strip()
        if incoming_card_id and incoming_card_id in self.memory_cards:
            self.card_write_stats["updated"] += 1
            return self._save_card_core(normalized_card)

        if is_program_card(normalized_card):
            self.card_write_stats["added"] += 1
            return self._save_card_core(normalized_card)

        if (
            self.card_update_dedup_config.enabled
            and self.memory_cards
            and self.llm_service is None
            and not self._warned_missing_card_update_llm
        ):
            logger.warning(
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

    def _synthesize_results(
        self,
        query: str,
        memory_state: str | None,
        cards: list[dict[str, Any]],
    ) -> str:
        if self.llm_service is None:
            return format_search_results(query, cards)

        cards_blob = []
        for card in cards:
            if isinstance(card, MemoryCard):
                expl_text = card.explanation.summary
            else:
                expl_text = ""
            cards_blob.append(
                "\n".join(
                    [
                        f"id: {card.id}",
                        f"category: {card.category}",
                        f"task_description_summary: {card.task_description_summary}",
                        f"task_description: {card.task_description}",
                        f"description: {card.description}",
                        f"keywords: {card.keywords}",
                        f"explanation: {expl_text}",
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
            "Retrieved cards:\n" + "\n\n".join(cards_blob) + "\n\nAnswer:"
        )

        try:
            text, _, _, _ = self.llm_service.generate(prompt)
            text = str(text or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning(
                "[Memory] LLM synthesis failed, fallback to plain output: {}", exc
            )

        return format_search_results(query, cards)

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

        cards: list[AnyCard] = []
        local_changed = False

        for hit in hits:
            entity_id = str(hit.get("entity_id") or "").strip()
            if not entity_id:
                continue

            concept = self.api.get_concept(entity_id, channel=self.channel)
            content = concept.get("content") or {}

            card_id = str(
                content.get("id") or self.card_id_by_entity.get(entity_id) or entity_id
            )
            card = concept_to_card(content, fallback_id=card_id)
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
        if (
            local_changed
            and self.memory_system is not None
            and self.generator is not None
        ):
            self.rebuild()

        if not cards:
            return f"Query: {query}\n\nNo relevant memories found."

        if self.enable_llm_synthesis:
            return self._synthesize_results(query, memory_state, cards)
        return format_search_results(query, cards)

    def _search_local_cards(self, query: str, memory_state: str | None = None) -> str:
        if not self.memory_cards:
            return f"Query: {query}\n\nNo relevant memories found."

        query_text = f"{query} {memory_state or ''}".strip().lower()
        tokens = [tok for tok in re.split(r"\W+", query_text) if tok]
        if not tokens:
            tokens = [query.strip().lower()] if query.strip() else []

        scored: list[tuple[int, dict[str, Any]]] = []
        for card in self.memory_cards.values():
            haystack_text = " ".join(
                [
                    str(card.description or ""),
                    str(card.task_description_summary or ""),
                    str(card.task_description or ""),
                    " ".join([str(x) for x in (card.keywords or [])]),
                    str(card.category or ""),
                ]
            ).lower()
            haystack_tokens = set(re.split(r"\W+", haystack_text))
            score = sum(1 for tok in tokens if tok and tok in haystack_tokens)
            if score > 0:
                scored.append((score, card))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_cards = [card for _, card in scored[: self.search_limit]]

        if not top_cards:
            return f"Query: {query}\n\nNo relevant memories found."

        if self.enable_llm_synthesis:
            return self._synthesize_results(query, memory_state, top_cards)
        return format_search_results(query, top_cards)

    def search(self, query: str, memory_state: str | None = None) -> str:
        if self.use_api and self.api is not None:
            self._sync_from_api(force_full=False)

        if self.research_agent is not None:
            try:
                return self.research_agent.research(
                    query, memory_state=memory_state
                ).integrated_memory
            except Exception as exc:
                logger.warning(
                    "[Memory] GAM search failed, falling back to non-agentic search: {}",
                    exc,
                )

        if self.use_api and self.api is not None:
            return self._search_via_api(query, memory_state=memory_state)
        return self._search_local_cards(query, memory_state=memory_state)

    def get_card(self, card_id: str) -> AnyCard | None:
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
            if not entity_id and looks_like_uuid(key):
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

    def __enter__(self) -> AmemGamMemory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._iters_after_rebuild > 0:
            try:
                self.rebuild()
            except Exception:
                pass
        self.close()
