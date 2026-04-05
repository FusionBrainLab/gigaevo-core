from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from loguru import logger

import gigaevo.memory.config as config
from gigaevo.memory.openai_inference import OpenAIInferenceService
from gigaevo.memory.shared_memory.agentic_runtime import (
    AgenticRuntime,
    load_agentic_runtime,
)
from gigaevo.memory.shared_memory.api_sync import ApiSync
from gigaevo.memory.shared_memory.card_dedup import CardDedup
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.gam_search import GamSearch
from gigaevo.memory.shared_memory.note_sync import NoteSync
from gigaevo.memory.shared_memory.protocols import (
    AgenticMemoryProtocol,
    GeneratorProtocol,
    LLMServiceProtocol,
    ResearchAgentProtocol,
)

load_dotenv()

from gigaevo.memory.shared_memory.card_conversion import (
    DEFAULT_MODEL_NAME,
    AnyCard,
    GigaEvoMemoryBase,
    build_entity_meta,
    card_to_concept_content,
    format_search_results,
    is_program_card,
    normalize_allowed_gam_tools,
    normalize_gam_pipeline_mode,
    normalize_gam_top_k_by_tool,
    normalize_memory_card,
    search_cards_by_keyword,
    synthesize_search_results,
)

# Re-export for backward compatibility (extracted to concept_api.py)
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.memory_config import MemoryConfig
from gigaevo.memory.shared_memory.utils import looks_like_uuid


class AmemGamMemory(GigaEvoMemoryBase):
    """API-backed memory where API is the source of truth and local GAM is retrieval runtime.

    Accepts either a ``MemoryConfig`` object (preferred) or legacy kwargs
    (for backward compatibility during migration).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
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
        *,
        config: MemoryConfig | None = None,
        runtime: AgenticRuntime | None = None,
    ):
        # --- Build MemoryConfig ---
        if config is not None:
            self.config = config
        elif checkpoint_path is not None:
            self.config = MemoryConfig.from_legacy_kwargs(
                checkpoint_path=checkpoint_path,
                base_url=base_url,
                use_api=use_api,
                namespace=namespace,
                author=author,
                channel=channel,
                search_limit=search_limit,
                enable_llm_synthesis=enable_llm_synthesis,
                enable_memory_evolution=enable_memory_evolution,
                enable_llm_card_enrichment=enable_llm_card_enrichment,
                rebuild_interval=rebuild_interval,
                enable_bm25=enable_bm25,
                sync_batch_size=sync_batch_size,
                sync_on_init=sync_on_init,
                allowed_gam_tools=allowed_gam_tools,
                gam_top_k_by_tool=gam_top_k_by_tool,
                gam_pipeline_mode=gam_pipeline_mode,
                card_update_dedup_config=card_update_dedup_config,
            )
        else:
            raise TypeError(
                "AmemGamMemory requires either config=MemoryConfig(...) "
                "or checkpoint_path='...'"
            )

        # --- Derived paths ---
        cfg = self.config
        self.checkpoint_dir = cfg.checkpoint_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.checkpoint_dir / "api_index.json"
        self.export_file = self.checkpoint_dir / "amem_exports" / "amem_memories.jsonl"
        self.gam_store_dir = self.checkpoint_dir / "gam_shared" / "amem_store"

        # --- Expose config fields as instance attrs (for existing code) ---
        api_cfg = cfg.api
        self.use_api = api_cfg is not None
        self.namespace = api_cfg.namespace if api_cfg else namespace
        self.author = api_cfg.author if api_cfg else author
        self.channel = api_cfg.channel if api_cfg else channel
        self.search_limit = cfg.search_limit
        self.rebuild_interval = cfg.rebuild_interval
        self.enable_bm25 = cfg.gam.enable_bm25
        self.sync_batch_size = api_cfg.sync_batch_size if api_cfg else 100
        self.enable_llm_synthesis = cfg.enable_llm_synthesis
        self.enable_memory_evolution = cfg.enable_memory_evolution
        self.enable_llm_card_enrichment = cfg.enable_llm_card_enrichment
        self.allowed_gam_tools = normalize_allowed_gam_tools(
            cfg.gam.allowed_tools or None
        )
        self.gam_top_k_by_tool = normalize_gam_top_k_by_tool(
            cfg.gam.top_k_by_tool or None
        )
        self.gam_pipeline_mode = normalize_gam_pipeline_mode(cfg.gam.pipeline_mode)
        self.card_update_dedup_config = cfg.dedup
        self._warned_missing_card_update_llm = False
        self._iters_after_rebuild = 0

        # --- API client ---
        self.api: _ConceptApiClient | None = None
        if self.use_api and api_cfg is not None:
            self.api = _ConceptApiClient(base_url=api_cfg.base_url)
        elif not self.use_api:
            logger.info("[Memory] API mode disabled. Running in local-only mode.")

        # --- Agentic runtime (DI or auto-detect) ---
        self._runtime: AgenticRuntime | None
        if runtime is not None:
            self._runtime = runtime
        else:
            self._runtime = load_agentic_runtime()
        self._AgenticMemorySystemCls: type[Any] | None = (
            self._runtime.memory_system_cls if self._runtime else None
        )
        self._MemoryNoteCls: type[Any] | None = (
            self._runtime.memory_note_cls if self._runtime else None
        )
        self._ResearchAgentCls: type[Any] | None = (
            self._runtime.research_agent_cls if self._runtime else None
        )
        self._AMemGeneratorCls: type[Any] | None = (
            self._runtime.generator_cls if self._runtime else None
        )

        self.card_store = CardStore(self.index_file)

        self.llm_service: LLMServiceProtocol | None
        self.generator: GeneratorProtocol | None
        self.llm_service, self.generator = self._init_llm_service_and_generator()
        self.memory_system: AgenticMemoryProtocol | None = self._init_storage()
        self.note_sync: NoteSync | None = None
        if self.memory_system is not None and self._MemoryNoteCls is not None:
            self.note_sync = NoteSync(
                self.memory_system, self._MemoryNoteCls, self.card_store
            )
        self.research_agent: ResearchAgentProtocol | None = None

        # --- Card dedup (always created; config.enabled gates scoring) ---
        self.dedup = CardDedup(
            card_store=self.card_store,
            llm_service=self.llm_service,
            config=self.card_update_dedup_config,
            allowed_gam_tools=self.allowed_gam_tools,
            gam_store_dir=self.gam_store_dir,
            export_file=self.export_file,
            checkpoint_dir=self.checkpoint_dir,
        )

        # --- GAM search ---
        self.gam: GamSearch | None = None
        if self._ResearchAgentCls is not None and self.generator is not None:
            self.gam = GamSearch(
                research_agent_cls=self._ResearchAgentCls,
                generator=self.generator,
                card_store=self.card_store,
                checkpoint_dir=self.checkpoint_dir,
                gam_store_dir=self.gam_store_dir,
                export_file=self.export_file,
                enable_bm25=self.enable_bm25,
                allowed_gam_tools=self.allowed_gam_tools,
                gam_top_k_by_tool=self.gam_top_k_by_tool,
                gam_pipeline_mode=self.gam_pipeline_mode,
            )

        # --- API sync (after note_sync so it can upsert notes) ---
        self.api_sync: ApiSync | None = None
        if self.api is not None:
            self.api_sync = ApiSync(
                client=self.api,
                card_store=self.card_store,
                note_sync=self.note_sync,
                namespace=self.namespace,
                channel=self.channel,
                sync_batch_size=self.sync_batch_size,
                search_limit=self.search_limit,
            )

        if (
            self.memory_system is not None
            and self.generator is not None
            and self.export_file.exists()
            and self.gam is not None
        ):
            try:
                self.gam.build()
                self.research_agent = self.gam.agent
            except Exception as exc:
                logger.debug("[Memory] Initial retriever load skipped: {}", exc)

        if self.use_api and api_cfg is not None and api_cfg.sync_on_init:
            self._sync_from_api(force_full=True)

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

    def _ensure_api_sync(self) -> ApiSync | None:
        """Lazily create ApiSync if mem.api was set post-construction."""
        if self.api_sync is not None:
            return self.api_sync
        if not self.use_api or self.api is None:
            return None
        self.api_sync = ApiSync(
            client=self.api,
            card_store=self.card_store,
            note_sync=self.note_sync,
            namespace=self.namespace,
            channel=self.channel,
            sync_batch_size=self.sync_batch_size,
            search_limit=self.search_limit,
        )
        return self.api_sync

    def _sync_from_api(self, force_full: bool = False) -> bool:
        sync = self._ensure_api_sync()
        if sync is None:
            return False
        changed = sync.sync(force_full=force_full)
        needs_retriever = (
            self.research_agent is None
            and self.memory_system is not None
            and self.generator is not None
        )
        if changed or needs_retriever:
            self.rebuild()
        else:
            self.card_store.persist()
        return changed

    def _apply_update_actions(
        self,
        incoming_card: AnyCard,
        updates: list[dict[str, Any]],
    ) -> list[str]:
        merges = self.dedup.compute_merges(incoming_card, updates)
        updated_ids: list[str] = []
        try:
            for card_id, merged_card in merges:
                self._save_card_core(merged_card)
                updated_ids.append(card_id)
        finally:
            if updated_ids:
                self.card_store.persist()
        return updated_ids

    def _save_card_core(self, card: AnyCard) -> tuple[str, bool]:
        """Save card to storage. Returns (card_id, rebuilt) where rebuilt
        indicates whether a periodic rebuild (which includes index persist)
        was triggered."""
        card_id = self.card_store.ensure_id(card)

        if self.enable_llm_card_enrichment and self.memory_system is not None:
            analysis = self.memory_system.analyze_content(card.description)
            if not card.keywords:
                card.keywords = analysis.get("keywords") or []
            if not card.task_description:
                card.task_description = analysis.get("context") or ""

        content = card_to_concept_content(card)
        name, tags, when_to_use = build_entity_meta(card)

        store = self.card_store
        if self.use_api and self.api is not None:
            response = self.api.save_concept(
                content=content,
                name=name,
                tags=tags,
                when_to_use=when_to_use,
                channel=self.channel,
                namespace=self.namespace,
                author=self.author,
                entity_id=store.entity_by_card_id.get(card_id),
            )
            store.save_entity(
                card_id,
                str(response["entity_id"]),
                str(response.get("version_id") or ""),
            )
        else:
            store.clear_entity(card_id)
        store.cards[card_id] = normalize_memory_card(card, fallback_id=card_id)

        if self.note_sync is not None:
            self.note_sync.upsert_agentic(store.cards[card_id])
        self.dedup.invalidate_retrievers()

        rebuilt = False
        self._iters_after_rebuild += 1
        if self._iters_after_rebuild >= self.rebuild_interval:
            self.rebuild()
            rebuilt = True

        return card_id, rebuilt

    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        normalized_card = normalize_memory_card(card)
        store = self.card_store
        store.write_stats["processed"] += 1
        incoming_card_id = str(normalized_card.id or "").strip()
        if incoming_card_id and incoming_card_id in store.cards:
            store.write_stats["updated"] += 1
            result, rebuilt = self._save_card_core(normalized_card)
            if not rebuilt:
                store.persist()
            return result

        if is_program_card(normalized_card):
            store.write_stats["added"] += 1
            result, rebuilt = self._save_card_core(normalized_card)
            if not rebuilt:
                store.persist()
            return result

        if (
            self.card_update_dedup_config.enabled
            and store.cards
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
            and store.cards
            and self.llm_service is not None
        ):
            # Sync llm_service in case it was set post-construction (tests)
            self.dedup.llm_service = self.llm_service
            scored_candidates = self.dedup.score_candidates(normalized_card)
            candidates_for_llm = self.dedup.format_for_llm(scored_candidates)
            decision = self.dedup.decide_action(normalized_card, candidates_for_llm)
            action = str(decision.get("action") or "add").strip().lower()

            if action == "discard":
                duplicate_id = str(decision.get("duplicate_of") or "").strip()
                if duplicate_id in store.cards:
                    store.write_stats["rejected"] += 1
                    return duplicate_id
            elif action == "update":
                updates = decision.get("updates")
                updated_ids = self._apply_update_actions(
                    normalized_card,
                    updates if isinstance(updates, list) else [],
                )
                if updated_ids:
                    store.write_stats["updated"] += 1
                    store.write_stats["updated_target_cards"] += len(updated_ids)
                    return updated_ids[0]

        store.write_stats["added"] += 1
        result, rebuilt = self._save_card_core(normalized_card)
        if not rebuilt:
            store.persist()
        return result

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
        cards: list[AnyCard],
    ) -> str:
        """Wrapper around pure synthesize_search_results function."""
        try:
            return synthesize_search_results(
                query=query,
                memory_state=memory_state,
                cards=cards,
                llm_service=self.llm_service,
            )
        except Exception as exc:
            logger.warning(
                "[Memory] LLM synthesis failed, fallback to plain output: {}", exc
            )
            return format_search_results(query, cards)

    def _search_via_api(self, query: str, memory_state: str | None = None) -> str:
        sync = self._ensure_api_sync()
        if sync is None:
            return self._search_local_cards(query, memory_state)

        cards, local_changed = sync.search(query, memory_state)

        will_rebuild = (
            local_changed
            and self.memory_system is not None
            and self.generator is not None
        )
        if not will_rebuild:
            self.card_store.persist()
        if will_rebuild:
            self.rebuild()

        if not cards:
            return f"Query: {query}\n\nNo relevant memories found."

        if self.enable_llm_synthesis:
            return self._synthesize_results(query, memory_state, cards)
        return format_search_results(query, cards)

    def _search_local_cards(self, query: str, memory_state: str | None = None) -> str:
        """Search local cards by keyword matching."""
        cards = self.card_store.cards
        if not cards:
            return f"Query: {query}\n\nNo relevant memories found."

        top_cards = search_cards_by_keyword(
            cards_dict=cards,
            query=query,
            memory_state=memory_state,
            search_limit=self.search_limit,
        )

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
        return self.card_store.cards.get(card_id)

    def get_card_write_stats(self) -> dict[str, int]:
        return dict(self.card_store.write_stats)

    def rebuild(self) -> None:
        serialized = self.card_store.serialize_all()
        self.card_store.persist(serialized=serialized)
        if self.memory_system is None or self.generator is None:
            return
        if self.note_sync is not None:
            self.note_sync.export_jsonl(self.export_file, serialized)
        if self.gam is not None:
            self.gam.build()
            self.research_agent = self.gam.agent
        self.dedup.invalidate_retrievers()
        self._iters_after_rebuild = 0

    def delete(self, memory_id: str) -> bool:
        key = str(memory_id).strip()
        store = self.card_store
        if self.use_api and self.api is not None:
            entity_id = store.entity_by_card_id.get(key)
            if not entity_id and looks_like_uuid(key):
                entity_id = key
            if not entity_id:
                return False
            self.api.delete_concept(entity_id)
            card_id = store.card_id_by_entity.pop(entity_id, key)
            store.entity_version.pop(entity_id, None)
        else:
            card_id = store.resolve_card_id(key)
            if card_id is None:
                return False
            store.clear_entity(card_id)

        store.entity_by_card_id.pop(card_id, None)
        store.cards.pop(card_id, None)
        if self.note_sync is not None:
            self.note_sync.remove(card_id)
        else:
            store.note_ids.discard(card_id)
        store.persist()

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
