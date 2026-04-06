from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from loguru import logger

import gigaevo.memory.config as _env_config  # noqa: F401
from gigaevo.memory.shared_memory.agentic_runtime import (
    AgenticRuntime,
    init_agentic_storage,
    init_llm_and_generator,
    load_agentic_runtime,
)
from gigaevo.memory.shared_memory.api_sync import ApiSync
from gigaevo.memory.shared_memory.base import GigaEvoMemoryBase
from gigaevo.memory.shared_memory.card_conversion import (
    AnyCard,
    is_program_card,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.card_dedup import CardDedup
from gigaevo.memory.shared_memory.card_search import (
    format_search_results,
    search_cards_by_keyword,
    synthesize_search_results,
)
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient
from gigaevo.memory.shared_memory.gam_search import GamSearch
from gigaevo.memory.shared_memory.memory_config import MemoryConfig
from gigaevo.memory.shared_memory.note_sync import NoteSync

if TYPE_CHECKING:
    from gigaevo.memory.shared_memory.protocols import (
        AgenticMemoryProtocol,
        GeneratorProtocol,
        LLMServiceProtocol,
        ResearchAgentProtocol,
    )

load_dotenv()


class AmemGamMemory(GigaEvoMemoryBase):
    """Orchestrator for card storage, search, sync, and dedup.

    Requires a ``MemoryConfig`` object for construction.
    """

    @property
    def _has_agentic(self) -> bool:
        return self.memory_system is not None and self.generator is not None

    def __init__(
        self,
        *,
        config: MemoryConfig,
        runtime: AgenticRuntime | None = None,
        llm_service: LLMServiceProtocol | None = None,
        generator: GeneratorProtocol | None = None,
    ) -> None:
        self.config = config

        cfg = self.config
        cfg.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._warned_missing_card_update_llm = False
        self._iters_after_rebuild = 0
        self._gam_build_failed = False

        # --- API client ---
        api_cfg = cfg.api
        self.api: _ConceptApiClient | None = None
        if api_cfg is not None:
            self.api = _ConceptApiClient(base_url=api_cfg.base_url)
        else:
            logger.info("[Memory] API mode disabled. Running in local-only mode.")

        # --- Agentic runtime (DI or auto-detect) ---
        rt = runtime if runtime is not None else load_agentic_runtime()
        _system_cls = rt.memory_system_cls if rt else None
        _note_cls = rt.memory_note_cls if rt else None
        _agent_cls = rt.research_agent_cls if rt else None
        _gen_cls = rt.generator_cls if rt else None

        self.card_store = CardStore(index_file=cfg.index_file)

        # --- LLM + generator (DI or environment-based) ---
        self.llm_service: LLMServiceProtocol | None
        self.generator: GeneratorProtocol | None
        if llm_service is not None or generator is not None:
            self.llm_service = llm_service
            self.generator = generator
        else:
            self.llm_service, self.generator = init_llm_and_generator(
                generator_cls=_gen_cls,
                dedup_enabled=cfg.dedup.enabled,
            )
        self.memory_system: AgenticMemoryProtocol | None = init_agentic_storage(
            llm_service=self.llm_service,
            system_cls=_system_cls,
            checkpoint_dir=cfg.checkpoint_path,
            enable_evolution=cfg.enable_memory_evolution,
        )
        self.note_sync: NoteSync | None = None
        if self.memory_system is not None and _note_cls is not None:
            self.note_sync = NoteSync(
                memory_system=self.memory_system,
                note_cls=_note_cls,
                card_store=self.card_store,
            )
        self.research_agent: ResearchAgentProtocol | None = None

        # --- Card dedup (always created; config.enabled gates scoring) ---
        self.dedup = CardDedup(
            card_store=self.card_store,
            llm_service=self.llm_service,
            config=cfg.dedup,
            allowed_gam_tools=cfg.gam.normalized_allowed_tools,
            gam_store_dir=cfg.gam_store_dir,
            export_file=cfg.export_file,
            checkpoint_dir=cfg.checkpoint_path,
        )

        # --- GAM search ---
        self.gam: GamSearch | None = None
        if _agent_cls is not None and self.generator is not None:
            self.gam = GamSearch(
                research_agent_cls=_agent_cls,
                generator=self.generator,
                card_store=self.card_store,
                checkpoint_dir=cfg.checkpoint_path,
                gam_store_dir=cfg.gam_store_dir,
                export_file=cfg.export_file,
                enable_bm25=cfg.gam.enable_bm25,
                allowed_gam_tools=cfg.gam.normalized_allowed_tools,
                gam_top_k_by_tool=cfg.gam.normalized_top_k_by_tool,
                gam_pipeline_mode=cfg.gam.normalized_pipeline_mode,
            )

        # --- API sync (after note_sync so it can upsert notes) ---
        self.api_sync: ApiSync | None = None
        if self.api is not None and api_cfg is not None:
            self.api_sync = ApiSync(
                client=self.api,
                card_store=self.card_store,
                note_sync=self.note_sync,
                namespace=api_cfg.namespace,
                channel=api_cfg.channel,
                author=api_cfg.author,
                sync_batch_size=api_cfg.sync_batch_size,
                search_limit=cfg.search_limit,
            )

        if self._has_agentic and cfg.export_file.exists() and self.gam is not None:
            try:
                self.gam.build()
                self.research_agent = self.gam.agent
            except Exception as exc:
                logger.debug("[Memory] Initial retriever load skipped: {}", exc)

        if api_cfg is not None and api_cfg.sync_on_init:
            self._sync_from_api(force_full=True)

    def _ensure_api_sync(self) -> ApiSync | None:
        """Lazily create ApiSync if mem.api was set post-construction."""
        if self.api_sync is not None:
            return self.api_sync
        if self.api is None:
            return None
        api_cfg = self.config.api
        self.api_sync = ApiSync(
            client=self.api,
            card_store=self.card_store,
            note_sync=self.note_sync,
            namespace=api_cfg.namespace if api_cfg else "default",
            channel=api_cfg.channel if api_cfg else "latest",
            author=api_cfg.author if api_cfg else None,
            sync_batch_size=api_cfg.sync_batch_size if api_cfg else 100,
            search_limit=self.config.search_limit,
        )
        return self.api_sync

    def _sync_from_api(self, force_full: bool = False) -> bool:
        sync = self._ensure_api_sync()
        if sync is None:
            return False
        changed = sync.sync(force_full=force_full)
        if changed:
            self._gam_build_failed = False  # new data — retry build
        needs_rebuild = changed or (
            self.research_agent is None
            and self._has_agentic
            and not self._gam_build_failed
        )
        if needs_rebuild:
            self.rebuild()
        else:
            self.card_store.persist()
        return changed

    def _apply_update_actions_from_merges(
        self, merges: list[tuple[str, AnyCard]]
    ) -> list[str]:
        """Apply pre-computed merges from dedup decision."""
        updated_ids: list[str] = []
        for card_id, merged_card in merges:
            try:
                self._save_card_core(merged_card)
                updated_ids.append(card_id)
            except Exception as exc:
                logger.warning("[Memory] Merge into card {!r} failed: {}", card_id, exc)
        if updated_ids:
            self.card_store.persist()
        return updated_ids

    def _save_card_core(self, card: AnyCard) -> tuple[str, bool]:
        """Save card to storage. Returns (card_id, rebuilt) where rebuilt
        indicates whether a periodic rebuild (which includes index persist)
        was triggered."""
        card_id = self.card_store.ensure_id(card)

        if self.config.enable_llm_card_enrichment and self.memory_system is not None:
            analysis = self.memory_system.analyze_content(card.description)
            if not card.keywords:
                card.keywords = analysis.get("keywords") or []
            if not card.task_description:
                card.task_description = analysis.get("context") or ""

        store = self.card_store
        sync = self._ensure_api_sync()
        if sync is not None:
            sync.save_card_to_api(card, card_id)
        else:
            store.clear_entity(card_id)
        store.cards[card_id] = normalize_memory_card(card, fallback_id=card_id)

        if self.note_sync is not None:
            self.note_sync.upsert_agentic(store.cards[card_id])
        self.dedup.invalidate_retrievers()

        rebuilt = False
        self._iters_after_rebuild += 1
        if self._iters_after_rebuild >= self.config.rebuild_interval:
            self.rebuild()
            rebuilt = True

        return card_id, rebuilt

    def _save_and_persist(self, card: AnyCard) -> str:
        """Save card and persist index unless a periodic rebuild already did."""
        card_id, rebuilt = self._save_card_core(card)
        if not rebuilt:
            self.card_store.persist()
        return card_id

    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        normalized_card = normalize_memory_card(card)
        store = self.card_store
        store.write_stats["processed"] += 1
        incoming_card_id = str(normalized_card.id or "").strip()

        if incoming_card_id and incoming_card_id in store.cards:
            store.write_stats["updated"] += 1
            return self._save_and_persist(normalized_card)

        if is_program_card(normalized_card):
            store.write_stats["added"] += 1
            return self._save_and_persist(normalized_card)

        dedup_ready = self.config.dedup.enabled and store.cards
        if dedup_ready and self.llm_service is None:
            if not self._warned_missing_card_update_llm:
                logger.warning(
                    "[Memory] card_update_dedup enabled but LLM unavailable; "
                    "falling back to regular save_card."
                )
                self._warned_missing_card_update_llm = True

        if dedup_ready and self.llm_service is not None:
            self.dedup.llm_service = self.llm_service
            decision = self.dedup.process_incoming(normalized_card)

            if decision.action == "discard":
                store.write_stats["rejected"] += 1
                if decision.duplicate_of and decision.duplicate_of in store.cards:
                    return decision.duplicate_of
                return store.ensure_id(normalized_card)

            if decision.action == "update" and decision.merges:
                updated_ids = self._apply_update_actions_from_merges(decision.merges)
                if updated_ids:
                    store.write_stats["updated"] += 1
                    store.write_stats["updated_target_cards"] += len(updated_ids)
                    return updated_ids[0]

        store.write_stats["added"] += 1
        return self._save_and_persist(normalized_card)

    def save(self, data: str, category: str = "general") -> str:
        return self.save_card({"category": category, "description": data})

    def _format_search_output(
        self,
        query: str,
        cards: list[AnyCard],
        memory_state: str | None = None,
    ) -> str:
        if not cards:
            return f"Query: {query}\n\nNo relevant memories found."
        if self.config.enable_llm_synthesis:
            return synthesize_search_results(
                query=query,
                memory_state=memory_state,
                cards=cards,
                llm_service=self.llm_service,
            )
        return format_search_results(query, cards)

    def _search_via_api(self, query: str, memory_state: str | None = None) -> str:
        sync = self._ensure_api_sync()
        if sync is None:
            return self._search_local_cards(query, memory_state)

        cards, local_changed = sync.search(query, memory_state)

        if local_changed and self._has_agentic:
            self.rebuild()
        else:
            self.card_store.persist()

        return self._format_search_output(query, cards, memory_state)

    def _search_local_cards(self, query: str, memory_state: str | None = None) -> str:
        """Search local cards by keyword matching."""
        top_cards = search_cards_by_keyword(
            cards_dict=self.card_store.cards,
            query=query,
            memory_state=memory_state,
            search_limit=self.config.search_limit,
        )
        return self._format_search_output(query, top_cards, memory_state)

    def search(self, query: str, memory_state: str | None = None) -> str:
        if self.api is not None:
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

        if self.api is not None:
            return self._search_via_api(query, memory_state=memory_state)
        return self._search_local_cards(query, memory_state=memory_state)

    def get_card(self, card_id: str) -> AnyCard | None:
        return self.card_store.cards.get(card_id)

    def get_card_write_stats(self) -> dict[str, int]:
        return dict(self.card_store.write_stats)

    def rebuild(self) -> None:
        serialized = self.card_store.serialize_all()
        self.card_store.persist(serialized=serialized)
        if not self._has_agentic:
            return
        if self.note_sync is not None:
            self.note_sync.export_jsonl(self.config.export_file, serialized)
        if self.gam is not None:
            try:
                self.gam.build()
                self.research_agent = self.gam.agent
                self._gam_build_failed = False
            except Exception as exc:
                logger.warning("[Memory] GAM build failed: {}", exc)
                self._gam_build_failed = True
        self.dedup.invalidate_retrievers()
        self._iters_after_rebuild = 0

    def delete(self, memory_id: str) -> bool:
        key = str(memory_id).strip()
        store = self.card_store
        sync = self._ensure_api_sync()
        if sync is not None:
            card_id = sync.delete_from_api(key)
            if card_id is None:
                return False
        else:
            resolved = store.resolve_card_id(key)
            if resolved is None:
                return False
            card_id = resolved
            store.clear_entity(card_id)

        store.cards.pop(card_id, None)
        if self.note_sync is not None:
            self.note_sync.remove(card_id)
        else:
            store.note_ids.discard(card_id)

        if self._has_agentic:
            self.rebuild()
        else:
            store.persist()

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
