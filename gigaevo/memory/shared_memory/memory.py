from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

from loguru import logger

from gigaevo.exceptions import MemoryRetrieverError
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
from gigaevo.memory.shared_memory.memory_state import MemoryState
from gigaevo.memory.shared_memory.note_sync import NoteSync
from gigaevo.memory.shared_memory.protocols import ResearchOutput

if TYPE_CHECKING:
    from gigaevo.memory.shared_memory.protocols import (
        AgenticMemoryProtocol,
        GeneratorProtocol,
        LLMServiceProtocol,
        ResearchAgentProtocol,
    )


class AmemGamMemory(GigaEvoMemoryBase):
    """Orchestrator for card storage, search, sync, and dedup.

    Requires a ``MemoryConfig`` object for construction.
    """

    @property
    def _has_agentic(self) -> bool:
        return self.memory_system is not None and self.generator is not None

    @property
    def is_ready(self) -> bool:
        """True if memory is fully initialized and ready for operations."""
        return self._state.is_ready

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
        self._state = MemoryState()
        self._last_seen_index_mtime: float = 0.0

        # --- API client ---
        api_cfg = cfg.api
        self.api: _ConceptApiClient | None = None
        if api_cfg is not None:
            self.api = _ConceptApiClient(base_url=api_cfg.base_url)
        else:
            logger.info(
                "[Memory][Store] API mode disabled. Running in local-only mode."
            )

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
                self.gam.build_research_agent()
                self.research_agent = self.gam.agent
            except MemoryRetrieverError as exc:
                logger.debug("[Memory][Store] Initial retriever load skipped: {}", exc)

        if cfg.index_file.exists():
            self._last_seen_index_mtime = cfg.index_file.stat().st_mtime

        if api_cfg is not None and api_cfg.sync_on_init:
            self._sync_from_api(force_full=True)

        self._state.mark_ready()

    def _get_api_sync(self) -> ApiSync | None:
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
        sync = self._get_api_sync()
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
            self._persist_index()
        return changed

    def _apply_dedup_merge_updates(
        self, merges: list[tuple[str, AnyCard]]
    ) -> list[str]:
        """Apply pre-computed merges from dedup decision."""
        updated_ids: list[str] = []
        for card_id, merged_card in merges:
            try:
                self._insert_new_card(merged_card)
                updated_ids.append(card_id)
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] Merge into card {!r} failed: {}", card_id, exc
                )
        if updated_ids:
            self._persist_index()
        return updated_ids

    def _insert_new_card(self, card: AnyCard) -> tuple[str, bool]:
        """Save card to storage. Returns (card_id, rebuilt) where rebuilt
        indicates whether a periodic rebuild (which includes index persist)
        was triggered."""
        card_id = self.card_store.ensure_id(card)

        if self.config.enable_llm_card_enrichment and self.memory_system is not None:
            analysis = self.memory_system.analyze_content(card.description)
            enrichments: dict[str, Any] = {}
            if not card.keywords:
                enrichments["keywords"] = analysis.get("keywords") or []
            if not card.task_description:
                enrichments["task_description"] = analysis.get("context") or ""
            if enrichments:
                card = card.model_copy(update=enrichments)

        store = self.card_store
        sync = self._get_api_sync()
        if sync is not None:
            sync.save_card_to_api(card, card_id)
        else:
            store.clear_entity(card_id)
        store.cards[card_id] = normalize_memory_card(card, fallback_id=card_id)

        if self.note_sync is not None:
            self.note_sync.sync_card_to_amem_with_evolution(store.cards[card_id])
        self.dedup.invalidate_retrievers()

        rebuilt = False
        self._iters_after_rebuild += 1
        if self._iters_after_rebuild >= self.config.rebuild_interval:
            self.rebuild()
            rebuilt = True

        return card_id, rebuilt

    def _save_new_card_and_flush(self, card: AnyCard) -> str:
        """Save card and persist index unless a periodic rebuild already did."""
        card_id, rebuilt = self._insert_new_card(card)
        if not rebuilt:
            self._persist_index()
        return card_id

    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        """Save a memory card, with optional dedup against existing cards.

        Args:
            card: Raw dict or Pydantic card to save. Normalized internally.

        Returns:
            Card ID of the saved (or deduplicated) card.
        """
        normalized_card = normalize_memory_card(card)
        store = self.card_store
        store.write_stats["processed"] += 1
        incoming_card_id = str(normalized_card.id or "").strip()

        if incoming_card_id and incoming_card_id in store.cards:
            store.write_stats["updated"] += 1
            return self._save_new_card_and_flush(normalized_card)

        if is_program_card(normalized_card):
            store.write_stats["added"] += 1
            return self._save_new_card_and_flush(normalized_card)

        dedup_ready = self.config.dedup.enabled and store.cards
        if dedup_ready and self.llm_service is None:
            if not self._warned_missing_card_update_llm:
                logger.warning(
                    "[Memory][Store] card_update_dedup enabled but LLM unavailable; "
                    "falling back to regular save_card."
                )
                self._warned_missing_card_update_llm = True

        if dedup_ready and self.llm_service is not None:
            self.dedup.llm_service = self.llm_service
            decision = self.dedup.run_dedup_on_incoming_card(normalized_card)

            if decision.action == "discard":
                store.write_stats["rejected"] += 1
                if decision.duplicate_of and decision.duplicate_of in store.cards:
                    return decision.duplicate_of
                return store.ensure_id(normalized_card)

            if decision.action == "update" and decision.merges:
                updated_ids = self._apply_dedup_merge_updates(decision.merges)
                if updated_ids:
                    store.write_stats["updated"] += 1
                    store.write_stats["updated_target_cards"] += len(updated_ids)
                    return updated_ids[0]

        store.write_stats["added"] += 1
        return self._save_new_card_and_flush(normalized_card)

    def save(self, data: str, category: str = "general") -> str:
        """Save a text description as a new memory card."""
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
        sync = self._get_api_sync()
        if sync is None:
            return self._search_local_cards(query, memory_state)

        cards, local_changed = sync.search(query, memory_state)

        if local_changed and self._has_agentic:
            self.rebuild()
        else:
            self._persist_index()

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

    def _persist_index(
        self, serialized: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Persist card_store and advance the self-seen mtime watermark.

        Without the bump, our own writes would later trip the staleness check
        and trigger a self-reload that discards in-memory state that hasn't
        been flushed yet (e.g. test mutations on `card_store.cards`).
        """
        self.card_store.persist(serialized=serialized)
        try:
            self._last_seen_index_mtime = self.config.index_file.stat().st_mtime
        except OSError as exc:
            logger.debug("[Memory][Store] post-persist mtime read failed: {}", exc)

    def _refresh_from_disk_if_stale(self) -> None:
        """Reload card_store + rebuild GAM agent if the on-disk index changed.

        Fixes the reader-vs-writer split-brain: when a reader instance is
        created before any cards exist (or before a writer's later additions),
        it must pick up subsequent on-disk writes performed by the separate
        writer instance. Triggered lazily on every search() call.
        """
        cfg = self.config
        if not cfg.index_file.exists():
            return
        try:
            mtime = cfg.index_file.stat().st_mtime
        except OSError as exc:
            logger.debug("[Memory][Store] mtime check failed: {}", exc)
            return
        if mtime <= self._last_seen_index_mtime:
            return

        try:
            self.card_store.reload()
        except Exception as exc:
            logger.warning("[Memory][Store] card_store reload failed: {}", exc)
            return

        if self._has_agentic and self.gam is not None and cfg.export_file.exists():
            try:
                self.gam.build_research_agent()
                self.research_agent = self.gam.agent
                self._gam_build_failed = False
            except MemoryRetrieverError as exc:
                logger.warning(
                    "[Memory][Store] Stale-refresh GAM rebuild failed: {}", exc
                )
                self.research_agent = None
                self._gam_build_failed = True

        self._last_seen_index_mtime = mtime

    def research(self, query: str, memory_state: str | None = None) -> ResearchOutput:
        """Self-healing structured search.

        Refreshes from disk if a separate writer instance has advanced the
        on-disk index, then dispatches to the GAM research agent. Falls back
        to a local-cards ResearchOutput when GAM is unavailable.
        """
        self._refresh_from_disk_if_stale()
        if self.research_agent is not None:
            try:
                return self.research_agent.research(query, memory_state=memory_state)
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] GAM research failed, falling back to local cards: {}",
                    exc,
                )
        text = self._search_local_cards(query, memory_state=memory_state)
        return ResearchOutput(integrated_memory=text, raw_memory=None)

    def search(self, query: str, memory_state: str | None = None) -> str:
        """Search memory cards. Tries GAM agent, then API, then local keyword match."""
        self._refresh_from_disk_if_stale()
        if self.api is not None:
            self._sync_from_api(force_full=False)

        if self.research_agent is not None:
            try:
                return self.research_agent.research(
                    query, memory_state=memory_state
                ).integrated_memory
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] GAM search failed, falling back to non-agentic search: {}",
                    exc,
                )

        if self.api is not None:
            return self._search_via_api(query, memory_state=memory_state)
        return self._search_local_cards(query, memory_state=memory_state)

    def get_card(self, card_id: str) -> AnyCard | None:
        """Return a card by ID, or None if not found."""
        return self.card_store.cards.get(card_id)

    def get_card_write_stats(self) -> dict[str, int]:
        return dict(self.card_store.write_stats)

    def rebuild(self) -> None:
        """Persist cards, re-export JSONL, rebuild GAM index and dedup retrievers."""
        serialized = self.card_store.serialize_all()
        self._persist_index(serialized=serialized)
        if not self._has_agentic:
            return
        if self.note_sync is not None:
            self.note_sync.export_jsonl(self.config.export_file, serialized)
        if self.gam is not None:
            # Track state only when already ready (not during initialization)
            track_state = self._state.current == "ready"
            if track_state:
                self._state.mark_building()
            try:
                self.gam.build_research_agent()
                self.research_agent = self.gam.agent
                self._gam_build_failed = False
                if track_state:
                    self._state.mark_ready()
            except MemoryRetrieverError as exc:
                logger.warning("[Memory][Store] GAM build failed: {}", exc)
                self.gam.clear_research_agent()
                self.research_agent = None
                self._gam_build_failed = True
                if track_state:
                    self._state.mark_error(f"GAM build failed: {exc}")
        self.dedup.invalidate_retrievers()
        self._iters_after_rebuild = 0

    def delete(self, memory_id: str) -> bool:
        """Delete a card by ID or entity ID. Returns True if found and removed."""
        key = str(memory_id).strip()
        store = self.card_store
        sync = self._get_api_sync()
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
            self._persist_index()

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
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._iters_after_rebuild > 0:
            try:
                self.rebuild()
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] Final rebuild during context exit failed; "
                    "some changes may not be persisted: {}",
                    exc,
                )
        self.close()
