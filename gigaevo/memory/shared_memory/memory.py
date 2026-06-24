from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

from loguru import logger

from gigaevo.exceptions import MemoryRetrieverError
from gigaevo.memory.core.deduplicator import LLMDeduplicator
from gigaevo.memory.core.events import (
    emit_memory_event,
    memory_event_context,
    resolve_memory_event_path,
)
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.protocols import Deduplicator, Evictor
from gigaevo.memory.core.write_ledger import WriteLedger
from gigaevo.memory.core.write_pipeline import MemoryWritePipeline
from gigaevo.memory.shared_memory.agentic_runtime import (
    AgenticRuntime,
    init_agentic_storage,
    load_agentic_runtime,
)
from gigaevo.memory.shared_memory.api_sync import ApiSync
from gigaevo.memory.shared_memory.base import GigaEvoMemoryBase
from gigaevo.memory.shared_memory.card_conversion import (
    AnyCard,
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
        evictor: Evictor | None = None,
        deduplicator: Deduplicator | None = None,
    ) -> None:
        self.config = config

        cfg = self.config
        cfg.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._event_path = resolve_memory_event_path(cfg.checkpoint_path)

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

        # --- LLM + generator (injected; no LLM means agentic features stay off) ---
        self.llm_service: LLMServiceProtocol | None = llm_service
        if generator is None and llm_service is not None and _gen_cls is not None:
            generator = _gen_cls({"llm_service": llm_service})
        self.generator: GeneratorProtocol | None = generator
        if self.llm_service is None:
            logger.info(
                "[Memory][Store] No LLM injected — synthesis/dedup-scoring/GAM off."
            )
        self.memory_system: AgenticMemoryProtocol | None = init_agentic_storage(
            llm_service=self.llm_service,
            system_cls=_system_cls,
            checkpoint_dir=cfg.checkpoint_path,
            enable_evolution=cfg.enable_memory_evolution,
            embedding_model_name=cfg.embedding_model_name,
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
        # An injected LLMDeduplicator carries its own config, which must drive
        # the engine — reconcile() consults engine.config, not wrapper config.
        engine_dedup_cfg = (
            deduplicator.config
            if isinstance(deduplicator, LLMDeduplicator)
            else cfg.dedup
        )
        self.dedup = CardDedup(
            card_store=self.card_store,
            llm_service=self.llm_service,
            config=engine_dedup_cfg,
            allowed_gam_tools=cfg.gam.normalized_allowed_tools,
            gam_store_dir=cfg.gam_store_dir,
            export_file=cfg.export_file,
            checkpoint_dir=cfg.checkpoint_path,
            embedding_model_name=cfg.embedding_model_name,
        )
        if deduplicator is None:
            deduplicator = LLMDeduplicator(config=cfg.dedup, engine=self.dedup)
        elif isinstance(deduplicator, LLMDeduplicator):
            # Always rebind: the write path builds a fresh backend per sweep,
            # so a shared singleton must not stay bound to a stale engine.
            deduplicator.engine = self.dedup
        self.write_pipeline = MemoryWritePipeline(
            store=self,
            evictor=evictor if evictor is not None else HarmEvictor(),
            deduplicator=deduplicator,
            ledger=WriteLedger(cfg.checkpoint_path / "write_ledger.jsonl"),
            event_path=self._event_path,
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
                allowed_gam_tools=cfg.gam.normalized_allowed_tools,
                gam_top_k_by_tool=cfg.gam.normalized_top_k_by_tool,
                embedding_model_name=cfg.embedding_model_name,
                max_iters=cfg.gam.max_iters,
                max_cards=cfg.gam.max_cards,
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
        self._emit_store_event(
            "store.init",
            {
                "checkpoint_path": cfg.checkpoint_path,
                "index_file": cfg.index_file,
                "export_file": cfg.export_file,
                "card_count": len(self.card_store.cards),
                "index_exists": cfg.index_file.exists(),
                "export_exists": cfg.export_file.exists(),
                "api_enabled": self.api is not None,
                "api_sync_enabled": self.api_sync is not None,
                "llm_enabled": self.llm_service is not None,
                "agentic_enabled": self._has_agentic,
                "note_sync_enabled": self.note_sync is not None,
                "gam_enabled": self.gam is not None,
                "research_agent_ready": self.research_agent is not None,
                "embedding_model_name": cfg.embedding_model_name,
                "search_limit": cfg.search_limit,
                "rebuild_interval": cfg.rebuild_interval,
            },
            level="INFO",
        )

    def _emit_store_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        level: str = "DEBUG",
    ) -> None:
        emit_memory_event(
            component="Store",
            event_type=event_type,
            payload=payload or {},
            level=level,
            event_path=self._event_path,
        )

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
            self._emit_store_event(
                "store.api_sync",
                {"force_full": force_full, "outcome": "no_sync_config"},
            )
            return False
        started = perf_counter()
        try:
            changed = sync.sync(force_full=force_full)
        except Exception as exc:
            self._emit_store_event(
                "store.api_sync",
                {
                    "force_full": force_full,
                    "outcome": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    "card_count": len(self.card_store.cards),
                },
                level="WARNING",
            )
            raise
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
        self._emit_store_event(
            "store.api_sync",
            {
                "outcome": "changed" if changed else "unchanged",
                "force_full": force_full,
                "changed": changed,
                "needs_rebuild": needs_rebuild,
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                "card_count": len(self.card_store.cards),
            },
            level="INFO" if changed else "DEBUG",
        )
        return changed

    def apply_merges(self, merges: list[tuple[str, AnyCard]]) -> list[str]:
        """Apply pre-computed dedup merges, replacing each target card in place.

        Each ``(card_id, merged_card)`` pair overwrites the existing card with
        the merged variant (re-running enrichment and API/A-mem sync, same as a
        fresh save). Failures on individual targets are logged and skipped, so
        the returned list may be shorter than ``merges``. The bank index is
        persisted once at the end iff at least one merge landed.

        Called by ``MemoryWritePipeline`` after the dedup LLM returns an
        ``update`` decision; the pipeline owns write_stats and ledger rows.

        Returns:
            The card ids that were successfully updated, in input order.
        """
        updated_ids: list[str] = []
        for card_id, merged_card in merges:
            try:
                self._insert_new_card(merged_card)
                updated_ids.append(card_id)
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] Merge into card {!r} failed: {}", card_id, exc
                )
                self._emit_store_event(
                    "store.merge_target_failed",
                    {
                        "card_id": card_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    level="WARNING",
                )
        if updated_ids:
            self._persist_index()
        self._emit_store_event(
            "store.merge",
            {
                "outcome": "updated" if updated_ids else "no_updates",
                "requested_count": len(merges),
                "updated_count": len(updated_ids),
                "updated_ids": updated_ids,
            },
            level="INFO" if updated_ids else "DEBUG",
        )
        return updated_ids

    def _insert_new_card(self, card: AnyCard) -> tuple[str, bool]:
        """Save card to storage. Returns (card_id, rebuilt) where rebuilt
        indicates whether a periodic rebuild (which includes index persist)
        was triggered."""
        card_id = self.card_store.ensure_id(card)
        enrichments: dict[str, Any] = {}

        if self.config.enable_llm_card_enrichment and self.memory_system is not None:
            analysis = self.memory_system.analyze_content(card.description)
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

        self._emit_store_event(
            "store.insert",
            {
                "outcome": "inserted",
                "card_id": card_id,
                "category": store.cards[card_id].category,
                "bank_card_count": len(store.cards),
                "sync_mode": "api" if sync is not None else "local",
                "note_sync_enabled": self.note_sync is not None,
                "enriched_fields": sorted(enrichments),
                "dedup_retrievers_invalidated": True,
                "iters_after_rebuild": self._iters_after_rebuild,
                "rebuild_interval": self.config.rebuild_interval,
                "rebuild_triggered": rebuilt,
            },
        )
        return card_id, rebuilt

    def save_card_direct(self, card: AnyCard) -> str:
        """Persist one already-normalized card, bypassing the write pipeline.

        Skips the harm gate, dedup reconciliation, write_stats, and the write
        ledger — the card is inserted as-is (with optional LLM enrichment and
        API/A-mem sync) and the bank index is flushed to disk, unless the
        insert already triggered a periodic rebuild (which persists itself).

        Use only when the ingest verdict is already decided: the pipeline's
        known-id update, program fast-path, and post-dedup add branches.
        External callers should go through ``save_card`` instead.

        Returns:
            The id the card was stored under (minted if the card had none).
        """
        card_id, rebuilt = self._insert_new_card(card)
        if not rebuilt:
            self._persist_index()
        return card_id

    def save_card(self, card: dict[str, Any] | AnyCard) -> str:
        """Save a memory card via the modular write pipeline.

        Args:
            card: Raw dict or Pydantic card to save. Normalized internally.

        Returns:
            Card ID of the saved (or deduplicated) card.
        """
        return self.write_pipeline.ingest(card)

    def sweep_harmful(self) -> list[str]:
        """Evict every card whose injection posterior is confidently harmful."""
        return self.write_pipeline.sweep()

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
        self._emit_store_event(
            "store.persist",
            {
                "outcome": "persisted",
                "index_file": self.config.index_file,
                "serialized_count": len(serialized) if serialized is not None else None,
                "bank_card_count": len(self.card_store.cards),
                "last_seen_index_mtime": self._last_seen_index_mtime,
            },
        )

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

        before_count = len(self.card_store.cards)
        previous_mtime = self._last_seen_index_mtime
        started = perf_counter()
        try:
            self.card_store.reload()
        except Exception as exc:
            logger.warning("[Memory][Store] card_store reload failed: {}", exc)
            self._emit_store_event(
                "store.refresh",
                {
                    "outcome": "reload_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "index_file": cfg.index_file,
                    "previous_mtime": previous_mtime,
                    "observed_mtime": mtime,
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                },
                level="WARNING",
            )
            return

        rebuild_outcome = "not_needed"
        if self._has_agentic and self.gam is not None and cfg.export_file.exists():
            try:
                self.gam.build_research_agent()
                self.research_agent = self.gam.agent
                self._gam_build_failed = False
                rebuild_outcome = "rebuilt"
            except MemoryRetrieverError as exc:
                logger.warning(
                    "[Memory][Store] Stale-refresh GAM rebuild failed: {}", exc
                )
                self.research_agent = None
                self._gam_build_failed = True
                rebuild_outcome = "rebuild_failed"

        self._last_seen_index_mtime = mtime
        self._emit_store_event(
            "store.refresh",
            {
                "outcome": "reloaded",
                "rebuild_outcome": rebuild_outcome,
                "index_file": cfg.index_file,
                "export_file": cfg.export_file,
                "previous_mtime": previous_mtime,
                "observed_mtime": mtime,
                "card_count_before": before_count,
                "card_count_after": len(self.card_store.cards),
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
            },
            level="INFO",
        )

    def research(
        self,
        query: str,
        memory_state: str | None = None,
        planning_request: str | None = None,
        *,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> ResearchOutput:
        """Self-healing structured search.

        Refreshes from disk if a separate writer instance has advanced the
        on-disk index, then dispatches to the GAM research agent. Falls back
        to a local-cards ResearchOutput when GAM is unavailable.
        """
        started = perf_counter()
        self._refresh_from_disk_if_stale()
        fallback_error: Exception | None = None
        if self.research_agent is not None:
            try:
                with memory_event_context(event_path=self._event_path):
                    output = self.research_agent.research(
                        query,
                        memory_state=memory_state,
                        planning_request=planning_request,
                        exclude_ids=exclude_ids,
                        random_drop_dose=random_drop_dose,
                    )
                self._emit_store_event(
                    "store.research",
                    {
                        "outcome": "ok",
                        "mode": "gam",
                        "query_chars": len(query),
                        "memory_state_chars": len(memory_state or ""),
                        "planning_request_chars": len(planning_request or ""),
                        "exclude_count": len(exclude_ids),
                        "exclude_ids": sorted(exclude_ids),
                        "random_drop_dose": random_drop_dose,
                        "raw_memory_type": type(output.raw_memory).__name__,
                        "has_raw_memory": output.raw_memory is not None,
                        "integrated_memory_chars": len(output.integrated_memory or ""),
                        "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    },
                )
                return output
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] GAM research failed, falling back to local cards: {}",
                    exc,
                )
                fallback_error = exc
        text = self._search_local_cards(query, memory_state=memory_state)
        payload = {
            "outcome": "fallback",
            "mode": "local_fallback",
            "query_chars": len(query),
            "memory_state_chars": len(memory_state or ""),
            "planning_request_chars": len(planning_request or ""),
            "exclude_count": len(exclude_ids),
            "exclude_ids": sorted(exclude_ids),
            "random_drop_dose": random_drop_dose,
            "integrated_memory_chars": len(text),
            "duration_ms": round((perf_counter() - started) * 1000.0, 3),
        }
        if fallback_error is not None:
            payload.update(
                {
                    "fallback_reason": "gam_exception",
                    "error_type": type(fallback_error).__name__,
                    "error": str(fallback_error),
                }
            )
        self._emit_store_event(
            "store.research",
            payload,
            level="INFO",
        )
        return ResearchOutput(integrated_memory=text, raw_memory=None)

    def search(self, query: str, memory_state: str | None = None) -> str:
        """Search memory cards. Tries GAM agent, then API, then local keyword match."""
        started = perf_counter()
        self._refresh_from_disk_if_stale()
        if self.api is not None:
            self._sync_from_api(force_full=False)

        fallback_error: Exception | None = None
        if self.research_agent is not None:
            try:
                with memory_event_context(event_path=self._event_path):
                    text = self.research_agent.research(
                        query, memory_state=memory_state
                    ).integrated_memory
                self._emit_store_event(
                    "store.search",
                    {
                        "outcome": "ok",
                        "mode": "gam",
                        "query_chars": len(query),
                        "memory_state_chars": len(memory_state or ""),
                        "result_chars": len(text or ""),
                        "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    },
                )
                return text
            except Exception as exc:
                logger.warning(
                    "[Memory][Store] GAM search failed, falling back to non-agentic search: {}",
                    exc,
                )
                fallback_error = exc

        if self.api is not None:
            try:
                text = self._search_via_api(query, memory_state=memory_state)
                mode = "api"
            except Exception as exc:
                self._emit_store_event(
                    "store.search",
                    {
                        "mode": "api",
                        "outcome": "exception",
                        "query_chars": len(query),
                        "memory_state_chars": len(memory_state or ""),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    },
                    level="WARNING",
                )
                raise
        else:
            text = self._search_local_cards(query, memory_state=memory_state)
            mode = "local"
        payload = {
            "outcome": "fallback" if fallback_error is not None else "ok",
            "mode": mode,
            "query_chars": len(query),
            "memory_state_chars": len(memory_state or ""),
            "result_chars": len(text or ""),
            "duration_ms": round((perf_counter() - started) * 1000.0, 3),
        }
        if fallback_error is not None:
            payload.update(
                {
                    "fallback_reason": "gam_exception",
                    "error_type": type(fallback_error).__name__,
                    "error": str(fallback_error),
                }
            )
        self._emit_store_event("store.search", payload)
        return text

    def get_card(self, card_id: str) -> AnyCard | None:
        """Return a card by ID, or None if not found."""
        return self.card_store.cards.get(card_id)

    def get_card_write_stats(self) -> dict[str, int]:
        return dict(self.card_store.write_stats)

    def rebuild(self) -> None:
        """Persist cards, re-export JSONL, rebuild GAM index and dedup retrievers."""
        started = perf_counter()
        serialized = self.card_store.serialize_all()
        self._persist_index(serialized=serialized)
        if not self._has_agentic:
            self._emit_store_event(
                "store.rebuild",
                {
                    "outcome": "persist_only_no_agentic",
                    "serialized_count": len(serialized),
                    "bank_card_count": len(self.card_store.cards),
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                },
                level="INFO",
            )
            return
        exported = False
        if self.note_sync is not None:
            self.note_sync.export_jsonl(self.config.export_file, serialized)
            exported = True
        outcome = "no_gam"
        if self.gam is not None:
            # Track state only when already ready (not during initialization)
            track_state = self._state.current == "ready"
            if track_state:
                self._state.mark_building()
            try:
                self.gam.build_research_agent()
                self.research_agent = self.gam.agent
                self._gam_build_failed = False
                outcome = "rebuilt"
                if track_state:
                    self._state.mark_ready()
            except MemoryRetrieverError as exc:
                logger.warning("[Memory][Store] GAM build failed: {}", exc)
                self.gam.clear_research_agent()
                self.research_agent = None
                self._gam_build_failed = True
                outcome = "gam_build_failed"
                if track_state:
                    self._state.mark_error(f"GAM build failed: {exc}")
        self.dedup.invalidate_retrievers()
        self._iters_after_rebuild = 0
        self._emit_store_event(
            "store.rebuild",
            {
                "outcome": outcome,
                "serialized_count": len(serialized),
                "bank_card_count": len(self.card_store.cards),
                "exported": exported,
                "export_file": self.config.export_file,
                "research_agent_ready": self.research_agent is not None,
                "dedup_retrievers_invalidated": True,
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
            },
            level="INFO" if outcome != "gam_build_failed" else "WARNING",
        )

    def delete(self, memory_id: str) -> bool:
        """Delete a card by ID or entity ID. Returns True if found and removed."""
        key = str(memory_id).strip()
        store = self.card_store
        sync = self._get_api_sync()
        if sync is not None:
            card_id = sync.delete_from_api(key)
            if card_id is None:
                self._emit_store_event(
                    "store.delete",
                    {"requested_id": key, "outcome": "not_found_api"},
                )
                return False
        else:
            resolved = store.resolve_card_id(key)
            if resolved is None:
                self._emit_store_event(
                    "store.delete",
                    {"requested_id": key, "outcome": "not_found_local"},
                )
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

        self._emit_store_event(
            "store.delete",
            {
                "requested_id": key,
                "card_id": card_id,
                "outcome": "deleted",
                "sync_mode": "api" if sync is not None else "local",
                "bank_card_count": len(store.cards),
            },
            level="INFO",
        )
        return True

    def close(self) -> None:
        if self.api is not None:
            self.api.close()
        self._emit_store_event(
            "store.close",
            {"outcome": "closed", "api_enabled": self.api is not None},
        )

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
                self._emit_store_event(
                    "store.close",
                    {
                        "outcome": "final_rebuild_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    level="WARNING",
                )
        self.close()
