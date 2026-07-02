"""LocalMemoryStore — CardBank ∘ VectorIndex ∘ ResearchAgent.

The bank is the source of truth and is persisted on every write; the vector
index follows it (incremental upserts per write, diff-sync rebuild to heal or
absorb external writers). Retrieval failures degrade to empty results; bank
corruption raises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.events import (
    MemoryResearch,
    MemoryStoreSync,
    MemoryStoreWrite,
    emit_memory_event,
)
from gigaevo.memory.storage.bank import CardBank, new_card_id
from gigaevo.memory.storage.base import (
    MemoryStore,
    ResearchRequest,
    ResearchResult,
    ScoredCard,
)
from gigaevo.memory.storage.config import StoreConfig
from gigaevo.memory.storage.index import VectorIndex
from gigaevo.memory.storage.research import ResearchAgent
from gigaevo.memory.storage.state import StoreState, validate_transition


class LocalMemoryStore(MemoryStore):
    def __init__(
        self,
        config: StoreConfig,
        llm: Any | None = None,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self._config = config
        self._state = StoreState.INITIALIZING
        try:
            self._bank = CardBank(config.bank_file)
            self._index = VectorIndex(config.index_dir, config.embed)
            self._agent = (
                ResearchAgent(
                    llm,
                    self._bank,
                    self._index,
                    config.embed,
                    config.research,
                    config.resolved_query_scopes,
                    str(prompts_dir) if prompts_dir is not None else None,
                )
                if llm is not None
                else None
            )
            self._sync_index("rebuild")
        except Exception:
            self._transition(StoreState.ERROR)
            raise
        self._transition(StoreState.READY)

    @property
    def is_ready(self) -> bool:
        return self._state is StoreState.READY

    @property
    def state(self) -> StoreState:
        return self._state

    def save(self, card: Card) -> str:
        self._refresh_if_stale()
        if not card.id:
            card = card.model_copy(update={"id": new_card_id()})
        self._bank.put(card)
        self._bank.persist()
        self._index_write(self._index.upsert, [card])
        emit_memory_event(
            MemoryStoreWrite(
                op="save",
                outcome="ok",
                card_ids=(card.id,),
                bank_count=len(self._bank),
            )
        )
        return card.id

    def get(self, card_id: str) -> Card | None:
        return self._bank.get(card_id)

    def delete(self, card_id: str) -> bool:
        self._refresh_if_stale()
        removed = self._bank.remove(card_id)
        if removed:
            self._bank.persist()
            self._index_write(self._index.remove, [card_id])
        emit_memory_event(
            MemoryStoreWrite(
                op="delete",
                outcome="ok" if removed else "not_found",
                card_ids=(card_id,),
                bank_count=len(self._bank),
            )
        )
        return removed

    def snapshot(self) -> tuple[Card, ...]:
        self._refresh_if_stale()
        return self._bank.snapshot()

    def apply_merges(self, merged: Sequence[Card]) -> list[str]:
        self._refresh_if_stale()
        saved: list[Card] = []
        for card in merged:
            if not card.id:
                card = card.model_copy(update={"id": new_card_id()})
            self._bank.put(card)
            saved.append(card)
        if saved:
            self._bank.persist()
            self._index_write(self._index.upsert, saved)
        emit_memory_event(
            MemoryStoreWrite(
                op="merge",
                outcome="ok" if saved else "noop",
                card_ids=tuple(card.id for card in saved),
                bank_count=len(self._bank),
            )
        )
        return [card.id for card in saved]

    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]:
        self._refresh_if_stale()
        try:
            hits = self._index.query(
                self._config.embed.nearest_scope, text, k, kind=kind
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Store] nearest() failed; returning no neighbors"
            )
            return []
        return [
            ScoredCard(card=card, distance=hit.distance)
            for hit in hits
            if (card := self._bank.get(hit.card_id)) is not None
        ]

    async def research(self, request: ResearchRequest) -> ResearchResult:
        started = perf_counter()
        self._refresh_if_stale()
        if self._agent is None:
            return self._finish_research(started, request, ResearchResult())
        try:
            result = await self._agent.research(request)
        except Exception as exc:
            logger.opt(exception=True).warning(
                "[Memory][Store] research() failed; returning no candidates"
            )
            return self._finish_research(started, request, ResearchResult(), exc)
        return self._finish_research(started, request, result)

    def rebuild(self) -> None:
        self._transition(StoreState.BUILDING)
        try:
            self._bank.refresh_if_stale()
            self._sync_index("rebuild")
        except Exception:
            self._transition(StoreState.ERROR)
            raise
        self._transition(StoreState.READY)

    def close(self) -> None:
        logger.debug(
            "[Memory][Store] closed ({} cards at {})",
            len(self._bank),
            self._config.path,
        )

    def _finish_research(
        self,
        started: float,
        request: ResearchRequest,
        result: ResearchResult,
        error: Exception | None = None,
    ) -> ResearchResult:
        if error is not None:
            outcome = "failed"
        else:
            outcome = "ok" if result.cards else "empty"
        emit_memory_event(
            MemoryResearch(
                outcome=outcome,
                iterations=result.iterations,
                query_chars=len(request.query),
                exclude_count=len(request.exclude_ids),
                candidate_ids=tuple(card.id for card in result.cards),
                duration_ms=(perf_counter() - started) * 1000.0,
                error=str(error) if error is not None else "",
            )
        )
        return result

    def _refresh_if_stale(self) -> None:
        if not self._bank.refresh_if_stale():
            return
        try:
            self._sync_index("refresh")
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Store] index refresh failed; serving the refreshed "
                "bank on a stale index until the next rebuild"
            )

    def _sync_index(self, op: str) -> None:
        started = perf_counter()
        cards = self._bank.snapshot()
        try:
            self._index.rebuild(cards)
        except Exception as exc:
            emit_memory_event(
                MemoryStoreSync(
                    op=op,
                    outcome="failed",
                    card_count=len(cards),
                    duration_ms=(perf_counter() - started) * 1000.0,
                    error=str(exc),
                )
            )
            raise
        emit_memory_event(
            MemoryStoreSync(
                op=op,
                outcome="ok",
                card_count=len(cards),
                duration_ms=(perf_counter() - started) * 1000.0,
            )
        )

    def _index_write(self, write: Callable[..., None], *args: Any) -> None:
        """Index writes are best-effort — the bank is the source of truth and
        :meth:`rebuild` heals the index; a failed write must not lose cards."""
        try:
            write(*args)
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Store] index write failed; index heals on next rebuild"
            )

    def _transition(self, new: StoreState) -> None:
        validate_transition(self._state, new)
        self._state = new
