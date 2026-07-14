"""LocalMemoryStore — CardBank ∘ VectorIndex ∘ ResearchAgent.

The persisted bank is the source of truth; the process-local in-memory vector
index follows it through incremental writes and full rebuilds on startup and
cross-process refresh. Retrieval failures degrade to empty results; bank
corruption raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import threading
from time import perf_counter
from typing import Any

from loguru import logger

from gigaevo.exceptions import MergeAborted, StorageError
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.events import (
    MemoryResearch,
    MemoryStoreSync,
    MemoryStoreWrite,
    emit_memory_event,
)
from gigaevo.memory.storage.bank import CardBank, CardBankFileLock, new_card_id
from gigaevo.memory.storage.base import (
    MemoryStore,
    MergeRetireResult,
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
        self._policy_identifiers = {
            "embedding_model": config.embed.embedding_model,
            "retrieval_models": tuple(getattr(llm, "model_names", ()) or ()),
        }
        self._state = StoreState.INITIALIZING
        self._lock = threading.RLock()
        self._bank_lock_state = threading.local()
        try:
            self._bank = CardBank(config.bank_file)
            self._index = VectorIndex(config.embed)
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
        with self._lock:
            with self._bank_file_lock(exclusive=True):
                self._refresh_from_disk_locked()
                before = self._bank.snapshot()
                if not card.id:
                    card = card.model_copy(update={"id": new_card_id()})
                try:
                    self._bank.put(card)
                    self._bank.persist()
                except Exception:
                    self._bank.restore_snapshot(before)
                    raise
            self._index_write(self._index.upsert, [card])
            bank_count = len(self._bank)
        emit_memory_event(
            MemoryStoreWrite(
                op="save",
                outcome="ok",
                card_ids=(card.id,),
                bank_count=bank_count,
            )
        )
        return card.id

    def update(
        self, card_id: str, transform: Callable[[Card], Card | None]
    ) -> Card | None:
        with self._lock:
            index_write: tuple[Callable[..., None], list[Card] | list[str]] | None = (
                None
            )
            with self._bank_file_lock(exclusive=True):
                self._refresh_from_disk_locked()
                current = self._bank.get(card_id)
                if current is None:
                    affected = None
                    outcome = "not_found"
                else:
                    before = self._bank.snapshot()
                    replacement = transform(current)
                    if replacement is not None and replacement.id != card_id:
                        raise ValueError("atomic card update cannot change the card id")
                    if replacement == current:
                        affected = current
                        outcome = "noop"
                    else:
                        try:
                            if replacement is None:
                                self._bank.remove(card_id)
                                affected = current
                                index_write = (self._index.remove, [card_id])
                            else:
                                self._bank.put(replacement)
                                affected = replacement
                                index_write = (self._index.upsert, [replacement])
                            self._bank.persist()
                        except Exception:
                            self._bank.restore_snapshot(before)
                            raise
                        outcome = "ok"
            if index_write is not None:
                write, args = index_write
                self._index_write(write, args)
            bank_count = len(self._bank)
        emit_memory_event(
            MemoryStoreWrite(
                op="update",
                outcome=outcome,
                card_ids=(card_id,),
                bank_count=bank_count,
            )
        )
        return affected

    def get(self, card_id: str) -> Card | None:
        with self._lock:
            with self._bank_file_lock(exclusive=False):
                self._refresh_from_disk_locked()
            return self._bank.get(card_id)

    def delete(self, card_id: str) -> bool:
        with self._lock:
            with self._bank_file_lock(exclusive=True):
                self._refresh_from_disk_locked()
                before = self._bank.snapshot()
                removed = self._bank.remove(card_id)
                if removed:
                    try:
                        self._bank.persist()
                    except Exception:
                        self._bank.restore_snapshot(before)
                        raise
                    self._index_write(self._index.remove, [card_id])
            bank_count = len(self._bank)
        emit_memory_event(
            MemoryStoreWrite(
                op="delete",
                outcome="ok" if removed else "not_found",
                card_ids=(card_id,),
                bank_count=bank_count,
            )
        )
        return removed

    def snapshot(self) -> tuple[Card, ...]:
        with self._lock:
            with self._bank_file_lock(exclusive=False):
                self._refresh_from_disk_locked()
            return self._bank.snapshot()

    def merge_retire(
        self,
        target_id: str,
        partner_id: str,
        fold: Callable[[Card, Card | None], Card | None],
    ) -> MergeRetireResult:
        with self._lock:
            survivor: Card | None = None
            removed_ids: list[str] = []
            with self._bank_file_lock(exclusive=True):
                self._refresh_from_disk_locked()
                target = self._bank.get(target_id)
                partner = self._bank.get(partner_id) if partner_id else None
                if target is None:
                    result = MergeRetireResult(outcome="target_missing")
                elif partner_id == target_id:
                    logger.warning(
                        "[Memory][Store] aborted merge with identical target and "
                        "partner id {}",
                        target_id,
                    )
                    result = MergeRetireResult(outcome="aborted")
                elif partner is not None and target_id in partner.absorbed_ids:
                    logger.warning(
                        "[Memory][Store] aborted reverse merge {} <- {}; target is "
                        "already absorbed by partner",
                        target_id,
                        partner_id,
                    )
                    result = MergeRetireResult(outcome="aborted")
                else:
                    try:
                        survivor = fold(target, partner)
                    except MergeAborted:
                        result = MergeRetireResult(outcome="aborted")
                    else:
                        if survivor is not None and survivor.id != target_id:
                            raise ValueError(
                                "atomic merge survivor must keep the target id"
                            )
                        if (
                            survivor is not None
                            and survivor.id in survivor.absorbed_ids
                        ):
                            logger.warning(
                                "[Memory][Store] aborted merge whose survivor {} "
                                "absorbs itself",
                                survivor.id,
                            )
                            survivor = None
                            result = MergeRetireResult(outcome="aborted")
                        else:
                            before = self._bank.snapshot()
                            try:
                                if survivor is None:
                                    self._bank.remove(target_id)
                                    removed_ids.append(target_id)
                                    result = MergeRetireResult(outcome="retired")
                                else:
                                    self._bank.put(survivor)
                                    result = MergeRetireResult(outcome="merged")
                                if partner is not None:
                                    self._bank.remove(partner_id)
                                    removed_ids.append(partner_id)
                                self._bank.persist()
                            except Exception:
                                self._bank.restore_snapshot(before)
                                raise
            if result.outcome == "merged" and survivor is not None:
                self._index_write(self._index.upsert, [survivor])
            if removed_ids:
                self._index_write(self._index.remove, removed_ids)
            bank_count = len(self._bank)
        emit_memory_event(
            MemoryStoreWrite(
                op="merge",
                outcome=(
                    "ok"
                    if result.outcome in {"merged", "retired"}
                    else "not_found"
                    if result.outcome == "target_missing"
                    else "noop"
                ),
                card_ids=tuple(
                    dict.fromkeys(
                        card_id for card_id in (target_id, partner_id) if card_id
                    )
                ),
                bank_count=bank_count,
            )
        )
        return result

    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]:
        try:
            with self._lock:
                with self._bank_file_lock(exclusive=False):
                    self._refresh_from_disk_locked()
                hits = self._index.query(
                    self._config.embed.nearest_scope, text, k, kind=kind
                )
                return [
                    ScoredCard(card=card, distance=hit.distance)
                    for hit in hits
                    if (card := self._bank.get(hit.card_id)) is not None
                ]
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Store] nearest() failed; returning no neighbors"
            )
            return []

    async def research(self, request: ResearchRequest) -> ResearchResult:
        started = perf_counter()
        if self._agent is None:
            return self._finish_research(started, request, ResearchResult())
        with self._lock:
            with self._bank_file_lock(exclusive=False):
                self._refresh_from_disk_locked()
        try:
            result = await self._agent.research(request)
        except Exception as exc:
            logger.opt(exception=True).warning(
                "[Memory][Store] research() failed; returning no candidates"
            )
            return self._finish_research(started, request, ResearchResult(), exc)
        return self._finish_research(started, request, result)

    def rebuild(self) -> None:
        with self._lock:
            self._transition(StoreState.BUILDING)
            try:
                with self._bank_file_lock(exclusive=False):
                    self._bank.reload()
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

    def _refresh_from_disk_locked(self) -> None:
        """Refresh this process' bank/index view if another process persisted."""
        if getattr(self._bank_lock_state, "depth", 0) > 1:
            return
        if self._bank.reload_if_changed():
            self._sync_index("refresh")

    @contextmanager
    def _bank_file_lock(self, *, exclusive: bool) -> Iterator[None]:
        depth = getattr(self._bank_lock_state, "depth", 0)
        if depth:
            mode = getattr(self._bank_lock_state, "mode", "shared")
            if exclusive and mode == "shared":
                raise StorageError(
                    "cannot upgrade a shared card-bank lock to exclusive"
                )
            self._bank_lock_state.depth = depth + 1
            try:
                yield
            finally:
                self._bank_lock_state.depth = depth
            return

        mode = "exclusive" if exclusive else "shared"
        with CardBankFileLock(self._bank.lock_path, exclusive=exclusive):
            self._bank_lock_state.depth = 1
            self._bank_lock_state.mode = mode
            try:
                yield
            finally:
                self._bank_lock_state.depth = 0
                self._bank_lock_state.mode = None

    def _transition(self, new: StoreState) -> None:
        validate_transition(self._state, new)
        self._state = new
