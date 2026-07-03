"""Periodic consolidation: a batch near-duplicate merge over the card bank.

The online librarian is greedy and order-dependent — two same-lever cards can
both enter as NEW if neither pulls the other into the reconcile agent's top-k
context at birth. This sweep is the standard drift fix: run the *same*
``NeighborSource`` nearest-card primitive over the whole bank instead of one
note, surface each card's top-``k`` neighbors as merge *candidates* (pure
top-k by rank — no distance cut), and let the consolidate agent be the
precision arbiter — it folds a candidate pair into one canonical card via
``CardAdmissionGate.merge`` only when it rules they name the same lever, and
abstains otherwise so generous candidate recall can never force-merge distinct
cards. On abstain the pass tries the next-nearest candidate. The absorbed card's
provenance is preserved on the survivor; the absorbed id is then removed.
Idempotent — a second run over a deduped bank finds no foldable pair and merges
nothing. ``ConsolidationScheduler`` throttles the pass to one background run per
``every_n`` cards written, serialized under the shared write lock.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.llm.agents.consolidate_cards import ConsolidateAgent
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.events import MemoryConsolidationPass, emit_memory_event
from gigaevo.memory.storage.base import MemoryStore, ScoredCard
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.memory.write.librarian import NeighborSource

if TYPE_CHECKING:
    from gigaevo.memory.write.writer import LibrarianWriteStack

# Grace given to a cancelled final pass to unwind before it is abandoned as an
# orphan, so a pass that swallows CancelledError cannot wedge engine teardown.
_CANCEL_GRACE_S = 1.0


async def consolidate(
    *,
    store: MemoryStore,
    gate: CardAdmissionGate,
    neighbors: NeighborSource,
    agent: ConsolidateAgent,
    k: int = 5,
    reviewed: set[frozenset[tuple[str, str]]] | None = None,
) -> int:
    """Fold near-duplicate idea cards into canonical cards. Returns merge count.

    Deletion of absorbed cards is deferred to the end of the pass so the bank is
    stable while neighbors are ranked, and so the ``consumed`` set is the sole
    guard against re-merging a pair in both directions.

    ``reviewed`` records every pair the agent declined, keyed by (id,
    description) so a ruling is invalidated the moment either card's prose
    changes. Pass a caller-owned set to carry declines across passes — without
    a distance cut every standing near-pair would otherwise re-pay its arbiter
    call on every pass.
    """
    cards = store.snapshot()
    consumed: set[str] = set()
    absorbed: list[str] = []
    # The merge ruling is symmetric, so once the agent declines an unordered pair
    # we must not pay a second LLM call to re-review it as (partner, card) — in
    # this pass or any later one.
    if reviewed is None:
        reviewed = set()
    merges = 0
    # Absorbed partners are deleted in a finally so a mid-pass agent failure
    # cannot leave an already-merged partner in the bank (its evidence is now
    # on the survivor — an undeleted partner would be double-counted).
    try:
        for card in cards:
            # Only idea cards drift into duplicates; program exemplar cards are
            # identity-keyed and re-authored each sweep, so never merge them.
            if card.kind is not CardKind.INSIGHT or card.id in consumed:
                continue
            desc = (card.description or "").strip()
            if not desc:
                continue
            # nearest() truncates at its k; the query card's own description and
            # this-pass consumed cards still occupy index slots (deletion is
            # deferred to the finally), so they crowd the fixed top-k and can
            # hide a valid partner past the cutoff. Over-fetch past self + every
            # consumed id so k genuine candidates always reach the arbiter.
            fetch = k + 1 + len(consumed)
            candidates = _drift_candidates(
                card, neighbors.nearest(desc, fetch, CardKind.INSIGHT), consumed
            )
            for partner in candidates:
                pair = frozenset(
                    {(card.id, card.description), (partner.id, partner.description)}
                )
                if pair in reviewed:
                    continue
                # Top-k only recalls candidates; the agent is the precision
                # arbiter. On abstain (the two are not the same lever) move on to
                # the next-nearest candidate rather than force-folding them.
                decision = await agent.arun(card_a=card, card_b=partner)
                if not decision.merge or decision.card is None:
                    reviewed.add(pair)
                    continue
                union = decision.card
                fid = gate.merge(
                    card.id,
                    Card(
                        # The survivor is ``card`` (target_id); the partner is
                        # folded away and deleted. The gate reads the submitted
                        # card's id as the ledger's incoming_id, so it must be the
                        # absorbed partner's — else the deleted card has no
                        # ledger/replay trace.
                        id=partner.id,
                        description=union.description,
                        explanation_summary=union.explanation_summary,
                        # Trust the agent's curated union keyword set; the gate's
                        # replace-on-merge fold takes it verbatim. Re-unioning the
                        # partner's raw list here would re-bloat the survivor.
                        keywords=tuple(union.keywords),
                        programs=partner.programs,
                        # Carry the partner's own absorbed-id chain forward so a
                        # multi-hop absorption keeps re-aliasing the earliest id
                        # onto this survivor (merge_cards adds partner.id itself).
                        absorbed_ids=partner.absorbed_ids,
                        gain_events=partner.gain_events,
                        task_description=card.task_description
                        or partner.task_description,
                        task_description_summary=(
                            card.task_description_summary
                            or partner.task_description_summary
                        ),
                    ),
                )
                # Queue the partner for deletion ONLY after a committed fold. The
                # gate returns a truthy id iff it folded the partner's evidence
                # onto the survivor and persisted it; it returns "" without
                # folding on a harmful-union eviction or a target miss, and it
                # raises if the underlying store persist fails (the harm-path
                # ``delete`` and ``apply_merges`` both persist unwrapped). Queuing
                # before the fold would orphan the partner on either the empty
                # return or that raise; queuing after means the finally only ever
                # deletes committed folds.
                if not fid:
                    consumed.add(card.id)
                    break
                absorbed.append(partner.id)
                consumed.add(card.id)
                consumed.add(partner.id)
                merges += 1
                break
    finally:
        # Delete each folded partner independently. store.delete drops the card
        # from the in-memory bank before its fallible disk persist, so guarding
        # per-id means one partner's persist failure cannot abort the loop and
        # leave the remaining partners live-yet-folded — a permanent gain
        # double-count (their evidence sits on the survivor while they still
        # score their own). A failed on-disk flush is logged and heals on the
        # next successful sweep persist.
        for cid in absorbed:
            try:
                store.delete(cid)
            except Exception as exc:
                logger.warning(
                    "[Memory][Consolidation] failed to delete folded partner {} "
                    "({}); removal flushes on the next sweep persist",
                    cid,
                    exc,
                )
    return merges


def _drift_candidates(
    card: Card,
    hits: list[ScoredCard],
    consumed: set[str],
) -> list[Card]:
    out: list[Card] = []
    for hit in hits:
        if hit.card.id == card.id or hit.card.id in consumed:
            continue
        out.append(hit.card)
    return out


class ConsolidationScheduler:
    """Throttles and serializes background bank consolidation.

    Counts cards written across sweeps and dispatches exactly one background
    consolidation pass per ``every_n``. The pass runs under the shared write
    lock so it can never interleave with a live write sweep, yet dispatch is
    non-blocking so the sweep that triggered it returns immediately. Declined
    pair rulings are remembered across passes (content-keyed), so a frequent
    cadence only pays the arbiter for pairs it has not already ruled on.
    """

    def __init__(
        self,
        *,
        stack: LibrarianWriteStack,
        run_lock: asyncio.Lock,
        every_n: int,
        k: int,
    ) -> None:
        self._stack = stack
        self._run_lock = run_lock
        self._every_n = every_n
        self._k = k
        self._writes_since = 0
        self._failures = 0
        self._task: asyncio.Task | None = None
        self._reviewed: set[frozenset[tuple[str, str]]] = set()

    @property
    def writes_since(self) -> int:
        return self._writes_since

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def note_writes(self, written: int) -> None:
        """Accumulate cards written and schedule one consolidation pass per
        ``every_n``. The cadence counter is consumed only when a pass is actually
        dispatched, so a dispatch that cannot run (un-built stack, a pass already
        in flight, or no running loop) leaves the writes pending for a later
        sweep rather than silently disabling consolidation."""
        if self._every_n <= 0 or written <= 0:
            return
        self._writes_since += written
        if self._writes_since >= self._every_n and self.schedule():
            self._writes_since = 0

    def schedule(self) -> bool:
        """Dispatch one background consolidation pass. Returns True iff a task was
        actually created; False when the stack is un-built, a pass is in flight,
        or there is no running loop."""
        if self._stack.store is None or self._stack.gate is None:
            return False
        if self._task is not None and not self._task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False  # no running loop (sync context); defer to a later increment
        self._task = loop.create_task(self._run())
        return True

    async def drain(self, *, timeout: float | None = None) -> None:
        """Await an in-flight consolidation pass so a pass scheduled by the final
        post-run sweep completes before the event loop is torn down — ``asyncio.run``
        cancels pending tasks on exit, which would otherwise silently drop the last
        consolidation. Bounded by ``timeout`` so a stalled memory-LLM call in that
        final pass cannot hang engine shutdown; on overrun the pass is cancelled."""
        task = self._task
        if task is None or task.done():
            return
        # asyncio.wait (not wait_for) for the cancel step: a pass that swallows
        # CancelledError is abandoned as an orphan after a short grace instead of
        # blocking teardown forever waiting for the cancel to be honoured.
        _, pending = await asyncio.wait({task}, timeout=timeout)
        if not pending:
            return
        task.cancel()
        _, still_pending = await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)
        if still_pending:
            logger.warning(
                "[Memory][Consolidation] final pass ignored cancel within {}s "
                "grace; abandoned as orphan at shutdown",
                _CANCEL_GRACE_S,
            )
        else:
            logger.warning(
                "[Memory][Consolidation] final pass exceeded {}s; cancelled at "
                "shutdown",
                timeout,
            )

    async def _run(self) -> None:
        # Consolidation rewrites the bank, so it runs under the same write lock as
        # a sweep — never interleaved with one — but is dispatched in the
        # background so the triggering sweep is not blocked waiting for it.
        store = self._stack.store
        gate = self._stack.gate
        neighbors = self._stack.neighbors
        agent = self._stack.consolidation_agent
        if store is None or gate is None or neighbors is None or agent is None:
            return  # un-built stack (schedule() guards this) — nothing to fold
        async with self._run_lock:
            try:
                merged = await consolidate(
                    store=store,
                    gate=gate,
                    neighbors=neighbors,
                    agent=agent,
                    k=self._k,
                    reviewed=self._reviewed,
                )
            except Exception as exc:
                self._failures += 1
                logger.warning(
                    "[Memory][Consolidation] pass failed ({}); skipping",
                    exc,
                )
                emit_memory_event(
                    MemoryConsolidationPass(
                        outcome="failed",
                        failures=self._failures,
                        error=str(exc),
                    )
                )
                return
            if merged:
                logger.info(
                    "[Memory][Consolidation] merged {} near-dup cards",
                    merged,
                )
            emit_memory_event(MemoryConsolidationPass(outcome="ok", merged=merged))
