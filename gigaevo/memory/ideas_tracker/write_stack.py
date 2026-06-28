"""LibrarianWriteStack: lazy assembly of the LLM-first write path.

Mirrors the reader's :class:`SelectorMemoryProvider`: a thin holder that builds
the store, admission gate, neighbor source, librarian, and consolidation agent
once, off the event loop, on first use. It also condenses the task description
into the one-line summary stamped on every card — folded into :meth:`ensure` so
there is no summary-before-stack ordering rule for the orchestrator to honour.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.llm.agents.factories import (
    create_consolidate_agent,
    create_program_author_agent,
    create_reconcile_agent,
    create_task_summary_agent,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.core.admission_gate import CardAdmissionGate
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.protocols import Evictor, ReputationModel
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.write_ledger import WriteLedger
from gigaevo.memory.ideas_tracker.dedup_policy import DedupPolicy
from gigaevo.memory.ideas_tracker.hf_cache import ensure_writable_hf_cache
from gigaevo.memory.ideas_tracker.librarian import Librarian


class LibrarianWriteStack:
    """Builds and holds the shared write-path components, lazily.

    ``backend`` is the Hydra ``memory/common/backend`` ``_partial_`` over
    ``build_local_backend`` that MemorySystem completes with the shared llm — the
    same partial it gave the read provider, so writer and reader share one bank.
    """

    def __init__(
        self,
        *,
        backend: Callable[..., Any] | None,
        llm: MultiModelRouter | None,
        task_description: str,
        evictor: Evictor | None = None,
        reputation: ReputationModel | None = None,
        checkpoint_dir: str | Any | None = None,
        dedup_policy: DedupPolicy | None = None,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self._backend = backend
        self._llm = llm
        self._task_description = task_description
        self._evictor = evictor
        self._reputation = (
            reputation if reputation is not None else BetaBinomialReputation()
        )
        self._checkpoint_dir = checkpoint_dir
        self._dedup_policy = dedup_policy if dedup_policy is not None else DedupPolicy()
        self._prompts_dir = prompts_dir
        self._store: Any | None = None
        self._gate: CardAdmissionGate | None = None
        self._librarian: Librarian | None = None
        self._neighbors: Any | None = None
        self._consolidation_agent: Any | None = None
        # genuine LLM-condensed one-liner, produced once per run; None until then
        # so the call is memoised.
        self._summary: str | None = None
        self._build_lock = asyncio.Lock()

    @property
    def store(self) -> Any | None:
        return self._store

    @property
    def gate(self) -> CardAdmissionGate | None:
        return self._gate

    @property
    def librarian(self) -> Librarian | None:
        return self._librarian

    @property
    def neighbors(self) -> Any | None:
        return self._neighbors

    @property
    def consolidation_agent(self) -> Any | None:
        return self._consolidation_agent

    @property
    def task_description_summary(self) -> str:
        return self._summary or ""

    async def ensure(self) -> None:
        """Build the store, gate, neighbors, librarian, and consolidation agent
        once. The build loads the embedding model (seconds of blocking I/O) so it
        runs off the event loop; the lock collapses a concurrent first-write race
        down to a single build."""
        if self._librarian is not None:
            return
        async with self._build_lock:
            if self._librarian is not None:
                return
            summary = await self.ensure_summary()
            await asyncio.to_thread(self._build, summary)

    async def ensure_summary(self) -> str:
        """Condense the task description into a one-line summary, once per run.

        Falls back to the full task description on any LLM failure (and to the
        empty string when there is no task text), so a memory-LLM hiccup can
        never block the write path.
        """
        if self._summary is not None:
            return self._summary
        if not self._task_description:
            self._summary = ""
            return self._summary
        try:
            agent = create_task_summary_agent(self._llm, prompts_dir=self._prompts_dir)
            resp = await agent.arun(task_description=self._task_description)
            self._summary = resp.summary.strip() or self._task_description
        except Exception as exc:
            logger.warning(
                "[Memory][IdeaTracker] task-summary LLM failed ({}); falling back "
                "to the full task description",
                exc,
            )
            self._summary = self._task_description
        return self._summary

    def _build(self, summary: str) -> None:
        # The embedding model the neighbor source loads next follows HF_HOME and
        # friends; redirect them to a writable dir before that download begins.
        ensure_writable_hf_cache()
        policy = self._dedup_policy
        store = self._backend(
            checkpoint_dir=self._checkpoint_dir, evictor=self._evictor
        )
        gate = CardAdmissionGate(
            store=store,
            evictor=self._evictor
            if self._evictor is not None
            else HarmEvictor(reputation=self._reputation),
            ledger=WriteLedger(store.checkpoint_path / "write_ledger.jsonl"),
        )
        # the backend IS the neighbor source: its nearest-card contract method
        # feeds both the online pre-gate and the batch consolidation pass.
        neighbors = store
        librarian = Librarian(
            agent=create_reconcile_agent(
                self._llm, self._task_description, prompts_dir=self._prompts_dir
            ),
            program_author=create_program_author_agent(
                self._llm, self._task_description, prompts_dir=self._prompts_dir
            ),
            gate=gate,
            store=store,
            neighbors=neighbors,
            eps=policy.online_eps,
            top_k=policy.online_top_k,
            max_cards=policy.max_cards_per_diff,
            task_description=self._task_description,
            task_description_summary=summary,
        )
        self._store = store
        self._gate = gate
        self._neighbors = neighbors
        self._librarian = librarian
        self._consolidation_agent = create_consolidate_agent(
            self._llm, self._task_description, prompts_dir=self._prompts_dir
        )
