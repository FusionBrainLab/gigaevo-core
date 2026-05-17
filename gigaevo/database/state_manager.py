import asyncio
from datetime import UTC, datetime

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState, validate_transition

# States after which the DagRunner never accesses the program again.
# Evict per-program locks for these states to prevent unbounded memory growth.
# DONE programs are picked up by EvolutionEngine (different ProgramStateManager),
# so the DagRunner's lock is no longer needed.
_TERMINAL_STATES = frozenset({ProgramState.DISCARDED, ProgramState.DONE})


def register_external_terminal_state(program: Program, new_state: ProgramState) -> None:
    """Pin an externally-finalized program at a terminal state, in memory.

    The narrow case is cross-engine migration ingestion: a fresh
    :class:`Program` rehydrated from a :class:`MigrantEnvelope` carries
    the source engine's last-known state, which has no causal
    predecessor in the local FSM. The in-run transition table is an
    in-run invariant, so :func:`validate_transition` is deliberately
    not consulted.

    No locking is needed because the program object is freshly
    constructed at the call site (typically with a freshly-generated
    UUID) and is not yet observable to any other coroutine. The
    function intentionally names the bypass so a code reviewer can
    grep for every cross-run terminal-state ingestion.

    ``new_state`` must be terminal (DONE or DISCARDED); restricting the
    bypass to terminal states keeps the exception out of paths where
    the FSM legitimately governs progression.

    Raises:
        ValueError: ``new_state`` is not a terminal state, signalling a
            misuse where the in-run FSM should govern the transition.
    """
    if new_state not in _TERMINAL_STATES:
        raise ValueError(
            f"register_external_terminal_state requires a terminal state; "
            f"got {new_state!r}. Use ProgramStateManager.set_in_memory_state "
            f"or ProgramStateManager.set_program_state for non-terminal "
            f"transitions."
        )
    program.state = new_state


class ProgramStateManager:
    """
    Serialize per-program updates (stage results & program state) and persist them.
    Locks ensure no in-process races on the same Program id.
    """

    def __init__(self, storage: ProgramStorage):
        self.storage = storage
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, program_id: str) -> asyncio.Lock:
        return self._locks.setdefault(program_id, asyncio.Lock())

    async def mark_stage_running(
        self,
        program: Program,
        stage_name: str,
        *,
        started_at: datetime | None = None,
    ) -> None:
        """Mark a stage as RUNNING in-memory only (not persisted to Redis).

        The RUNNING state is only used locally during DAG execution.
        The final COMPLETED/FAILED state (with started_at preserved) is
        persisted by update_stage_result().
        """
        async with self._lock_for(program.id):
            ts = started_at or datetime.now(UTC)
            # Preserve input_hash from existing result if present
            existing = program.stage_results.get(stage_name)
            input_hash = existing.input_hash if existing else None
            program.stage_results[stage_name] = ProgramStageResult(
                status=StageState.RUNNING,
                started_at=ts,
                input_hash=input_hash,
            )
            # No Redis write — RUNNING state is transient and never read back.
            # DAG reads stage state from in-memory program.stage_results.

    async def update_stage_result(
        self,
        program: Program,
        stage_name: str,
        result: ProgramStageResult,
    ) -> None:
        """Set a stage result and persist the entire program.

        Note: This persists the ENTIRE program object (metrics, metadata, lineage, etc.),
        not just the stage_result. This is why additional snapshots are not needed.
        Uses write_exclusive (2 RT) because the DAG holds exclusive ownership.
        """
        async with self._lock_for(program.id):
            program.stage_results[stage_name] = result
            await self.storage.write_exclusive(program)

    async def write_exclusive(self, program: Program) -> None:
        """Fast write without WATCH/MERGE. Safe only during DAG execution (exclusive ownership)."""
        async with self._lock_for(program.id):
            await self.storage.write_exclusive(program)

    async def update_program(self, program: Program) -> None:
        """Update program (for metadata, lineage, etc.) with proper locking."""
        async with self._lock_for(program.id):
            await self.storage.update(program)

    async def set_program_state(
        self, program: Program, new_state: ProgramState
    ) -> None:
        """Set program state with validation and atomic persistence."""
        async with self._lock_for(program.id):
            if program.state == new_state:
                logger.debug(
                    "[ProgramStateManager] {} already in state {}, skipping",
                    program.short_id,
                    new_state,
                )
                return

            old_state = program.state
            try:
                validate_transition(old_state, new_state)
            except ValueError as e:
                logger.error(
                    "[ProgramStateManager] Invalid state transition for {}: {}",
                    program.short_id,
                    e,
                )
                raise

            program.state = new_state
            old = old_state.value if old_state else None
            # Use fast path (2 RT) when old state is known — safe because
            # per-program asyncio.Lock above prevents concurrent writes.
            if old is not None:
                await self.storage.fast_state_transition(program, old, new_state.value)
            else:
                await self.storage.atomic_state_transition(
                    program, old, new_state.value
                )

            logger.debug(
                "[ProgramStateManager] {} {} → {}",
                program.short_id,
                old_state,
                new_state,
            )

        # Evict after releasing — terminal programs are never transitioned again.
        if new_state in _TERMINAL_STATES:
            self._locks.pop(program.id, None)

    async def set_in_memory_state(
        self, program: Program, new_state: ProgramState
    ) -> None:
        """Validate the FSM transition and update ``program.state`` in memory.

        Companion to :meth:`set_program_state` for call sites that have
        already persisted the new state via a separate batch operation
        (``batch_transition_by_ids`` / ``batch_transition_state``) and
        only need to bring the in-memory :class:`Program` instance in
        sync with the storage write. Acquires the same per-program lock
        :meth:`set_program_state` uses so an in-memory mirror cannot
        race a concurrent canonical transition.

        Raises:
            ValueError: the (current, new) pair is not in the gigaevo
                program-state FSM. Surfacing this at the call boundary
                catches illegal mirror writes that would otherwise leave
                the in-memory state out of sync with the persisted state.
        """
        async with self._lock_for(program.id):
            if program.state == new_state:
                return
            validate_transition(program.state, new_state)
            program.state = new_state
        if new_state in _TERMINAL_STATES:
            self._locks.pop(program.id, None)
