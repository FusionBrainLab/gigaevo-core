import asyncio
from datetime import datetime, timezone

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState, validate_transition


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
        """Mark a stage as RUNNING in-memory (not persisted to Redis).

        The RUNNING state is only used locally during DAG execution.
        The final COMPLETED/FAILED state (with started_at preserved) is
        persisted by update_stage_result().
        """
        async with self._lock_for(program.id):
            ts = started_at or datetime.now(timezone.utc)
            # Preserve input_hash from existing result if present
            existing = program.stage_results.get(stage_name)
            input_hash = existing.input_hash if existing else None
            program.stage_results[stage_name] = ProgramStageResult(
                status=StageState.RUNNING,
                started_at=ts,
                input_hash=input_hash,
            )
            await self.storage.update(program)

    async def update_stage_result(
        self,
        program: Program,
        stage_name: str,
        result: ProgramStageResult,
    ) -> None:
        """Set a stage result and persist the entire program.

        Note: This persists the ENTIRE program object (metrics, metadata, lineage, etc.),
        not just the stage_result. This is why additional snapshots are not needed.
        """
        async with self._lock_for(program.id):
            program.stage_results[stage_name] = result
            await self.storage.update(program)

    async def update_program(self, program: Program) -> None:
        """Update program (for metadata, lineage, etc.) with proper locking."""
        async with self._lock_for(program.id):
            await self.storage.update(program)

    async def set_program_state(
        self, program: Program, new_state: ProgramState
    ) -> None:
        """Set program state with validation and atomic persistence."""
        async with self._lock_for(program.id):
            logger.debug(
                f"[ProgramStateManager] Setting program {program.id[:8]} state from {program.state} to {new_state}"
            )

            if program.state == new_state:
                logger.debug(
                    f"[ProgramStateManager] Program {program.id[:8]} already in state {new_state}, skipping"
                )
                return

            old_state = program.state
            try:
                validate_transition(old_state, new_state)
            except ValueError as e:
                logger.error(
                    f"[ProgramStateManager] Invalid state transition for {program.id[:8]}: {e}"
                )
                raise

            # Update program object
            program.state = new_state
            logger.debug(
                f"[ProgramStateManager] Updated program {program.id[:8]} state to {new_state}"
            )

            old = old_state.value if old_state else None
            await self.storage.atomic_state_transition(program, old, new_state.value)

            logger.debug(
                f"[ProgramStateManager] Atomically transitioned {program.id[:8]} state {old_state} -> {new_state}"
            )
