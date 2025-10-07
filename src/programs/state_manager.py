import asyncio
from datetime import datetime, timezone

from src.database.program_storage import ProgramStorage
from src.programs.program import Program
from src.programs.program_state import ProgramState
from src.programs.core_types import ProgramStageResult, StageState


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
        """Mark a stage as RUNNING and persist."""
        async with self._lock_for(program.id):
            ts = started_at or datetime.now(timezone.utc)
            program.stage_results[stage_name] = ProgramStageResult(
                status=StageState.RUNNING,
                started_at=ts,
            )
            await self.storage.update(program)

    async def update_stage_result(
        self,
        program: Program,
        stage_name: str,
        result: ProgramStageResult,
    ) -> None:
        """Set a stage result and persist."""
        async with self._lock_for(program.id):
            program.stage_results[stage_name] = result
            await self.storage.update(program)

    async def set_program_state(self, program: Program, new_state: ProgramState) -> None:
        async with self._lock_for(program.id):
            if program.state == new_state:
                return

            old_state = program.state  # keep a copy
            program.state = new_state
            await self.storage.update(program)
            old = old_state.value if old_state else None
            await self.storage.transition_status(program.id, old, new_state.value)
            await self.storage.publish_status_event(new_state.value, program.id)

