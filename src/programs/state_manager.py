from datetime import datetime, timezone
from loguru import logger
import asyncio
import weakref
from typing import Optional, List, Dict

from src.programs.program import Program
from src.programs.stages.base import ProgramStageResult, StageState
from src.database.program_storage import ProgramStorage, RedisProgramStorage
from src.programs.program_state import ProgramState

class ProgramStateManager:
    """
    Centralized manager for all stage state transitions and persistence for a Program.
    Handles marking stages as running, updating results, and saving to Redis.
    """

    def __init__(self, storage: ProgramStorage):
        """Create a ProgramStateManager."""
        self.storage = storage
        # Use WeakValueDictionary to automatically clean up locks when programs are garbage collected
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_cleanup_counter = 0
        self._max_locks_before_cleanup = 1000  # Cleanup when we hit this many locks

    def _get_lock(self, program_id: str) -> asyncio.Lock:
        """Get a lock for a given program ID with automatic cleanup."""
        if program_id not in self._locks:
            self._locks[program_id] = asyncio.Lock()
            
            # Periodic cleanup to prevent unbounded memory growth
            self._lock_cleanup_counter += 1
            if self._lock_cleanup_counter >= self._max_locks_before_cleanup:
                self._cleanup_unused_locks()
                self._lock_cleanup_counter = 0
                
        return self._locks[program_id]
    
    def _cleanup_unused_locks(self) -> None:
        """Clean up locks for programs that are likely no longer active."""
        # Remove locks that are not currently acquired (indicating the program is idle)
        # This is a heuristic cleanup - locks for truly active programs will be recreated quickly
        idle_locks = [
            program_id for program_id, lock in self._locks.items() 
            if not lock.locked()
        ]
        
        # Keep some recent locks but clean up older ones
        if len(idle_locks) > self._max_locks_before_cleanup // 2:
            # Remove the oldest half of idle locks
            to_remove = idle_locks[:len(idle_locks) // 2]
            for program_id in to_remove:
                self._locks.pop(program_id, None)
            
            logger.debug(f"[StateManager] Cleaned up {len(to_remove)} idle locks, {len(self._locks)} remaining")

    async def mark_stage_running(
        self,
        program: Program,
        stage_name: str,
        *,
        started_at: Optional[datetime] = None,
    ) -> None:
        """Mark *stage_name* as RUNNING and persist.

        Accepts an optional *started_at* so callers can reuse a pre-computed
        timestamp (micro-optimisation for large frontiers).
        """
        async with self._get_lock(program.id):
            ts = started_at or datetime.now(timezone.utc)
            program.stage_results[stage_name] = ProgramStageResult(
                status=StageState.RUNNING, started_at=ts
            )
            await self.storage.update(program)

    async def update_stage_result(
        self, program: Program, stage_name: str, result: ProgramStageResult
    ) -> None:
        """
        Update the result for a stage, update status, and persist.
        """
        async with self._get_lock(program.id):
            program.stage_results[stage_name] = result
            await self.storage.update(program)

    async def set_program_state(self, program: Program, new_state: ProgramState) -> None:
        """Update *program* to *new_state* and persist (publishes event)."""
        async with self._get_lock(program.id):
            if program.state == new_state:
                return  # no-op
            program.state = new_state
            await self.storage.update(program)

            # Publish event for downstream listeners (e.g., scheduler, evolution)
            await self.storage.publish_status_event(new_state.value, program.id)