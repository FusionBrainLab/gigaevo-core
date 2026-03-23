from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Any

from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState, validate_transition


class PopulationSnapshot:
    """Epoch-based cache for ``storage.get_all()``.

    Shared across all collector instances via ``storage.snapshot``.
    Call :meth:`bump` at phase boundaries to invalidate; collectors that
    see a stale epoch will re-fetch exactly once (others piggyback).

    Single-slot cache keyed on ``(epoch, exclude)``. Works correctly when all
    callers within an epoch use the same exclude value (which is true in practice:
    EvolutionaryStatisticsCollector always uses exclude={"stage_results"}, all
    other callers use exclude=None). If a new caller with a different exclude
    value is added, the cache will thrash with no benefits -- a latent hazard
    that would require a dict-based multi-slot cache to fully address.
    """

    __slots__ = ("_epoch", "_cached_epoch", "_cached_exclude", "_cached", "_lock")

    def __init__(self) -> None:
        self._epoch: int = 0
        self._cached_epoch: int = -1
        self._cached_exclude: frozenset[str] | None = None
        self._cached: list[Program] = []
        self._lock = asyncio.Lock()

    def bump(self) -> None:
        """Increment the epoch, invalidating the cached snapshot."""
        self._epoch += 1

    async def get_all(
        self,
        storage: ProgramStorage,
        *,
        exclude: frozenset[str] | None = None,
    ) -> list[Program]:
        """Return cached programs if epoch+exclude matches, else fetch + cache."""
        if self._cached_epoch == self._epoch and self._cached_exclude == exclude:
            return self._cached
        async with self._lock:
            if self._cached_epoch == self._epoch and self._cached_exclude == exclude:
                return self._cached
            programs = await storage.get_all(exclude=exclude)
            self._cached = programs
            self._cached_exclude = exclude
            self._cached_epoch = self._epoch
            return programs


class ProgramStorage(ABC):
    """Abstract interface for persisting :class:`Program` objects."""

    def __init__(self) -> None:
        self.snapshot = PopulationSnapshot()

    @abstractmethod
    async def add(self, program: Program) -> None: ...

    @abstractmethod
    async def update(self, program: Program) -> None: ...

    async def write_exclusive(self, program: Program) -> None:
        """Fast write without WATCH/MERGE. Default falls back to update().

        Safe only in exclusive-ownership contexts (during DAG execution).
        Implementations may override with a 2 RT path (INCR + SET).
        """
        await self.update(program)

    @abstractmethod
    async def get(self, program_id: str) -> Program | None: ...

    @abstractmethod
    async def mget(self, program_ids: list[str]) -> list[Program]: ...

    @abstractmethod
    async def exists(self, program_id: str) -> bool: ...

    @abstractmethod
    async def publish_status_event(
        self,
        status: str,
        program_id: str,
        extra: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_all(self, *, exclude: frozenset[str] | None = None) -> list[Program]:
        """Return all programs.

        Args:
            exclude: Optional set of field names to skip during deserialization.
                Excluded fields get their Pydantic defaults. Implementations
                that don't support projection may ignore this parameter.
        """
        ...

    @abstractmethod
    async def get_all_by_status(
        self, status: str, *, exclude: frozenset[str] | None = None
    ) -> list[Program]: ...

    @abstractmethod
    async def get_ids_by_status(self, status: str) -> list[str]:
        """Return IDs of programs with the given status (no full fetch)."""
        ...

    @abstractmethod
    async def count_by_status(self, status: str) -> int:
        """Return count of programs with the given status (without fetching data)."""
        ...

    @abstractmethod
    async def get_all_program_ids(self) -> list[str]: ...

    @abstractmethod
    async def transition_status(
        self, program_id: str, old: str | None, new: str
    ) -> None: ...

    @abstractmethod
    async def atomic_state_transition(
        self, program: Program, old_state: str | None, new_state: str
    ) -> None:
        """
        Atomically update program state AND status set membership in a single transaction.
        This ensures program.state and status sets never get out of sync.

        Args:
            program: Program object with updated state
            old_state: Previous state value (for removing from old set)
            new_state: New state value (for adding to new set)

        Raises:
            StorageError: If atomic operation fails
        """
        ...

    @abstractmethod
    async def acquire_instance_lock(self) -> bool:
        """
        Acquire an exclusive lock on this storage prefix to prevent multiple instances.

        Returns:
            True if lock was acquired, False if another instance holds the lock

        Raises:
            StorageError: If lock acquisition fails or another instance is detected
        """
        ...

    @abstractmethod
    async def release_instance_lock(self) -> None:
        """
        Release the instance lock acquired by acquire_instance_lock().
        Should be called during shutdown.
        """
        ...

    @abstractmethod
    async def renew_instance_lock(self) -> bool:
        """
        Renew the instance lock to prevent expiry.
        Should be called periodically while instance is running.

        Returns:
            True if renewal succeeded, False if lock was lost
        """
        ...

    async def fast_state_transition(
        self, program: Program, old_state: str, new_state: str
    ) -> None:
        """Fast state transition: 2 RT (INCR + pipeline) instead of ~5 RT.

        Safe only when the caller holds exclusive single-process ownership
        (e.g., asyncio.Lock in ProgramStateManager). Does NOT provide cross-process
        safety — assumes each program is processed by exactly one engine instance.
        Default falls back to atomic_state_transition.
        """
        await self.atomic_state_transition(program, old_state, new_state)

    async def batch_transition_state(
        self,
        programs: list[Program],
        old_state: str,
        new_state: str,
    ) -> int:
        """Batch-transition programs between states.

        Default implementation falls back to individual transitions.
        Subclasses may override with pipelined operations.
        Returns the number of programs transitioned.
        """
        old_enum = ProgramState(old_state)
        new_enum = ProgramState(new_state)
        count = 0
        for prog in programs:
            validate_transition(old_enum, new_enum)
            prog.state = new_enum
            await self.atomic_state_transition(prog, old_state, new_state)
            count += 1
        return count

    async def remove_ids_from_status_set(self, status: str, ids: list[str]) -> None:
        """Remove specific IDs from a status set. No-op by default."""

    async def wait_for_activity(self, timeout: float) -> None:
        """
        Block up to `timeout` seconds until storage observes activity (e.g., new
        program or status change). Default implementation just sleeps.
        Storages with push/notify capability should override.
        """
        await asyncio.sleep(timeout)

    async def save_run_state(self, field: str, value: int) -> None:
        """Persist a named integer counter for resume support. No-op by default."""

    async def load_run_state(self, field: str) -> int | None:
        """Load a previously saved integer counter. Returns None if not found."""
        return None

    async def recover_stranded_programs(self) -> int:
        """Reset RUNNING programs to QUEUED after a crash/kill (for resume).

        Returns the number of programs recovered.
        """
        return 0

    @abstractmethod
    async def close(self) -> None: ...
