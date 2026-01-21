from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Any

from gigaevo.programs.program import Program


class ProgramStorage(ABC):
    """Abstract interface for persisting :class:`Program` objects."""

    @abstractmethod
    async def add(self, program: Program) -> None: ...

    @abstractmethod
    async def update(self, program: Program) -> None: ...

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
    async def get_all(self) -> list[Program]: ...

    @abstractmethod
    async def get_all_by_status(self, status: str) -> list[Program]: ...

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

    async def wait_for_activity(self, timeout: float) -> None:
        """
        Block up to `timeout` seconds until storage observes activity (e.g., new
        program or status change). Default implementation just sleeps.
        Storages with push/notify capability should override.
        """
        await asyncio.sleep(timeout)

    @abstractmethod
    async def close(self) -> None: ...
