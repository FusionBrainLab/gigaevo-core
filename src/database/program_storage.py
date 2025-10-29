from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from src.programs.program import Program


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
    async def transition_status(
        self, program_id: str, old: str | None, new: str
    ) -> None: ...

    async def wait_for_activity(self, timeout: float) -> None:
        """
        Block up to `timeout` seconds until storage observes activity (e.g., new
        program or status change). Default implementation just sleeps.
        Storages with push/notify capability should override.
        """
        await asyncio.sleep(timeout)
