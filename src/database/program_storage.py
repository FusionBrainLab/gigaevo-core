from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.programs.program import Program


class ProgramStorage(ABC):
    """Abstract interface for persisting :class:`Program` objects."""

    @abstractmethod
    async def add(self, program: Program) -> None: ...

    @abstractmethod
    async def update(self, program: Program) -> None: ...

    @abstractmethod
    async def get(self, program_id: str) -> Optional[Program]: ...

    @abstractmethod
    async def exists(self, program_id: str) -> bool: ...

    @abstractmethod
    async def publish_status_event(
        self,
        status: str,
        program_id: str,
        extra: Optional[dict] | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_all(self) -> List[Program]: ...

    @abstractmethod
    async def get_all_by_status(self, status: str) -> List[Program]: ...
