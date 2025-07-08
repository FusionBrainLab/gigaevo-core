"""Program-level storage abstraction (interface only).

Concrete backends are implemented in separate modules (e.g.
:pyfile:`redis_program_storage.py`).  This file purposely keeps **only** the
abstract :class:`ProgramStorage` base-class and re-exports the default Redis
backend for backward-compatibility so that existing imports continue to work::

    from src.database.program_storage import RedisProgramStorage

No behavioural logic lives here – the goal is to minimise import time and
circular dependencies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.programs.program import Program
from src.utils import json as _json_utils

# Publicly re-export the selected JSON backend so that legacy code using
# ``from src.database.program_storage import json`` keeps working.
json = _json_utils.json  # type: ignore

__all__ = [
    "ProgramStorage",
    "RedisProgramStorageConfig",
    "RedisProgramStorage",
    "json",
]


class ProgramStorage(ABC):
    """Abstract interface for persisting :class:`Program` objects."""

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    @abstractmethod
    async def add(self, program: Program) -> None: ...

    @abstractmethod
    async def update(self, program: Program) -> None: ...

    @abstractmethod
    async def get(self, program_id: str) -> Optional[Program]: ...

    @abstractmethod
    async def exists(self, program_id: str) -> bool: ...

    # ------------------------------------------------------------------
    # Event / status helpers
    # ------------------------------------------------------------------
    @abstractmethod
    async def publish_status_event(
        self,
        status: str,
        program_id: str,
        extra: Optional[dict] | None = None,
    ) -> None: ...

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    @abstractmethod
    async def get_all(self) -> List[Program]: ...

    @abstractmethod
    async def get_all_by_status(self, status: str) -> List[Program]: ...


# ---------------------------------------------------------------------------
# Compatibility re-exports (default Redis backend)
# ---------------------------------------------------------------------------

from src.database.redis_program_storage import (  # noqa: E402  (import after ProgramStorage definition)
    RedisProgramStorage,
    RedisProgramStorageConfig,
) 