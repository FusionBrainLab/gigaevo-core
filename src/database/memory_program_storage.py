# -*- coding: utf-8 -*-
"""In-memory ProgramStorage used for unit/integration tests when we don't want
an external Redis instance.  The public interface mirrors
:class:`src.database.program_storage.ProgramStorage`.

NOT thread-safe beyond the GIL – only use in single-process test runs.
"""
from __future__ import annotations

import asyncio
import json
import heapq
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.database.program_storage import ProgramStorage
from src.programs.program import Program
from src.exceptions import StorageError


class MemoryProgramStorage(ProgramStorage):
    """Simple dict-backed storage."""

    def __init__(self) -> None:
        # id -> Program
        self._data: Dict[str, Program] = {}
        # Set of code hashes for duplicate detection
        self._code_hashes: set[str] = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    async def add(self, program: Program) -> None:  # noqa: D401
        async with self._lock:
            if program.id in self._data:
                raise StorageError(f"Program {program.id} already exists")
            self._data[program.id] = Program.from_dict(program.to_dict())

    async def update(self, program: Program) -> None:  # noqa: D401
        async with self._lock:
            self._data[program.id] = Program.from_dict(program.to_dict())

    async def get(self, program_id: str) -> Optional[Program]:  # noqa: D401
        async with self._lock:
            p = self._data.get(program_id)
            return Program.from_dict(p.to_dict()) if p else None

    async def exists(self, program_id: str) -> bool:  # noqa: D401
        async with self._lock:
            return program_id in self._data

    async def remove(self, program_id: str) -> None:  # noqa: D401
        async with self._lock:
            self._data.pop(program_id, None)

    # --- batch helpers -------------------------------------------------

    async def get_all(self, filter_fn: Optional[Callable[[Program], bool]] = None) -> List[Program]:  # type: ignore[override]
        async with self._lock:
            progs = [Program.from_dict(p.to_dict()) for p in self._data.values()]
            if filter_fn:
                progs = [p for p in progs if filter_fn(p)]
            return progs



    async def get_all_by_status(self, status: str) -> List[Program]:
        """Get all programs with a specific status."""
        async with self._lock:
            return [
                Program.from_dict(p.to_dict()) 
                for p in self._data.values() 
                if p.state == status
            ]

    async def get_top(self, score_fn: Callable[[Program], float], k: int = 10, only_complete: bool = True) -> List[Program]:  # noqa: D401
        candidates = await self.get_all_complete() if only_complete else await self.get_all()
        return heapq.nlargest(k, candidates, key=score_fn)

    # --- code hash helpers --------------------------------------------

    async def code_hash_exists(self, code_hash: str) -> bool:
        return code_hash in self._code_hashes

    async def add_code_hash(self, code_hash: str) -> None:
        self._code_hashes.add(code_hash)

    # Bulk helper – trivial in-memory implementation
    async def filter_new_code_hashes(self, code_hashes):  # type: ignore[override]
        unseen = [h for h in code_hashes if h not in self._code_hashes]
        self._code_hashes.update(unseen)
        return unseen

    # ------------------------------------------------------------------
    # Event stream helpers – no-op for memory backend
    # ------------------------------------------------------------------

    async def publish_status_event(self, status: str, program_id: str, extra: Optional[Dict[str, Any]] = None) -> None:  # noqa: D401
        # Record timestamp for debugging
        logger.debug(f"[MemoryProgramStorage] status={status} id={program_id} extra={extra}")

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def close(self):  # pragma: no cover
        # Nothing to close
        pass

    # ------------------------------------------------------------------
    # Test helpers – minimal subset of Redis API for test compatibility
    # ------------------------------------------------------------------

    class _DummyRedis:
        """Very small subset of the aioredis API needed by tests."""

        def __init__(self, outer: "MemoryProgramStorage") -> None:
            self._outer = outer

        # Key/value commands ------------------------------------------------
        async def flushdb(self):  # noqa: D401
            self._outer._data.clear()

        async def close(self):  # noqa: D401
            # No-op for in-memory
            pass

        # For compatibility where tests might pipeline etc.
        def __getattr__(self, name):  # noqa: D401
            raise AttributeError(f"DummyRedis has no attr {name}")

    async def _conn(self):  # type: ignore[override]
        """Return dummy redis object for tests that call `await storage._conn()`."""
        return self._DummyRedis(self)

 