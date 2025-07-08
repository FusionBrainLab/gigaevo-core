# -*- coding: utf-8 -*-
"""In-memory ArchiveStorage used by the test-suite when running without Redis.
Keeps a simple dictionary mapping *cell_key* → Program.

Because tests are single-threaded we only need a single asyncio.Lock for
atomicity.  This implementation is obviously **not** suitable for a multi-
process production environment – use RedisArchiveStorage there.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional

from src.evolution.storage.archive_storage import ArchiveStorage
from src.programs.program import Program


class MemoryArchiveStorage(ArchiveStorage):
    """Pure python ArchiveStorage – no external dependencies."""

    def __init__(self) -> None:
        self._store: Dict[str, Program] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # ArchiveStorage interface
    # ------------------------------------------------------------------

    async def get_elite(self, key: str) -> Optional[Program]:
        async with self._lock:
            prog = self._store.get(key)
            return Program.from_dict(prog.to_dict()) if prog else None  # return copy

    async def set_elite_if_better(
        self,
        key: str,
        new_program: Program,
        is_better: Callable[[Program, Optional[Program]], bool],
    ) -> bool:
        async with self._lock:
            current = self._store.get(key)
            if current is None or is_better(new_program, current):
                # store *copy* so callers cannot mutate internal state
                self._store[key] = Program.from_dict(new_program.to_dict())
                return True
            return False

    async def mget_elites(self, keys: List[str]) -> List[Optional[Program]]:
        async with self._lock:
            result: List[Optional[Program]] = []
            for k in keys:
                p = self._store.get(k)
                result.append(Program.from_dict(p.to_dict()) if p else None)
            return result

    async def remove_elite(self, key: str, program: Program) -> None:  # noqa: D401
        async with self._lock:
            self._store.pop(key, None)

    # ------------------------------------------------------------------
    # Convenience for Map-Elites islands
    # ------------------------------------------------------------------

    async def iter_all_elites(self, island_prefix: str | None = None) -> List[Program]:  # pragma: no cover
        async with self._lock:
            if island_prefix is None:
                return [Program.from_dict(p.to_dict()) for p in self._store.values()]

            # Filter by key prefix (same semantics as Redis implementation)
            matching: List[Program] = []
            for key, prog in self._store.items():
                if key.startswith(island_prefix):
                    matching.append(Program.from_dict(prog.to_dict()))
            return matching 