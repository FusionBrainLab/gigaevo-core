"""Archive storage abstraction for MAP-Elites islands.

Phase-1: purely an interface wrapper so we can inject a storage object
without touching the existing `MapElitesIsland` implementation.  The
actual Redis calls are still made directly by the islands; we merely
hand them the underlying redis client via `_conn()`.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from redis import asyncio as aioredis

from src.database.program_storage import (
    ProgramStorage,
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from src.programs.program import Program
from src.programs.state_manager import ProgramStateManager
from src.programs.program_state import ProgramState
from src.utils.json import dumps, loads


class ArchiveStorage(ABC):
    """Abstract persistence interface for island archives."""

    @abstractmethod
    async def get_elite(self, key: str) -> Optional[Program]: ...

    @abstractmethod
    async def set_elite_if_better(
        self,
        key: str,
        new_program: Program,
        is_better: Callable[[Program, Optional[Program]], bool],
    ) -> bool: ...

    @abstractmethod
    async def mget_elites(self, keys: List[str]) -> List[Optional[Program]]: ...

    @abstractmethod
    async def remove_elite(self, key: str, program: Program) -> None: ...

    async def iter_all_elites(self) -> List[Program]:
        """Return a list of all elite programs currently stored."""
        raise NotImplementedError


class RedisArchiveStorage(ArchiveStorage):
    """Redis-backed archive storage that stores only program IDs."""

    def __init__(self, program_storage: RedisProgramStorage, key_prefix: str = "metaevolve"):
        self._program_storage = program_storage
        self.key_prefix = key_prefix
        self._state_manager = ProgramStateManager(program_storage)

    def _cell_key(self, cell_hash: str) -> str:
        return f"{self.key_prefix}:archive:{cell_hash}"

    async def _get_program_by_id(self, program_id: Optional[str]) -> Optional[Program]:
        if not program_id:
            return None
        try:
            return await self._program_storage.get(program_id)
        except Exception as exc:
            logger.error(f"[RedisArchiveStorage] Failed to fetch program ID: {exc}")
            return None

    async def get_elite(self, key: str) -> Optional[Program]:
        async def _inner(redis):
            return await self._get_program_by_id(await redis.get(self._cell_key(key)))
        return await self._program_storage._execute("archive_get", _inner)

    async def set_elite_if_better(
        self,
        key: str,
        new_program: Program,
        is_better: Callable[[Program, Optional[Program]], bool],
    ) -> bool:
        async def _tx(redis):
            cell_key = self._cell_key(key)

            if not hasattr(redis, "watch"):
                current = await self._get_program_by_id(await redis.get(cell_key))
                if current and not is_better(new_program, current):
                    return False
                await redis.set(cell_key, new_program.id)
                return True

            while True:
                try:
                    await redis.watch(cell_key)
                    current = await self._get_program_by_id(await redis.get(cell_key))
                    if current and not is_better(new_program, current):
                        await redis.unwatch()
                        return False

                    pipe = redis.pipeline(transaction=True)
                    pipe.set(cell_key, new_program.id)
                    await pipe.execute()
                    return True
                except aioredis.WatchError:
                    continue  # retry

        return await self._program_storage._execute("archive_cas", _tx)

    async def mget_elites(self, keys: List[str]) -> List[Optional[Program]]:
        async def _inner(redis):
            pipe = redis.pipeline()
            for k in keys:
                pipe.get(self._cell_key(k))
            raw_ids = await pipe.execute()
            return await self._program_storage.mget([rid for rid in raw_ids if rid])
        return await self._program_storage._execute("archive_mget", _inner)

    async def remove_elite(self, key: str, program: Program) -> None:
        async def _inner(redis):
            await redis.delete(self._cell_key(key))
        await self._program_storage._execute("archive_del", _inner)

        try:
            await self._state_manager.set_program_state(program, ProgramState.DISCARDED)
        except Exception as exc:
            logger.warning(f"[RedisArchiveStorage] Could not mark {program.id} discarded: {exc}")

    async def remove_elite_by_id(self, program_id: str) -> None:
        """Remove an elite from the archive by program ID (searches all cells)."""
        # Find the cell key containing this program ID
        pattern = f"{self.key_prefix}:archive:*"
        async def _inner(redis):
            cursor = "0"
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
                if keys:
                    pipe = redis.pipeline()
                    for k in keys:
                        pipe.get(k)
                    raw_ids = await pipe.execute()
                    for k, rid in zip(keys, raw_ids):
                        if rid and rid.decode() == program_id:
                            await redis.delete(k)
                            return True
                if cursor in ("0", 0):
                    break
            return False
        removed = await self._program_storage._execute("archive_remove_by_id", _inner)
        if not removed:
            logger.warning(f"[RedisArchiveStorage] Program ID {program_id} not found in archive for removal.")

    async def iter_all_elites(self) -> List[Program]:
        """Return a list of all elite programs currently stored with optimized Redis usage and error recovery."""
        pattern = f"{self.key_prefix}:archive:*"

        async def _scan(redis):
            cursor = "0"
            program_ids: List[str] = []

            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
                if keys:
                    # Use a single pipeline for batch operations
                    pipe = redis.pipeline()
                    for k in keys:
                        pipe.get(k)
                    raw_ids = await pipe.execute()
                    program_ids.extend(rid for rid in raw_ids if rid)
                if cursor in ("0", 0):
                    break

            # Batch fetch programs efficiently
            if not program_ids:
                return []
            
            return await self._program_storage.mget(program_ids)

        return await self._program_storage._execute("archive_iter_all", _scan)
