"""Archive storage abstraction for MAP-Elites islands.

Phase-1: purely an interface wrapper so we can inject a storage object
without touching the existing `MapElitesIsland` implementation.  The
actual Redis calls are still made directly by the islands; we merely
hand them the underlying redis client via `_conn()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple
import random

from loguru import logger
from redis import asyncio as aioredis

from src.database.redis_program_storage import (
    RedisProgramStorage,
)
from src.programs.program import Program
from src.programs.state_manager import ProgramStateManager


class ArchiveStorage(ABC):
    """Abstract persistence interface for island archives."""

    @abstractmethod
    async def get_elite(self, cell: Tuple[int, ...]) -> Optional[Program]: ...

    @abstractmethod
    async def add_elite(self, cell: Tuple[int, ...], program: Program, is_better: Callable[[Program, Optional[Program]], bool]) -> bool: ...

    @abstractmethod
    async def remove_elite(self, cell: Tuple[int, ...]) -> bool: ...

    @abstractmethod
    async def get_all_elites(self) -> List[Program]: ...

    @abstractmethod
    async def remove_elite_by_id(self, program_id: str) -> bool: ...

    @abstractmethod
    async def clear_all_elites(self) -> int: ...


class RedisArchiveStorage(ArchiveStorage):
    """
    Redis-backed archive storage for MAP-Elites islands.
    Stores only program IDs, with all program data in RedisProgramStorage.
    """
    def __init__(self, program_storage: RedisProgramStorage, key_prefix: str = "metaevolve"):
        self._program_storage = program_storage
        self.key_prefix = key_prefix
        self._state_manager = ProgramStateManager(program_storage)

    def cell_key_from_tuple(self, cell: Tuple[int, ...]) -> str:
        """Convert a cell coordinate tuple to a Redis cell key string (e.g., '0,1,2')."""
        return ','.join(map(str, cell))

    def _redis_cell_key(self, cell: Tuple[int, ...]) -> str:
        """Get the full Redis key for a cell tuple."""
        return f"{self.key_prefix}:archive:{self.cell_key_from_tuple(cell)}"

    async def add_elite(self, cell: Tuple[int, ...], program: Program, is_better: Callable[[Program, Optional[Program]], bool]) -> bool:
        """
        Add or replace the elite in the given cell if the new program is better.
        Returns True if the program was added/replaced, False otherwise.
        """
        # Check if program exists in storage before adding as elite
        try:
            stored_program = await self._program_storage.get(program.id)
            if stored_program is None:
                logger.error(f"Cannot add program {program.id} as elite - program not found in storage")
                return False
        except Exception as e:
            logger.error(f"Failed to check program existence for {program.id}: {e}")
            return False
            
        redis_key = self._redis_cell_key(cell)
        async def _tx(redis):
            if not hasattr(redis, "watch"):
                current_id = await redis.get(redis_key)
                current = await self._get_program_by_id(current_id)
                if current and not is_better(program, current):
                    return False
                await redis.set(redis_key, program.id)
                logger.debug(f"Added/replaced elite {program.id} in cell {cell}")
                return True
            while True:
                try:
                    await redis.watch(redis_key)
                    current_id = await redis.get(redis_key)
                    current = await self._get_program_by_id(current_id)
                    if current and not is_better(program, current):
                        await redis.unwatch()
                        return False
                    pipe = redis.pipeline(transaction=True)
                    pipe.set(redis_key, program.id)
                    await pipe.execute()
                    logger.debug(f"Added/replaced elite {program.id} in cell {cell}")
                    return True
                except aioredis.WatchError:
                    continue  # retry
        return await self._program_storage._execute("archive_add_elite", _tx)

    async def get_elite(self, cell: Tuple[int, ...]) -> Optional[Program]:
        """
        Retrieve the elite program for the given cell, or None if empty.
        """
        redis_key = self._redis_cell_key(cell)
        async def _inner(redis):
            program_id = await redis.get(redis_key)
            return await self._get_program_by_id(program_id)
        return await self._program_storage._execute("archive_get_elite", _inner)

    async def remove_elite(self, cell: Tuple[int, ...]) -> bool:
        """
        Remove the elite from the given cell. Returns True if removed, False if not found.
        """
        redis_key = self._redis_cell_key(cell)
        async def _inner(redis):
            result = await redis.delete(redis_key)
            return result > 0
        removed = await self._program_storage._execute("archive_remove_elite", _inner)
        if removed:
            logger.debug(f"Removed elite from cell {cell}")
        else:
            logger.debug(f"No elite found to remove in cell {cell}")
        return removed

    async def get_all_elites(self) -> List[Program]:
        """
        Retrieve all elite programs in the archive.
        """
        pattern = f"{self.key_prefix}:archive:*"
        async def _scan(redis):
            cursor = "0"
            program_ids: List[str] = []
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
                if keys:
                    pipe = redis.pipeline()
                    for k in keys:
                        pipe.get(k)
                    raw_ids = await pipe.execute()
                    program_ids.extend(rid for rid in raw_ids if rid)
                if cursor in ("0", 0):
                    break
            if not program_ids:
                return []
            
            random.shuffle(program_ids)
            
            return await self._program_storage.mget(program_ids)
        elites = await self._program_storage._execute("archive_get_all_elites", _scan)
        logger.debug(f"Retrieved {len(elites)} elites from archive.")
        return elites

    async def remove_elite_by_id(self, program_id: str) -> bool:
        """
        Remove an elite from the archive by its program ID. Returns True if removed, False if not found.
        """
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
                        if rid and rid == program_id:
                            await redis.delete(k)
                            logger.debug(f"Removed elite with program_id {program_id} from archive.")
                            return True
                if cursor in ("0", 0):
                    break
            return False
        removed = await self._program_storage._execute("archive_remove_by_id", _inner)
        if not removed:
            logger.warning(f"Program ID {program_id} not found in archive for removal.")
        return removed

    async def clear_all_elites(self) -> int:
        """
        Remove all elites from the archive. Returns the number of removed entries.
        """
        pattern = f"{self.key_prefix}:archive:*"
        async def _inner(redis):
            cursor = "0"
            total = 0
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
                if keys:
                    await redis.delete(*keys)
                    total += len(keys)
                if cursor in ("0", 0):
                    break
            return total
        total = await self._program_storage._execute("archive_clear_all", _inner)
        logger.info(f"Cleared {total} elites from the archive.")
        return total

    async def _get_program_by_id(self, program_id: Optional[str]) -> Optional[Program]:
        if not program_id:
            return None
        try:
            return await self._program_storage.get(program_id)
        except Exception as exc:
            logger.error(f"[RedisArchiveStorage] Failed to fetch program ID: {exc}")
            return None
