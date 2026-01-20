from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger
from redis.exceptions import WatchError

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.programs.program import Program

CellDescriptor = tuple[int, ...]


# ------------------------------- Interface -------------------------------


class ArchiveStorage(ABC):
    """Elite archive keyed by behavior-space cells."""

    @abstractmethod
    async def get_elite(self, cell: CellDescriptor) -> Program | None: ...

    @abstractmethod
    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program | None], bool],
    ) -> bool: ...

    @abstractmethod
    async def remove_elite(self, cell: CellDescriptor) -> bool: ...

    @abstractmethod
    async def get_all_elites(self) -> list[str]: ...

    # Returns unique program IDs that are currently elites in any cell.

    @abstractmethod
    async def remove_elite_by_id(self, program_id: str) -> bool: ...

    @abstractmethod
    async def clear_all_elites(self) -> int: ...

    # Returns number of cells cleared.

    @abstractmethod
    async def bulk_add_elites(
        self,
        placements: list[tuple[CellDescriptor, Program]],
        is_better: Callable[[Program, Program | None], bool],
    ) -> int: ...

    # Adds multiple elites at once (e.g., during re-indexing). Returns number of successful adds.

    @abstractmethod
    async def size(self) -> int: ...

    # Returns number of occupied cells.


class RedisArchiveStorage(ArchiveStorage):
    """
    Redis-backed archive with optimistic locking and reverse index.

    Data structures:
      - `prefix:archive` (hash): cell -> program_id
      - `prefix:archive:reverse` (hash): program_id -> cell (1:1 mapping)

    Note: Each program can only be elite in ONE cell at a time.
    """

    def __init__(
        self, program_storage: RedisProgramStorage, key_prefix: str | None = None
    ) -> None:
        self._storage = program_storage
        prefix = key_prefix or program_storage.config.key_prefix
        self._hash_key = f"{prefix}:archive"
        self._reverse_key = f"{prefix}:archive:reverse"

    # -------- small helpers --------

    @staticmethod
    def _field(cell: CellDescriptor) -> str:
        return ",".join(map(str, cell))

    async def _hget(self, field: str) -> str | None:
        async def _op(r):
            return await r.hget(self._hash_key, field)

        return await self._storage.with_redis("archive:hget", _op)

    async def _hvals(self) -> list[str]:
        async def _op(r):
            return await r.hvals(self._hash_key)

        return await self._storage.with_redis("archive:hvals", _op) or []

    async def _hlen(self) -> int:
        async def _op(r):
            return await r.hlen(self._hash_key)

        return await self._storage.with_redis("archive:hlen", _op)

    async def _hgetall(self) -> dict[str, str]:
        async def _op(r):
            return await r.hgetall(self._hash_key)

        return await self._storage.with_redis("archive:hgetall", _op) or {}

    async def get_elite(self, cell: CellDescriptor) -> Program | None:
        pid = await self._hget(self._field(cell))
        return await self._storage.get(pid) if pid else None

    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program | None], bool],
    ) -> bool:
        """Add elite with optimistic locking (WATCH/MULTI/EXEC)."""
        if not await self._storage.exists(program.id):
            logger.debug("[Archive] add ignored: program {} not in storage", program.id)
            return False

        field = self._field(cell)

        async def _op(r):
            while True:
                try:
                    # Watch for concurrent modifications
                    await r.watch(self._hash_key)

                    current_id = await r.hget(self._hash_key, field)
                    if current_id:
                        current_prog = await self._storage.get(current_id)
                        if current_prog and not is_better(program, current_prog):
                            await r.unwatch()
                            return False

                    # Begin atomic transaction
                    pipe = r.pipeline()
                    pipe.multi()

                    # Update main archive: cell -> program_id
                    pipe.hset(self._hash_key, field, program.id)

                    # Update reverse index (1:1 mapping)
                    if current_id and current_id != program.id:
                        # Remove old program's reverse entry
                        pipe.hdel(self._reverse_key, current_id)

                    # Set new program's reverse entry: program_id -> cell
                    pipe.hset(self._reverse_key, program.id, field)

                    await pipe.execute()
                    return True

                except WatchError:
                    # Concurrent modification, retry
                    continue

        ok = await self._storage.with_redis("archive:add_elite", _op)
        if ok:
            logger.debug("[Archive] cell {} -> {}", field, program.id)
        return bool(ok)

    async def remove_elite(self, cell: CellDescriptor) -> bool:
        """Remove elite from cell and update reverse index."""
        field = self._field(cell)

        async def _op(r):
            # Get current program in this cell
            current_id = await r.hget(self._hash_key, field)
            if not current_id:
                return False

            pipe = r.pipeline(transaction=False)
            pipe.hdel(self._hash_key, field)
            pipe.hdel(self._reverse_key, current_id)  # 1:1 mapping, just delete
            await pipe.execute()
            return True

        removed = await self._storage.with_redis("archive:remove_elite", _op)
        if removed:
            logger.debug("[Archive] removed cell {}", field)
        return bool(removed)

    async def get_all_elites(self) -> list[str]:
        """Return all elite program IDs (already unique due to 1:1 mapping)."""
        ids = await self._hvals()
        return sorted(ids)

    async def size(self) -> int:
        return await self._hlen()

    async def remove_elite_by_id(self, program_id: str) -> bool:
        """Remove program using reverse index (O(1) lookup)."""

        async def _op(r):
            # Look up cell from reverse index (1:1 mapping)
            cell = await r.hget(self._reverse_key, program_id)
            if not cell:
                return False

            pipe = r.pipeline(transaction=False)
            pipe.hdel(self._hash_key, cell)
            pipe.hdel(self._reverse_key, program_id)
            await pipe.execute()
            return True

        removed = await self._storage.with_redis("archive:remove_elite_by_id", _op)
        if removed:
            logger.debug("[Archive] removed id {}", program_id)
        return bool(removed)

    async def clear_all_elites(self) -> int:
        """Clear all elites and reverse index."""
        mapping = await self._hgetall()
        count = len(mapping)
        if count == 0:
            return 0

        async def _op(r):
            pipe = r.pipeline(transaction=False)
            pipe.delete(self._hash_key)
            pipe.delete(self._reverse_key)
            await pipe.execute()

        await self._storage.with_redis("archive:clear_all", _op)

        logger.debug(
            "[Archive] cleared {} elites ({} unique ids)",
            count,
            len(set(mapping.values())),
        )
        return count

    async def bulk_add_elites(
        self,
        placements: list[tuple[CellDescriptor, Program]],
        is_better: Callable[[Program, Program | None], bool],
    ) -> int:
        if not placements:
            return 0

        # Note: This naive implementation processes items sequentially.
        # A more optimized version would group by cell and select the best per cell first,
        # but since this runs during re-indexing (rarely), correctness > raw speed for now.

        added_count = 0
        for cell, program in placements:
            if await self.add_elite(cell, program, is_better):
                added_count += 1

        return added_count
