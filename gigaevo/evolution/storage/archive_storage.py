from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final

from loguru import logger
from redis.exceptions import WatchError

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.programs.program import Program

CellDescriptor = tuple[int, ...]


# Retry budget for the optimistic compare-and-swap in
# :meth:`RedisArchiveStorage.add_elite`. Each retry is one Redis round
# trip; exhaustion surfaces as ``False`` with a warning so a contended
# cell cannot hang a caller indefinitely.
_WATCH_MAX_ATTEMPTS: Final[int] = 50


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
        is_better: Callable[[Program, Program], bool],
    ) -> bool: ...

    @abstractmethod
    async def remove_elite(self, cell: CellDescriptor) -> bool: ...

    @abstractmethod
    async def get_all_elites(self) -> list[str]: ...

    # Returns unique program IDs that are currently elites in any cell.

    @abstractmethod
    async def remove_elite_by_id(self, program_id: str) -> bool: ...

    @abstractmethod
    async def bulk_remove_elites_by_id(self, program_ids: list[str]) -> int:
        """Remove multiple elites atomically. Returns number actually removed."""
        ...

    @abstractmethod
    async def clear_all_elites(self) -> int: ...

    # Returns number of cells cleared.

    @abstractmethod
    async def bulk_add_elites(
        self,
        placements: list[tuple[CellDescriptor, Program]],
        is_better: Callable[[Program, Program], bool],
    ) -> int: ...

    # Adds multiple elites at once (e.g., during re-indexing). Returns number of successful adds.

    @abstractmethod
    async def size(self) -> int: ...

    # Returns number of occupied cells.


class RedisArchiveStorage(ArchiveStorage):
    """Redis-backed archive with bounded optimistic locking + reverse index.

    Data structures:
      - ``{prefix}:archive`` (hash): cell -> program_id
      - ``{prefix}:archive:reverse`` (hash): program_id -> cell (1:1 mapping)

    Each program can be elite in at most one cell at a time; the reverse
    index keeps that invariant cheap to enforce on a swap.

    Every read consults Redis (one HGET / HVALS / HLEN per call); there
    is no in-memory mirror of the archive hash. ``add_elite`` runs
    optimistic WATCH/MULTI/EXEC with the caller-supplied ``is_better``
    predicate, capped at :data:`_WATCH_MAX_ATTEMPTS` retries.
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

    async def get_elite(self, cell: CellDescriptor) -> Program | None:
        field = self._field(cell)
        pid = await self._hget(field)
        if not pid:
            return None
        return await self._storage.get(pid)

    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program], bool],
    ) -> bool:
        """Add elite via bounded WATCH/MULTI/EXEC.

        Reads the current occupant from Redis, compares the candidate
        against it via ``is_better``, and either swaps or rejects. The
        retry loop terminates after :data:`_WATCH_MAX_ATTEMPTS`
        observations of ``WatchError``; on exhaustion the call returns
        ``False`` with a warning.
        """
        field = self._field(cell)

        # The candidate must exist in the program storage before being
        # installed as the elite.
        if not await self._storage.exists(program.id):
            logger.debug("[Archive] add ignored: program {} not in storage", program.id)
            return False

        async def _op(r):
            attempts = 0
            while True:
                attempts += 1
                if attempts > _WATCH_MAX_ATTEMPTS:
                    logger.warning(
                        "[Archive] add_elite gave up on cell {} after {} WATCH retries",
                        field,
                        _WATCH_MAX_ATTEMPTS,
                    )
                    return False
                try:
                    async with r.pipeline() as pipe:
                        await pipe.watch(self._hash_key)
                        redis_id = await pipe.hget(self._hash_key, field)

                        if redis_id:
                            redis_prog = await self._storage.get(redis_id)
                            if redis_prog and not is_better(program, redis_prog):
                                await pipe.unwatch()
                                return False

                        pipe.multi()
                        pipe.hset(self._hash_key, field, program.id)
                        if redis_id and redis_id != program.id:
                            pipe.hdel(self._reverse_key, redis_id)
                        pipe.hset(self._reverse_key, program.id, field)
                        await pipe.execute()
                        return True

                except WatchError:
                    # Another writer updated the archive hash; loop and
                    # re-read the current occupant.
                    continue

        ok = await self._storage.with_redis("archive:add_elite", _op)
        if ok:
            logger.debug("[Archive] cell {} -> {}", field, program.id)
        return bool(ok)

    async def remove_elite(self, cell: CellDescriptor) -> bool:
        """Remove elite from cell and update reverse index."""
        field = self._field(cell)

        async def _op(r):
            current_id = await r.hget(self._hash_key, field)
            if not current_id:
                return False

            pipe = r.pipeline(transaction=False)
            pipe.hdel(self._hash_key, field)
            pipe.hdel(self._reverse_key, current_id)
            await pipe.execute()
            return True

        removed = await self._storage.with_redis("archive:remove_elite", _op)
        if removed:
            logger.debug("[Archive] removed cell {}", field)
        return bool(removed)

    async def get_all_elites(self) -> list[str]:
        """Return all elite program IDs (already unique due to 1:1 mapping)."""
        vals = await self._hvals()
        return sorted(vals)

    async def size(self) -> int:
        return await self._hlen()

    async def remove_elite_by_id(self, program_id: str) -> bool:
        """Remove program using reverse index (O(1) lookup)."""

        async def _op(r):
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

    async def bulk_remove_elites_by_id(self, program_ids: list[str]) -> int:
        """Remove multiple elites using two Redis pipelines. Returns number actually removed."""
        if not program_ids:
            return 0

        async def _op(r):
            pipe = r.pipeline(transaction=False)
            for pid in program_ids:
                pipe.hget(self._reverse_key, pid)
            cells = await pipe.execute()

            pipe2 = r.pipeline(transaction=False)
            removed = 0
            for pid, cell in zip(program_ids, cells):
                if cell:
                    pipe2.hdel(self._hash_key, cell)
                    pipe2.hdel(self._reverse_key, pid)
                    removed += 1
            if removed:
                await pipe2.execute()
            return removed

        count = await self._storage.with_redis("archive:bulk_remove_elites_by_id", _op)
        if count:
            logger.debug("[Archive] bulk removed {} ids", count)
        return int(count)

    async def clear_all_elites(self) -> int:
        """Clear all elites and reverse index. Returns cells cleared."""
        count = await self._hlen()
        if count == 0:
            return 0

        async def _op(r):
            pipe = r.pipeline(transaction=False)
            pipe.delete(self._hash_key)
            pipe.delete(self._reverse_key)
            await pipe.execute()

        await self._storage.with_redis("archive:clear_all", _op)
        logger.debug("[Archive] cleared {} elites", count)
        return count

    async def bulk_add_elites(
        self,
        placements: list[tuple[CellDescriptor, Program]],
        is_better: Callable[[Program, Program], bool],
    ) -> int:
        """Sequential add_elite over the placements list.

        Iterates one ``add_elite`` per placement. A future optimisation
        could group by cell and pick the best per cell first; the
        simple sequential form is correct and adequate for the
        re-indexing path that calls it.
        """
        if not placements:
            return 0

        added_count = 0
        for cell, program in placements:
            if await self.add_elite(cell, program, is_better):
                added_count += 1

        return added_count
