from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

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

    An in-memory write-through cache (``_elite_cache``) mirrors the Redis
    archive hash.  Reads hit the cache first; writes update both cache and
    Redis atomically.  This eliminates 4-5 Redis round-trips per
    ``add_elite`` call for the vast majority of programs that don't improve
    the current cell occupant.  Safe because a single engine instance holds
    an exclusive Redis instance lock per prefix.
    """

    def __init__(
        self, program_storage: RedisProgramStorage, key_prefix: str | None = None
    ) -> None:
        self._storage = program_storage
        prefix = key_prefix or program_storage.config.key_prefix
        self._hash_key = f"{prefix}:archive"
        self._reverse_key = f"{prefix}:archive:reverse"
        # In-memory write-through cache: cell_field -> elite Program
        self._elite_cache: dict[str, Program] = {}
        # Reverse: program_id -> cell_field (for remove_by_id)
        self._elite_reverse: dict[str, str] = {}
        self._cache_loaded: bool = False

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

    async def _ensure_cache(self) -> None:
        """Lazily populate the in-memory elite cache from Redis."""
        if self._cache_loaded:
            return
        mapping = await self._hgetall()  # field -> program_id
        if mapping:
            pids = list(mapping.values())
            programs = await self._storage.mget(pids)
            pid_to_prog = {p.id: p for p in programs}
            for field, pid in mapping.items():
                prog = pid_to_prog.get(pid)
                if prog is not None:
                    self._elite_cache[field] = prog
                    self._elite_reverse[pid] = field
        self._cache_loaded = True

    def _cache_set(self, field: str, program: Program) -> None:
        """Update cache for a cell, evicting old occupant if different."""
        old = self._elite_cache.get(field)
        if old is not None and old.id != program.id:
            self._elite_reverse.pop(old.id, None)
        self._elite_cache[field] = program
        self._elite_reverse[program.id] = field

    def _cache_remove_field(self, field: str) -> None:
        old = self._elite_cache.pop(field, None)
        if old is not None:
            self._elite_reverse.pop(old.id, None)

    def _cache_remove_id(self, program_id: str) -> str | None:
        """Remove by program ID; returns cell field if found."""
        field = self._elite_reverse.pop(program_id, None)
        if field is not None:
            self._elite_cache.pop(field, None)
        return field

    def _cache_clear(self) -> None:
        self._elite_cache.clear()
        self._elite_reverse.clear()

    async def get_elite(self, cell: CellDescriptor) -> Program | None:
        await self._ensure_cache()
        return self._elite_cache.get(self._field(cell))

    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program | None], bool],
    ) -> bool:
        """Add elite, using in-memory cache to skip Redis reads for non-improving programs."""
        await self._ensure_cache()
        field = self._field(cell)

        # Fast path: compare in-memory (0 Redis RT for rejected programs)
        current_prog = self._elite_cache.get(field)
        if current_prog is not None and not is_better(program, current_prog):
            return False

        # Program improves (or cell is empty).  Verify it exists in storage
        # before committing to Redis.
        if not await self._storage.exists(program.id):
            logger.debug("[Archive] add ignored: program {} not in storage", program.id)
            return False

        current_id = current_prog.id if current_prog else None

        async def _op(r):
            while True:
                try:
                    await r.watch(self._hash_key)

                    # Re-check Redis state in case of concurrent modification
                    # (defensive; single-engine makes this unlikely)
                    redis_id = await r.hget(self._hash_key, field)
                    if redis_id and redis_id != (current_id or ""):
                        # Cache was stale — reload current from Redis
                        redis_prog = await self._storage.get(redis_id)
                        if redis_prog and not is_better(program, redis_prog):
                            await r.unwatch()
                            # Fix cache to match Redis
                            self._cache_set(field, redis_prog)
                            return False

                    pipe = r.pipeline()
                    pipe.multi()
                    pipe.hset(self._hash_key, field, program.id)
                    if redis_id and redis_id != program.id:
                        pipe.hdel(self._reverse_key, redis_id)
                    pipe.hset(self._reverse_key, program.id, field)
                    await pipe.execute()
                    return True

                except WatchError:
                    continue

        ok = await self._storage.with_redis("archive:add_elite", _op)
        if ok:
            self._cache_set(field, program)
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
            self._cache_remove_field(field)
            logger.debug("[Archive] removed cell {}", field)
        return bool(removed)

    async def get_all_elites(self) -> list[str]:
        """Return all elite program IDs (already unique due to 1:1 mapping)."""
        await self._ensure_cache()
        return sorted(p.id for p in self._elite_cache.values())

    async def size(self) -> int:
        await self._ensure_cache()
        return len(self._elite_cache)

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
            self._cache_remove_id(program_id)
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
            for pid in program_ids:
                self._cache_remove_id(pid)
            logger.debug("[Archive] bulk removed {} ids", count)
        return int(count)

    async def clear_all_elites(self) -> int:
        """Clear all elites and reverse index."""
        await self._ensure_cache()
        count = len(self._elite_cache)
        if count == 0:
            return 0

        async def _op(r):
            pipe = r.pipeline(transaction=False)
            pipe.delete(self._hash_key)
            pipe.delete(self._reverse_key)
            await pipe.execute()

        await self._storage.with_redis("archive:clear_all", _op)
        self._cache_clear()

        logger.debug("[Archive] cleared {} elites", count)
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
