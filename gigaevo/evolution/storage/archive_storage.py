from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

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
    async def size(self) -> int: ...

    # Returns number of occupied cells.


class RedisArchiveStorage(ArchiveStorage):
    """
    Redis-backed archive: a single hash at `prefix:archive`
      field = cell serialized as "c0,c1,c2,..."
      value = program_id (string)
    """

    def __init__(
        self, program_storage: RedisProgramStorage, key_prefix: str | None = None
    ) -> None:
        self._storage = program_storage
        self._state_manager = ProgramStateManager(program_storage)
        prefix = key_prefix or program_storage.config.key_prefix
        self._hash_key = f"{prefix}:archive"

    # -------- small helpers --------

    @staticmethod
    def _field(cell: CellDescriptor) -> str:
        return ",".join(map(str, cell))

    async def _hget(self, field: str) -> str | None:
        async def _op(r):
            return await r.hget(self._hash_key, field)

        return await self._storage._with_redis("archive:hget", _op)

    async def _hset(self, field: str, program_id: str) -> None:
        async def _op(r):
            await r.hset(self._hash_key, field, program_id)

        await self._storage._with_redis("archive:hset", _op)

    async def _hdel(self, *fields: str) -> int:
        async def _op(r):
            return await r.hdel(self._hash_key, *fields)

        return await self._storage._with_redis("archive:hdel", _op)

    async def _hvals(self) -> list[str]:
        async def _op(r):
            return await r.hvals(self._hash_key)

        return await self._storage._with_redis("archive:hvals", _op) or []

    async def _hlen(self) -> int:
        async def _op(r):
            return await r.hlen(self._hash_key)

        return await self._storage._with_redis("archive:hlen", _op)

    async def _hgetall(self) -> dict[str, str]:
        async def _op(r):
            return await r.hgetall(self._hash_key)

        return await self._storage._with_redis("archive:hgetall", _op) or {}

    async def _delete_hash(self) -> None:
        async def _op(r):
            await r.delete(self._hash_key)

        await self._storage._with_redis("archive:delete", _op)

    async def _discard_if_exists(self, program_id: str) -> None:
        prog = await self._storage.get(program_id)
        if prog is not None:
            await self._state_manager.set_program_state(prog, ProgramState.DISCARDED)

    async def get_elite(self, cell: CellDescriptor) -> Program | None:
        pid = await self._hget(self._field(cell))
        return await self._storage.get(pid) if pid else None

    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program | None], bool],
    ) -> bool:
        if not await self._storage.exists(program.id):
            logger.debug("[Archive] add ignored: program {} not in storage", program.id)
            return False

        field = self._field(cell)

        async def _op(r):
            current_id = await r.hget(self._hash_key, field)
            if current_id:
                current_prog = await self._storage.get(current_id)
                if current_prog and not is_better(program, current_prog):
                    return False
            await r.hset(self._hash_key, field, program.id)
            return True

        ok = await self._storage._with_redis("archive:add_elite", _op)
        if ok:
            logger.debug("[Archive] cell {} -> {}", field, program.id)
        return bool(ok)

    async def remove_elite(self, cell: CellDescriptor) -> bool:
        field = self._field(cell)
        pid = await self._hget(field)
        removed = await self._hdel(field) > 0
        if removed and pid:
            await self._discard_if_exists(pid)
            logger.debug(
                "[Archive] removed cell {} (program {}) -> DISCARDED", field, pid
            )
        return removed

    async def get_all_elites(self) -> list[str]:
        """
        Return unique program IDs that are elites across all cells.
        """
        ids = await self._hvals()
        # Deduplicate and preserve a stable order (optional: sorted).
        return sorted(set(ids))

    async def size(self) -> int:
        return await self._hlen()

    async def remove_elite_by_id(self, program_id: str) -> bool:
        mapping = await self._hgetall()
        fields = [k for k, v in mapping.items() if v == program_id]
        if not fields:
            return False

        await self._hdel(*fields)
        await self._discard_if_exists(program_id)
        logger.debug(
            "[Archive] removed id {} from {} cell(s) -> DISCARDED",
            program_id,
            len(fields),
        )
        return True

    async def clear_all_elites(self) -> int:
        mapping = await self._hgetall()
        count = len(mapping)
        if count == 0:
            return 0

        await self._delete_hash()

        # Discard all unique program IDs that were elites
        for pid in set(mapping.values()):
            await self._discard_if_exists(pid)

        logger.debug(
            "[Archive] cleared {} elites ({} unique ids) -> DISCARDED",
            count,
            len(set(mapping.values())),
        )
        return count
