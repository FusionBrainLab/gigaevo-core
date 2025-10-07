from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger

from src.database.redis_program_storage import RedisProgramStorage
from src.programs.program import Program

CellDescriptor = tuple[int, ...]

__all__ = ["ArchiveStorage", "RedisArchiveStorage"]


class ArchiveStorage(ABC):
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
    async def get_all_elites(self) -> list[Program]: ...

    @abstractmethod
    async def remove_elite_by_id(self, program_id: str) -> bool: ...

    @abstractmethod
    async def clear_all_elites(self) -> int: ...


class RedisArchiveStorage(ArchiveStorage):
    def __init__(self, program_storage: RedisProgramStorage, key_prefix: str | None = None) -> None:
        self._storage = program_storage
        prefix = key_prefix or program_storage.config.key_prefix
        self._key = f"{prefix}:archive"

    @staticmethod
    def _field(cell: CellDescriptor) -> str:
        return ",".join(map(str, cell))

    async def get_elite(self, cell: CellDescriptor) -> Program | None:
        field = self._field(cell)

        async def _get(r):
            return await r.hget(self._key, field)

        pid = await self._storage._with_redis("archive_get_elite", _get)
        return await self._storage.get(pid) if pid else None

    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program | None], bool],
    ) -> bool:
        if not await self._storage.exists(program.id):
            logger.debug("archive add ignored: program {} not in storage", program.id)
            return False

        field = self._field(cell)

        async def _put(r):
            current_id = await r.hget(self._key, field)
            if current_id:
                current = await self._storage.get(current_id)
                if current and not is_better(program, current):
                    return False
            await r.hset(self._key, field, program.id)
            return True

        ok = await self._storage._with_redis("archive_add_elite", _put)
        if ok:
            logger.debug("archive cell {} -> {}", field, program.id)
        return ok

    async def remove_elite(self, cell: CellDescriptor) -> bool:
        field = self._field(cell)

        async def _del(r):
            return (await r.hdel(self._key, field)) > 0

        return await self._storage._with_redis("archive_remove_elite", _del)

    async def get_all_elites(self) -> list[Program]:
        async def _vals(r):
            return await r.hvals(self._key)

        ids = await self._storage._with_redis("archive_hvals", _vals)
        if not ids:
            return []
        # dedupe while preserving order
        seen = set[str]()
        unique_ids = [i for i in ids if not (i in seen or seen.add(i))]
        return await self._storage.mget(unique_ids)

    async def remove_elite_by_id(self, program_id: str) -> bool:
        async def _find_and_del(r):
            mapping = await r.hgetall(self._key)
            fields = [k for k, v in mapping.items() if v == program_id]
            if not fields:
                return False
            await r.hdel(self._key, *fields)
            return True

        return await self._storage._with_redis("archive_remove_by_id", _find_and_del)

    async def clear_all_elites(self) -> int:
        async def _clear(r):
            n = await r.hlen(self._key)
            await r.delete(self._key)
            return n

        return await self._storage._with_redis("archive_clear_all", _clear)
