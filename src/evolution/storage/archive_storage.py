from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger

from src.database.redis_program_storage import RedisProgramStorage
from src.database.state_manager import ProgramStateManager
from src.programs.program import Program
from src.programs.program_state import ProgramState

CellDescriptor = tuple[int, ...]


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
    def __init__(
        self, program_storage: RedisProgramStorage, key_prefix: str | None = None
    ) -> None:
        self._storage = program_storage
        self._state_manager = ProgramStateManager(program_storage)
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

        # Get the program ID before removal so we can update its state
        program_id = None

        async def _get_before_del(r):
            nonlocal program_id
            program_id = await r.hget(self._key, field)
            return program_id

        await self._storage._with_redis("archive_get_before_remove", _get_before_del)

        async def _del(r):
            return (await r.hdel(self._key, field)) > 0

        removed = await self._storage._with_redis("archive_remove_elite", _del)

        # If elite was removed, get the program and set its state to DISCARDED
        if removed and program_id:
            program = await self._storage.get(program_id)
            if program:
                await self._state_manager.set_program_state(
                    program, ProgramState.DISCARDED
                )
                logger.debug(
                    "archive removed elite from cell {} (program {}), set state to DISCARDED",
                    field,
                    program_id,
                )

        return removed

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
        # Get the fields before removal for logging
        fields_to_remove = []

        async def _find_fields(r):
            nonlocal fields_to_remove
            mapping = await r.hgetall(self._key)
            fields_to_remove = [k for k, v in mapping.items() if v == program_id]
            return len(fields_to_remove) > 0

        found = await self._storage._with_redis("archive_find_by_id", _find_fields)
        if not found:
            return False

        async def _del(r):
            await r.hdel(self._key, *fields_to_remove)
            return True

        removed = await self._storage._with_redis("archive_remove_by_id", _del)

        # If elite was removed, set its state to DISCARDED
        if removed:
            program = await self._storage.get(program_id)
            if program:
                await self._state_manager.set_program_state(
                    program, ProgramState.DISCARDED
                )
                logger.debug(
                    "archive removed elite by ID {} from {} cells, set state to DISCARDED",
                    program_id,
                    len(fields_to_remove),
                )

        return removed

    async def clear_all_elites(self) -> int:
        # Get all program IDs before clearing
        program_ids = []

        async def _get_all_ids(r):
            nonlocal program_ids
            program_ids = await r.hvals(self._key)
            n = await r.hlen(self._key)
            await r.delete(self._key)
            return n

        count = await self._storage._with_redis("archive_clear_all", _get_all_ids)

        # Set all removed programs to DISCARDED state
        if count > 0 and program_ids:
            # Deduplicate program IDs
            unique_ids = list(set(program_ids))
            programs = await self._storage.mget(unique_ids)

            # Set state for each program
            for program in programs:
                if program:
                    await self._state_manager.set_program_state(
                        program, ProgramState.DISCARDED
                    )

            logger.debug(
                "archive cleared {} elites ({} unique programs), set all to DISCARDED",
                count,
                len(unique_ids),
            )

        return count
