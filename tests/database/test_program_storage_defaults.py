"""Default implementations and contracts on the ProgramStorage ABC."""

from __future__ import annotations

from typing import Any

import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.exceptions import StorageError


class _StubStorage(ProgramStorage):
    """Minimal concrete subclass: only get_all_program_ids is functional."""

    def __init__(self, ids: list[str], *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self._ids = ids
        self.closed = False

    @property
    def key_prefix(self) -> str:
        return "stub"

    async def add(self, program) -> None:
        raise NotImplementedError

    async def update(self, program) -> None:
        raise NotImplementedError

    async def get(self, program_id: str):
        raise NotImplementedError

    async def mget(self, program_ids, *, exclude=None):
        raise NotImplementedError

    async def exists(self, program_id: str) -> bool:
        raise NotImplementedError

    async def publish_status_event(
        self, status: str, program_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        raise NotImplementedError

    async def get_all(self, *, exclude=None):
        raise NotImplementedError

    async def get_all_by_status(self, status: str, *, exclude=None):
        raise NotImplementedError

    async def get_ids_by_status(self, status: str):
        raise NotImplementedError

    async def count_by_status(self, status: str) -> int:
        raise NotImplementedError

    async def remove(self, program_id: str) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def transition_status(self, program_id: str, old, new) -> None:
        raise NotImplementedError

    async def atomic_state_transition(self, program, old_state, new_state) -> None:
        raise NotImplementedError

    async def acquire_instance_lock(self) -> bool:
        return True

    async def release_instance_lock(self) -> None:
        pass

    async def renew_instance_lock(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True

    async def get_all_program_ids(self) -> list[str]:
        return list(self._ids)


@pytest.mark.asyncio
async def test_size_defaults_to_program_id_count():
    assert await _StubStorage(["a", "b", "c"]).size() == 3


@pytest.mark.asyncio
async def test_has_data_reflects_emptiness():
    assert await _StubStorage(["a"]).has_data() is True
    assert await _StubStorage([]).has_data() is False


def test_require_writable_raises_when_read_only():
    s = _StubStorage([], read_only=True)
    assert s.read_only is True
    with pytest.raises(StorageError, match="read-only"):
        s.require_writable("add")


def test_require_writable_noop_when_writable():
    _StubStorage([]).require_writable("add")


@pytest.mark.asyncio
async def test_async_context_manager_closes():
    async with _StubStorage([]) as s:
        assert s.closed is False
    assert s.closed is True
