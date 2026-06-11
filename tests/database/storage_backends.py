"""Registry of ProgramStorage backends the contract suite runs against.

Adding a backend = appending one StorageBackend entry; the entire
contract suite in test_storage_contract.py then applies to it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import fakeredis.aioredis

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.redis.config import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage


@dataclass(frozen=True)
class StorageBackend:
    """Parametrizable backend factory for contract suite."""

    id: str
    make: Callable[..., AsyncIterator[ProgramStorage]]


@asynccontextmanager
async def _fakeredis_storage(
    *, read_only: bool = False
) -> AsyncIterator[ProgramStorage]:
    """RedisProgramStorage backed by fakeredis (async)."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
        read_only=read_only,
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    try:
        yield storage
    finally:
        await storage.close()


BACKENDS: list[StorageBackend] = [
    StorageBackend(id="redis-fake", make=_fakeredis_storage),
]
