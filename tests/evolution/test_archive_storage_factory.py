"""ArchiveStorageFactory construction and prefix scoping."""

from __future__ import annotations

from gigaevo.evolution.storage.archive_storage import (
    RedisArchiveStorage,
    RedisArchiveStorageFactory,
)
from tests.database.storage_backends import _fakeredis_storage


async def test_factory_builds_redis_archive_with_explicit_prefix():
    async with _fakeredis_storage() as storage:
        archive = RedisArchiveStorageFactory(storage)("island_0")
        assert isinstance(archive, RedisArchiveStorage)
        assert archive._hash_key == "island_0:archive"


async def test_factory_falls_back_to_storage_prefix():
    async with _fakeredis_storage() as storage:
        archive = RedisArchiveStorageFactory(storage)()
        assert archive._hash_key == f"{storage.key_prefix}:archive"
