"""Factory construction of read-only storage instances."""

from __future__ import annotations

from gigaevo.database.factory import RedisRunConfig, build_readonly_redis_storage
from gigaevo.database.redis_program_storage import RedisProgramStorage


def test_builds_readonly_storage_with_url_and_prefix():
    storage = build_readonly_redis_storage(
        host="localhost", port=6379, db=3, key_prefix="myprob"
    )
    assert isinstance(storage, RedisProgramStorage)
    assert storage.read_only is True
    assert storage.key_prefix == "myprob"
    assert "localhost:6379/3" in str(storage.config.redis_url)


def test_run_config_round_trips_through_factory():
    cfg = RedisRunConfig(redis_db=3, redis_prefix="myprob")
    storage = build_readonly_redis_storage(
        host=cfg.redis_host,
        port=cfg.redis_port,
        db=cfg.redis_db,
        key_prefix=cfg.redis_prefix,
    )
    assert storage.key_prefix == "myprob"
    assert storage.read_only is True
