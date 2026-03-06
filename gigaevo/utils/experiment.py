"""Experiment lifecycle utilities."""

from __future__ import annotations

from loguru import logger
from omegaconf import DictConfig

from gigaevo.database.redis_program_storage import RedisProgramStorage

REDIS_NOT_EMPTY_ERROR = """
ERROR: Redis database is not empty!

  Database {db} at {host}:{port} contains existing programs.

To prevent accidental data loss, you must manually flush the database.

Run this command to flush:
  redis-cli -h {host} -p {port} -n {db} FLUSHDB

Or use a different database number:
  python run.py redis.db=<number> ...

Or set resume=true to continue with existing data:
  python run.py redis.resume=true ...
"""


async def check_redis_resume(
    storage: RedisProgramStorage,
    cfg: DictConfig,
) -> bool:
    """Check Redis state and determine if we should resume.

    Args:
        storage: The Redis program storage
        cfg: Hydra config with redis.db, redis.host, redis.port, redis.resume

    Returns:
        True if should resume from existing data, False if starting fresh

    Raises:
        RuntimeError: If database has data but resume=False
    """
    has_data = await storage.has_data()
    resume = cfg.redis.get("resume", False)

    if has_data and not resume:
        error_msg = REDIS_NOT_EMPTY_ERROR.format(
            db=cfg.redis.db,
            host=cfg.redis.host,
            port=cfg.redis.port,
        )
        logger.error(error_msg)
        raise RuntimeError(
            f"Redis database {cfg.redis.db} is not empty. Flush manually to proceed."
        )

    if has_data and resume:
        logger.info(
            "Resuming experiment on database {} (found existing data)", cfg.redis.db
        )
    elif resume:
        logger.info(
            "Resume requested but database {} is empty. Starting fresh.", cfg.redis.db
        )

    return has_data and resume
