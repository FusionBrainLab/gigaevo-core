"""Experiment lifecycle utilities."""

from __future__ import annotations

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage

STORAGE_NOT_EMPTY_ERROR = """
ERROR: program storage is not empty!

  {location} contains existing programs.

To prevent accidental data loss, flush it manually:
  {flush_hint}

Or set resume=true to continue with existing data:
  python run.py redis.resume=true ...
"""


async def check_storage_resume(
    storage: ProgramStorage,
    *,
    resume: bool,
    location: str,
    flush_hint: str,
) -> bool:
    """Decide fresh-start vs resume; refuse to clobber existing data.

    Returns True iff existing data should be resumed.
    Raises RuntimeError if storage has data and resume is False.
    """
    has_data = await storage.has_data()
    if has_data and not resume:
        logger.error(
            STORAGE_NOT_EMPTY_ERROR.format(location=location, flush_hint=flush_hint)
        )
        raise RuntimeError(f"{location} is not empty. Flush manually to proceed.")
    if has_data:
        logger.info("Resuming experiment: {} has existing data", location)
    elif resume:
        logger.info("Resume requested but {} is empty. Starting fresh.", location)
    return has_data
