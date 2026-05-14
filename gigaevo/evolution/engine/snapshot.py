"""Versioned snapshot of generic engine state, persisted to Redis.

Written by the base :class:`EvolutionEngine` via ``_write_snapshot`` and read
by any stage or external consumer that needs engine-aware behavior.

Last-writer-wins semantics: the engine is single-process async; exactly one
coroutine writes the snapshot. No CAS, no retries.

Sync + async access
-------------------
Some readers are sync (notably ``Stage.compute_hash``, a classmethod used for
cache-key computation). To avoid forcing them async, this module maintains a
process-wide ``_CURRENT_SNAPSHOT`` variable that is updated by every
``_write_snapshot`` call alongside the Redis write. Sync readers call
``get_current_snapshot()``; async readers (or out-of-process consumers)
call ``load_engine_snapshot(storage)``.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict

ENGINE_SNAPSHOT_KEY = "engine:snapshot"


class EngineSnapshot(BaseModel):
    total_mutants: int = 0
    programs_processed: int = 0
    completion_reason: str | None = None
    version: int = 0

    model_config = ConfigDict(frozen=True, extra="forbid")


class _SnapshotStorage(Protocol):
    async def load_run_state_str(self, field: str) -> str | None: ...
    async def save_run_state(self, field: str, value: int | str) -> None: ...


_CURRENT_SNAPSHOT: EngineSnapshot = EngineSnapshot()


def get_current_snapshot() -> EngineSnapshot:
    """Return the in-process snapshot mirror. Sync-safe."""
    return _CURRENT_SNAPSHOT


def set_current_snapshot(snap: EngineSnapshot) -> None:
    """Overwrite the in-process mirror. Called by ``EvolutionEngine._write_snapshot``
    and ``_load_snapshot_on_resume`` only — do not call from application code.
    """
    global _CURRENT_SNAPSHOT
    _CURRENT_SNAPSHOT = snap


def _reset_current_snapshot_for_tests() -> None:
    """Reset the module-level mirror to defaults. Use in test fixtures only."""
    global _CURRENT_SNAPSHOT
    _CURRENT_SNAPSHOT = EngineSnapshot()


async def load_engine_snapshot(storage: _SnapshotStorage) -> EngineSnapshot:
    """Load the snapshot from Redis, returning defaults if absent or corrupt."""
    raw = await storage.load_run_state_str(ENGINE_SNAPSHOT_KEY)
    if raw is None:
        return EngineSnapshot()
    try:
        return EngineSnapshot.model_validate_json(raw)
    except Exception as exc:
        logger.warning("engine:snapshot JSON corrupt ({}); returning defaults", exc)
        return EngineSnapshot()
