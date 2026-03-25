from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Any

from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState, validate_transition


class PopulationSnapshot:
    """Epoch-based cache for ``storage.get_all()``.

    Shared across all collector instances via ``storage.snapshot``.
    Call :meth:`bump` at phase boundaries to invalidate; collectors that
    see a stale epoch will re-fetch exactly once (others piggyback).

    Supports two invalidation modes:

    * **Full** (``bump()``): clears cached data, forcing a complete refetch.
      Use when program *data* (metrics, lineage) may have changed.
    * **Incremental** (``bump(incremental=True)``): preserves cached data and
      only fetches new/removed programs on the next ``get_all``.  Use when
      only program *states* (set membership) changed — not data fields that
      the caller reads.

    Single-slot cache keyed on ``(epoch, exclude)``. Works correctly when all
    callers within an epoch use the same exclude value (which is true in practice:
    EvolutionaryStatisticsCollector always uses exclude={"stage_results", "metadata"}, all
    other callers use exclude=None). If a new caller with a different exclude
    value is added, the cache will thrash with no benefits -- a latent hazard
    that would require a dict-based multi-slot cache to fully address.
    """

    __slots__ = (
        "_epoch",
        "_cached_epoch",
        "_cached_exclude",
        "_cached",
        "_lock",
        "_id_cache",
    )

    def __init__(self) -> None:
        self._epoch: int = 0
        self._cached_epoch: int = -1
        self._cached_exclude: frozenset[str] | None = None
        self._cached: list[Program] = []
        self._lock = asyncio.Lock()
        self._id_cache: dict[str, Program] = {}

    def bump(self, *, incremental: bool = False) -> None:
        """Increment the epoch, invalidating the cached snapshot.

        Args:
            incremental: If True, keep the internal program-ID cache so the
                next ``get_all`` can do an incremental update (fetch only
                new/removed programs).  Safe only when program *data* fields
                read by callers (metrics, lineage, generation) have not changed
                since the last full fetch — i.e. only program state/set
                membership changed.  If False (default), the ID cache is
                cleared, forcing a full refetch.
        """
        self._epoch += 1
        if incremental:
            self._prepare_id_cache()
        else:
            self._id_cache.clear()

    async def get_all(
        self,
        storage: ProgramStorage,
        *,
        exclude: frozenset[str] | None = None,
    ) -> list[Program]:
        """Return cached programs if epoch+exclude matches, else fetch + cache."""
        if self._cached_epoch == self._epoch and self._cached_exclude == exclude:
            return self._cached
        async with self._lock:
            if self._cached_epoch == self._epoch and self._cached_exclude == exclude:
                return self._cached

            if self._id_cache and self._cached_exclude == exclude:
                programs = await self._incremental_update(storage, exclude)
            else:
                programs = await storage.get_all(exclude=exclude)
                # Defer _id_cache build to _incremental_update (avoid overhead
                # when the next bump is a full invalidation).
                self._id_cache.clear()

            self._cached = programs
            self._cached_exclude = exclude
            self._cached_epoch = self._epoch
            return programs

    async def _incremental_update(
        self,
        storage: ProgramStorage,
        exclude: frozenset[str] | None,
    ) -> list[Program]:
        """Fetch only new/removed programs, reuse cached objects for the rest."""
        current_ids = set(await storage.get_all_program_ids())
        cached_ids = set(self._id_cache)

        new_ids = current_ids - cached_ids
        removed_ids = cached_ids - current_ids

        if new_ids:
            new_programs = await storage.mget(list(new_ids), exclude=exclude)
            for p in new_programs:
                self._id_cache[p.id] = p

        for rid in removed_ids:
            del self._id_cache[rid]

        return list(self._id_cache.values())

    def _prepare_id_cache(self) -> None:
        """Build _id_cache from _cached if not yet populated (lazy init)."""
        if not self._id_cache and self._cached:
            self._id_cache = {p.id: p for p in self._cached}


class ProgramStorage(ABC):
    """Abstract interface for persisting :class:`Program` objects."""

    def __init__(self) -> None:
        self.snapshot = PopulationSnapshot()

    @abstractmethod
    async def add(self, program: Program) -> None: ...

    @abstractmethod
    async def update(self, program: Program) -> None: ...

    async def write_exclusive(self, program: Program) -> None:
        """Fast write without WATCH/MERGE. Default falls back to update().

        Safe only in exclusive-ownership contexts (during DAG execution).
        Implementations may override with a 2 RT path (INCR + SET).
        """
        await self.update(program)

    @abstractmethod
    async def get(self, program_id: str) -> Program | None: ...

    @abstractmethod
    async def mget(
        self,
        program_ids: list[str],
        *,
        exclude: frozenset[str] | None = None,
    ) -> list[Program]:
        """Return programs for the given IDs (skipping missing keys).

        Args:
            exclude: Optional set of field names to skip during deserialization.
                Excluded fields get their Pydantic defaults. Same semantics as
                :meth:`get_all` ``exclude``.
        """
        ...

    @abstractmethod
    async def exists(self, program_id: str) -> bool: ...

    @abstractmethod
    async def publish_status_event(
        self,
        status: str,
        program_id: str,
        extra: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_all(self, *, exclude: frozenset[str] | None = None) -> list[Program]:
        """Return all programs.

        Args:
            exclude: Optional set of field names to skip during deserialization.
                Excluded fields get their Pydantic defaults. Implementations
                that don't support projection may ignore this parameter.
        """
        ...

    @abstractmethod
    async def get_all_by_status(
        self, status: str, *, exclude: frozenset[str] | None = None
    ) -> list[Program]: ...

    @abstractmethod
    async def get_ids_by_status(self, status: str) -> list[str]:
        """Return IDs of programs with the given status (no full fetch)."""
        ...

    @abstractmethod
    async def count_by_status(self, status: str) -> int:
        """Return count of programs with the given status (without fetching data)."""
        ...

    @abstractmethod
    async def get_all_program_ids(self) -> list[str]: ...

    @abstractmethod
    async def transition_status(
        self, program_id: str, old: str | None, new: str
    ) -> None: ...

    @abstractmethod
    async def atomic_state_transition(
        self, program: Program, old_state: str | None, new_state: str
    ) -> None:
        """
        Atomically update program state AND status set membership in a single transaction.
        This ensures program.state and status sets never get out of sync.

        Args:
            program: Program object with updated state
            old_state: Previous state value (for removing from old set)
            new_state: New state value (for adding to new set)

        Raises:
            StorageError: If atomic operation fails
        """
        ...

    @abstractmethod
    async def acquire_instance_lock(self) -> bool:
        """
        Acquire an exclusive lock on this storage prefix to prevent multiple instances.

        Returns:
            True if lock was acquired, False if another instance holds the lock

        Raises:
            StorageError: If lock acquisition fails or another instance is detected
        """
        ...

    @abstractmethod
    async def release_instance_lock(self) -> None:
        """
        Release the instance lock acquired by acquire_instance_lock().
        Should be called during shutdown.
        """
        ...

    @abstractmethod
    async def renew_instance_lock(self) -> bool:
        """
        Renew the instance lock to prevent expiry.
        Should be called periodically while instance is running.

        Returns:
            True if renewal succeeded, False if lock was lost
        """
        ...

    async def fast_state_transition(
        self, program: Program, old_state: str, new_state: str
    ) -> None:
        """Fast state transition: 2 RT (INCR + pipeline) instead of ~5 RT.

        Safe only when the caller holds exclusive single-process ownership
        (e.g., asyncio.Lock in ProgramStateManager). Does NOT provide cross-process
        safety — assumes each program is processed by exactly one engine instance.
        Default falls back to atomic_state_transition.
        """
        await self.atomic_state_transition(program, old_state, new_state)

    async def batch_transition_state(
        self,
        programs: list[Program],
        old_state: str,
        new_state: str,
    ) -> int:
        """Batch-transition programs between states.

        Default implementation falls back to individual transitions.
        Subclasses may override with pipelined operations.
        Returns the number of programs transitioned.
        """
        old_enum = ProgramState(old_state)
        new_enum = ProgramState(new_state)
        count = 0
        for prog in programs:
            validate_transition(old_enum, new_enum)
            prog.state = new_enum
            await self.atomic_state_transition(prog, old_state, new_state)
            count += 1
        return count

    async def batch_transition_by_ids(
        self,
        program_ids: list[str],
        old_state: str,
        new_state: str,
    ) -> int:
        """Batch-transition programs by ID.

        Default implementation falls back to mget + batch_transition_state.
        Subclasses may override with optimized raw-blob patching.
        """
        if not program_ids:
            return 0
        programs = await self.mget(program_ids)
        matching = [p for p in programs if p.state.value == old_state]
        if not matching:
            return 0
        return await self.batch_transition_state(matching, old_state, new_state)

    async def remove_ids_from_status_set(self, status: str, ids: list[str]) -> None:
        """Remove specific IDs from a status set. No-op by default."""

    async def batch_move_status_sets(
        self,
        program_ids: list[str],
        from_status: str,
        to_status: str,
    ) -> None:
        """Move IDs between status sets WITHOUT modifying program data blobs.

        This is a lightweight alternative to ``batch_transition_by_ids`` for
        transitions to terminal states (e.g. DISCARDED) where the stored program
        blob is never read again.  Skips MGET/parse/patch/serialize — only does
        SREM from *from_status* set + SADD to *to_status* set.

        .. warning::
            After this call the blob's ``state`` field will still contain the old
            state value.  Only use this when: (1) the target state is terminal,
            and (2) no production code reads the blob of programs in that state.

        Default implementation falls back to ``batch_transition_by_ids``.
        """
        if program_ids:
            await self.batch_transition_by_ids(program_ids, from_status, to_status)

    async def wait_for_activity(self, timeout: float) -> None:
        """
        Block up to `timeout` seconds until storage observes activity (e.g., new
        program or status change). Default implementation just sleeps.
        Storages with push/notify capability should override.
        """
        await asyncio.sleep(timeout)

    async def save_run_state(self, field: str, value: int) -> None:
        """Persist a named integer counter for resume support. No-op by default."""

    async def load_run_state(self, field: str) -> int | None:
        """Load a previously saved integer counter. Returns None if not found."""
        return None

    async def recover_stranded_programs(self) -> int:
        """Reset RUNNING programs to QUEUED after a crash/kill (for resume).

        Returns the number of programs recovered.
        """
        return 0

    @abstractmethod
    async def close(self) -> None: ...
