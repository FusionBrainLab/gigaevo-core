from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import math
from typing import TYPE_CHECKING, Final

from loguru import logger
from redis.exceptions import WatchError

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.dataplane import (
    CellKey,
    DataPlane,
    EliteInserted,
    EliteRejected,
    EliteSwapped,
    Err,
    Freshness,
    FreshnessEventual,
    ProgramId,
    Token,
    mint_root,
)
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.dataplane.engine_startup import EngineRoot

CellDescriptor = tuple[int, ...]


# Retry budget for the optimistic CAS in
# :meth:`RedisArchiveStorage.add_elite`; exhaustion surfaces as
# ``False`` so a contended cell cannot hang a caller indefinitely.
_WATCH_MAX_ATTEMPTS: Final[int] = 50

# Stable-by-default tie-break: occupant wins equal-score comparisons,
# matching :class:`SumArchiveSelector`'s strict ``>`` callback.
_DEFAULT_TIEBREAK_BIT: Final[int] = 0


# ------------------------------- Helpers ---------------------------------


def _reduce_selector_score(
    selector: Callable[[Program, Program], bool], program: Program
) -> float | None:
    """Probe a selector for its scalar score, if it exposes one.

    Selectors that subclass :class:`ArchiveSelector` expose a scalar via
    ``reduce_to_score``; bare callables and multi-criteria selectors
    return ``None`` and the caller takes the WATCH path. NaN / inf
    scores are also rejected here so the fallback handles a degenerate
    metric without round-tripping to Lua.
    """
    reducer = getattr(selector, "reduce_to_score", None)
    if reducer is None:
        return None
    score = reducer(program)
    if score is None:
        return None
    score_f = float(score)
    if math.isnan(score_f) or math.isinf(score_f):
        return None
    return score_f


# ------------------------------- Interface -------------------------------


class ArchiveStorage(ABC):
    """Elite archive keyed by behavior-space cells."""

    @abstractmethod
    async def get_elite(
        self,
        cell: CellDescriptor,
        *,
        freshness: Freshness | None = None,
    ) -> Program | None: ...

    @abstractmethod
    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program], bool],
    ) -> bool: ...

    @abstractmethod
    async def remove_elite(self, cell: CellDescriptor) -> bool: ...

    @abstractmethod
    async def get_all_elites(self) -> list[str]: ...

    # Returns unique program IDs that are currently elites in any cell.

    @abstractmethod
    async def remove_elite_by_id(self, program_id: str) -> bool: ...

    @abstractmethod
    async def bulk_remove_elites_by_id(self, program_ids: list[str]) -> int:
        """Remove multiple elites atomically. Returns number actually removed."""
        ...

    @abstractmethod
    async def clear_all_elites(self) -> int: ...

    # Returns number of cells cleared.

    @abstractmethod
    async def bulk_add_elites(
        self,
        placements: list[tuple[CellDescriptor, Program]],
        is_better: Callable[[Program, Program], bool],
    ) -> int: ...

    # Adds multiple elites at once (e.g., during re-indexing). Returns number of successful adds.

    @abstractmethod
    async def size(self) -> int: ...

    # Returns number of occupied cells.


class RedisArchiveStorage(ArchiveStorage):
    """Redis-backed archive with bounded optimistic locking + reverse index.

    Data structures:
      - ``{prefix}:archive`` (hash): cell -> program_id
      - ``{prefix}:archive:reverse`` (hash): program_id -> cell (1:1)
      - ``{prefix}:archive:scores`` (hash): cell -> occupant score
        (populated by the dataplane swap path only)

    ``add_elite`` dispatches on the comparator's expressiveness: scalar-
    reducible selectors take the dataplane path (one atomic Lua swap),
    multi-criteria selectors fall back to a WATCH/MULTI/EXEC CAS
    bounded by :data:`_WATCH_MAX_ATTEMPTS`.
    """

    def __init__(
        self,
        program_storage: RedisProgramStorage,
        key_prefix: str | None = None,
        *,
        dataplane: DataPlane | None = None,
        engine_root: EngineRoot | None = None,
    ) -> None:
        self._storage = program_storage
        prefix = key_prefix or program_storage.config.key_prefix
        self._hash_key = f"{prefix}:archive"
        self._reverse_key = f"{prefix}:archive:reverse"
        self._dataplane = dataplane
        # When supplied, per-call swap tokens are derived by linear
        # split from this root rather than ad-hoc minted.
        self._engine_root = engine_root

    # -------- small helpers --------

    @staticmethod
    def _field(cell: CellDescriptor) -> str:
        return ",".join(map(str, cell))

    async def _hget(self, field: str) -> str | None:
        async def _op(r):
            return await r.hget(self._hash_key, field)

        return await self._storage.with_redis("archive:hget", _op)

    async def _hvals(self) -> list[str]:
        async def _op(r):
            return await r.hvals(self._hash_key)

        return await self._storage.with_redis("archive:hvals", _op) or []

    async def _hlen(self) -> int:
        async def _op(r):
            return await r.hlen(self._hash_key)

        return await self._storage.with_redis("archive:hlen", _op)

    async def get_elite(
        self,
        cell: CellDescriptor,
        *,
        freshness: Freshness | None = None,
    ) -> Program | None:
        """Return the elite program occupying ``cell``, or ``None``.

        Default :class:`FreshnessEventual` reads admit whatever Redis
        has now. A stricter floor (e.g. :class:`FreshnessAtLeast`)
        routes through :meth:`DataPlane.read_program` and raises
        :class:`StaleReadError` on epoch underflow; a wired dataplane
        is required, otherwise :class:`RuntimeError` is raised so a
        silent stale read is not possible.
        """
        field = self._field(cell)
        pid = await self._hget(field)
        if not pid:
            return None
        effective_freshness: Freshness = (
            freshness if freshness is not None else FreshnessEventual()
        )
        if isinstance(effective_freshness, FreshnessEventual):
            return await self._storage.get(pid)
        if self._dataplane is None:
            raise RuntimeError(
                "RedisArchiveStorage.get_elite: non-eventual freshness "
                "requires a wired DataPlane; none is attached"
            )
        result = await self._dataplane.read_program(
            ProgramId(pid), freshness=effective_freshness
        )
        if isinstance(result, Err):
            raise result.error
        if result.value is None:
            return None
        # Freshness has cleared; defer Program decoding to a single
        # owner so atomic_counter / dict-field / exclude semantics stay
        # consistent with non-coordinator reads.
        return await self._storage.get(pid)

    async def add_elite(
        self,
        cell: CellDescriptor,
        program: Program,
        is_better: Callable[[Program, Program], bool],
    ) -> bool:
        """Add ``program`` to ``cell``, dispatching on comparator shape.

        Scalar-reducible comparators take the dataplane CAS path; the
        rest fall back to bounded WATCH/MULTI/EXEC. Returns ``True``
        when the candidate was installed (inserted or swapped).
        """
        field = self._field(cell)

        # Candidate must already exist in program storage; both paths
        # trust the caller to have persisted the blob.
        if not await self._storage.exists(program.id):
            logger.debug("[Archive] add ignored: program {} not in storage", program.id)
            return False

        score = _reduce_selector_score(is_better, program)
        if self._dataplane is not None and score is not None:
            return await self._add_elite_via_dataplane(field, program, score)

        async def _op(r):
            attempts = 0
            while True:
                attempts += 1
                if attempts > _WATCH_MAX_ATTEMPTS:
                    logger.warning(
                        "[Archive] add_elite gave up on cell {} after {} WATCH retries",
                        field,
                        _WATCH_MAX_ATTEMPTS,
                    )
                    return False
                try:
                    async with r.pipeline() as pipe:
                        await pipe.watch(self._hash_key)
                        redis_id = await pipe.hget(self._hash_key, field)

                        if redis_id:
                            redis_prog = await self._storage.get(redis_id)
                            if redis_prog and not is_better(program, redis_prog):
                                await pipe.unwatch()
                                return False

                        pipe.multi()
                        pipe.hset(self._hash_key, field, program.id)
                        if redis_id and redis_id != program.id:
                            pipe.hdel(self._reverse_key, redis_id)
                        pipe.hset(self._reverse_key, program.id, field)
                        await pipe.execute()
                        return True

                except WatchError:
                    continue

        ok = await self._storage.with_redis("archive:add_elite", _op)
        if ok:
            logger.debug("[Archive] cell {} -> {}", field, program.id)
        return bool(ok)

    async def _add_elite_via_dataplane(
        self, field: str, program: Program, score: float
    ) -> bool:
        """Atomic CAS through :meth:`DataPlane.try_replace_elite`.

        Per-call token: derived from the engine cell root when wired,
        otherwise ad-hoc via :func:`mint_root`. Tiebreak bit pinned to
        :data:`_DEFAULT_TIEBREAK_BIT` (occupant wins).
        """
        assert self._dataplane is not None  # type narrowing
        cell_key = CellKey(field)
        if self._engine_root is not None:
            token: Token[CellKey] = self._engine_root.split_cell_token(cell_key)
        else:
            token = mint_root(cell_key)
        result = await self._dataplane.try_replace_elite(
            cell_key,
            ProgramId(program.id),
            token=token,
            candidate_score=score,
            tiebreak_bit=_DEFAULT_TIEBREAK_BIT,
        )
        if isinstance(result, Err):
            logger.warning(
                "[Archive] dataplane swap on cell {} failed: {}", field, result.error
            )
            return False
        outcome = result.value
        if isinstance(outcome, EliteInserted):
            logger.debug("[Archive] cell {} -> {} (inserted)", field, program.id)
            return True
        if isinstance(outcome, EliteSwapped):
            logger.debug(
                "[Archive] cell {} -> {} (swapped {})",
                field,
                program.id,
                outcome.displaced_id,
            )
            return True
        if isinstance(outcome, EliteRejected):
            return False
        # Unknown variant: log so a coordinator API drift surfaces on
        # the first call rather than masquerading as a no-op swap.
        logger.warning(
            "[Archive] dataplane swap on cell {} returned unknown outcome {!r}",
            field,
            outcome,
        )
        return False

    async def remove_elite(self, cell: CellDescriptor) -> bool:
        """Remove elite from cell and update reverse index."""
        field = self._field(cell)

        async def _op(r):
            current_id = await r.hget(self._hash_key, field)
            if not current_id:
                return False

            pipe = r.pipeline(transaction=False)
            pipe.hdel(self._hash_key, field)
            pipe.hdel(self._reverse_key, current_id)
            await pipe.execute()
            return True

        removed = await self._storage.with_redis("archive:remove_elite", _op)
        if removed:
            logger.debug("[Archive] removed cell {}", field)
        return bool(removed)

    async def get_all_elites(self) -> list[str]:
        """Return all elite program IDs (already unique due to 1:1 mapping)."""
        vals = await self._hvals()
        return sorted(vals)

    async def size(self) -> int:
        return await self._hlen()

    async def remove_elite_by_id(self, program_id: str) -> bool:
        """Remove program using reverse index (O(1) lookup)."""

        async def _op(r):
            cell = await r.hget(self._reverse_key, program_id)
            if not cell:
                return False

            pipe = r.pipeline(transaction=False)
            pipe.hdel(self._hash_key, cell)
            pipe.hdel(self._reverse_key, program_id)
            await pipe.execute()
            return True

        removed = await self._storage.with_redis("archive:remove_elite_by_id", _op)
        if removed:
            logger.debug("[Archive] removed id {}", program_id)
        return bool(removed)

    async def bulk_remove_elites_by_id(self, program_ids: list[str]) -> int:
        """Remove multiple elites using two Redis pipelines. Returns number actually removed."""
        if not program_ids:
            return 0

        async def _op(r):
            pipe = r.pipeline(transaction=False)
            for pid in program_ids:
                pipe.hget(self._reverse_key, pid)
            cells = await pipe.execute()

            pipe2 = r.pipeline(transaction=False)
            removed = 0
            for pid, cell in zip(program_ids, cells):
                if cell:
                    pipe2.hdel(self._hash_key, cell)
                    pipe2.hdel(self._reverse_key, pid)
                    removed += 1
            if removed:
                await pipe2.execute()
            return removed

        count = await self._storage.with_redis("archive:bulk_remove_elites_by_id", _op)
        if count:
            logger.debug("[Archive] bulk removed {} ids", count)
        return int(count)

    async def clear_all_elites(self) -> int:
        """Clear all elites and reverse index. Returns cells cleared."""
        count = await self._hlen()
        if count == 0:
            return 0

        async def _op(r):
            pipe = r.pipeline(transaction=False)
            pipe.delete(self._hash_key)
            pipe.delete(self._reverse_key)
            await pipe.execute()

        await self._storage.with_redis("archive:clear_all", _op)
        logger.debug("[Archive] cleared {} elites", count)
        return count

    async def bulk_add_elites(
        self,
        placements: list[tuple[CellDescriptor, Program]],
        is_better: Callable[[Program, Program], bool],
    ) -> int:
        """Sequential :meth:`add_elite` over the placements list."""
        if not placements:
            return 0

        added_count = 0
        for cell, program in placements:
            if await self.add_elite(cell, program, is_better):
                added_count += 1

        return added_count
