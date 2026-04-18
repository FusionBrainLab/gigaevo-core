"""Opponent archive provider for adversarial co-evolution.

Reads opponent programs from a MAP-Elites archive in Redis.
Mirrors the RedisPromptStatsProvider pattern: injectable, async, multi-source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import random
import time

from loguru import logger
import numpy as np
from redis import asyncio as aioredis  # noqa: I001
from scipy.special import softmax

from gigaevo.adversarial.structured_logging import (
    emit_cell_pick,
    emit_hof_fetch,
    emit_hof_rotate,
)
from gigaevo.evolution.strategies.utils import weighted_sample_without_replacement


def _softmax_weights(fitnesses: list[float]) -> list[float]:
    """Softmax weights with min-max normalization and auto-temperature.

    Mirrors ``FitnessProportionalEliteSelector._compute_weights`` exactly so
    that opponent sampling uses the same distribution as parent selection.

    - Normalises fitnesses to [0, 1] (scale- and shift-invariant).
    - Auto-temperature = max(std(normalised), 0.01).
    - Falls back to uniform when all fitnesses are identical.
    """
    arr = np.asarray(fitnesses, dtype=np.float64)
    fitness_range = float(np.ptp(arr))
    if fitness_range < 1e-10:
        n = len(arr)
        return [1.0 / n] * n
    arr = (arr - arr.min()) / fitness_range
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    temp = max(std, 0.01)
    return softmax(arr / temp).tolist()


@dataclass
class OpponentProgram:
    """An opponent program fetched from the archive."""

    program_id: str
    code: str
    fitness: float


class OpponentArchiveProvider(ABC):
    """Abstract interface for fetching opponent programs."""

    @abstractmethod
    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        """Fetch up to n opponent programs from the archive.

        Returns:
            List of OpponentProgram (may be fewer than n if archive is small).
        """
        ...

    @abstractmethod
    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        """Return the top-k opponents by fitness (deterministic, not stochastic).

        Unlike get_opponents(), this always returns the globally best k programs
        from the current cache — suitable for mutation prompt feedback where the
        LLM should see the strongest rivals.

        Args:
            k: Number of opponents to return.
            higher_is_better: If True (default), highest fitness = strongest rival.
                Set to False for metrics where lower values are better.

        Returns:
            List of up to k OpponentProgram sorted by fitness (desc if higher_is_better).
        """
        ...

    @abstractmethod
    async def get_programs_by_ids(self, ids: list[str]) -> list[OpponentProgram]:
        """Return full OpponentProgram objects for the given IDs.

        Unlike get_codes_by_ids (codes only), returns fitness too — needed
        for ranking when showing l < k opponents in the mutation prompt.
        """
        ...

    @abstractmethod
    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        """Return codes for the given opponent program IDs.

        Called by FetchOpponentResultsStage after FetchOpponentIdsStage has
        sampled fresh opponent IDs.  Implementations should serve codes from
        an already-warm internal cache (no extra Redis round-trips needed).

        Returns:
            List of code strings for IDs found; IDs not in the archive are
            silently skipped.
        """
        ...

    async def close(self) -> None:
        """Close all Redis connections."""
        ...


class RedisOpponentArchiveProvider(OpponentArchiveProvider):
    """Reads opponent programs from MAP-Elites archive(s) in Redis.

    The archive lives at ``island_{island_id}:archive`` (hash: cell -> program_id).
    Programs are stored at ``{prefix}:program:{id}`` (JSON).

    Caches the opponent list for ``cache_ttl`` seconds to avoid
    hammering Redis on every evaluation.

    Args:
        host: Redis host
        port: Redis port
        sources: List of {"db": int, "prefix": str} dicts.
            Each source is one opponent run's Redis DB + key prefix.
        island_id: Island ID for archive key (default: "fitness_island").
        cache_ttl: Seconds to cache the opponent list (default: 30).
    """

    def __init__(
        self,
        host: str,
        port: int,
        sources: list[dict[str, int | str]],
        island_id: str = "fitness_island",
        cache_ttl: float = 30.0,
    ):
        self._host = host
        self._port = port
        self._sources = [(int(s["db"]), str(s["prefix"])) for s in sources]
        self._island_id = island_id
        self._cache_ttl = cache_ttl
        self._cache: list[OpponentProgram] = []
        self._cache_time: float = 0.0
        self._redis_clients: dict[int, aioredis.Redis] = {}  # type: ignore[type-arg]

    def _get_redis(self, db: int) -> aioredis.Redis:  # type: ignore[type-arg]
        if db not in self._redis_clients:
            self._redis_clients[db] = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=db,
                decode_responses=True,
            )
        return self._redis_clients[db]

    async def get_opponents_seeded(self, n: int, seed: int) -> list[OpponentProgram]:
        """Deterministic opponent sampling — same seed + same archive → same result.

        Used for re-evaluation fingerprinting: replaying the seed against the
        current archive reveals whether the sampled opponent set has changed.

        Falls back to uniform sampling if any fitness is non-finite.
        """
        now = time.monotonic()
        if not self._cache or (now - self._cache_time) > self._cache_ttl:
            await self._refresh_cache()
            self._cache_time = now

        if not self._cache:
            return []

        if len(self._cache) <= n:
            return list(self._cache)

        fitnesses = [o.fitness for o in self._cache]
        rng = np.random.default_rng(seed)

        if not all(np.isfinite(f) for f in fitnesses):
            indices = rng.choice(len(self._cache), size=n, replace=False)
            return [self._cache[int(i)] for i in indices]

        weights = _softmax_weights(fitnesses)
        indices = rng.choice(
            len(self._cache),
            size=n,
            replace=False,
            p=weights,
        )
        return [self._cache[int(i)] for i in indices]

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        """Fetch up to n opponents using softmax fitness-proportional sampling.

        Sampling mirrors ``FitnessProportionalEliteSelector``: fitnesses are
        min-max normalised to [0, 1], softmax is applied with auto-temperature
        (``max(std(normalised), 0.01)``), and selection is without replacement.
        This keeps opponent sampling consistent with parent selection.
        """
        now = time.monotonic()
        if not self._cache or (now - self._cache_time) > self._cache_ttl:
            await self._refresh_cache()
            self._cache_time = now

        if not self._cache:
            return []

        if len(self._cache) <= n:
            return list(self._cache)

        fitnesses = [o.fitness for o in self._cache]
        if not all(np.isfinite(f) for f in fitnesses):
            logger.warning(
                "[OpponentProvider] non-finite fitness detected; falling back to uniform sampling"
            )
            return random.sample(self._cache, n)

        weights = _softmax_weights(fitnesses)
        return weighted_sample_without_replacement(self._cache, weights, n)

    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        now = time.monotonic()
        if not self._cache or (now - self._cache_time) > self._cache_ttl:
            await self._refresh_cache()
            self._cache_time = now
        # Two-key sort: primary on fitness, secondary on program_id for
        # deterministic tiebreak. Without the id key, Python's stable sort
        # would preserve cache-insertion order — and that order depends on
        # Redis HVALS iteration which is not guaranteed stable across
        # cache refreshes. F25 in FAILURE_MODES.md.
        if higher_is_better:
            return sorted(self._cache, key=lambda o: (-o.fitness, o.program_id))[:k]
        return sorted(self._cache, key=lambda o: (o.fitness, o.program_id))[:k]

    async def close(self) -> None:
        """Close all Redis connections."""
        for client in self._redis_clients.values():
            await client.close()

    async def _refresh_cache(self) -> None:
        """Read all opponent programs from all source archives."""
        opponents: list[OpponentProgram] = []
        archive_key = f"island_{self._island_id}:archive"

        for db, prefix in self._sources:
            try:
                r = self._get_redis(db)
                # Archive: cell -> program_id
                program_ids = await r.hvals(archive_key)
                if not program_ids:
                    logger.debug(
                        "[OpponentProvider] empty archive db={} key={}",
                        db,
                        archive_key,
                    )
                    continue

                # Fetch programs in bulk via pipeline
                pipe = r.pipeline(transaction=False)
                for pid in program_ids:
                    pipe.get(f"{prefix}:program:{pid}")
                raw_programs = await pipe.execute()

                for pid, raw in zip(program_ids, raw_programs):
                    if raw is None:
                        continue
                    try:
                        data = json.loads(raw)
                        code = data.get("code", "")
                        metrics = data.get("metrics", {})
                        fitness = float(metrics.get("fitness", 0.0))
                        if code:
                            opponents.append(
                                OpponentProgram(
                                    program_id=pid,
                                    code=code,
                                    fitness=fitness,
                                )
                            )
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            "[OpponentProvider] failed to parse program {}: {}",
                            pid,
                            e,
                        )
            except Exception as exc:
                logger.warning("[OpponentProvider] error reading db={}: {}", db, exc)

        self._cache = opponents
        logger.debug(
            "[OpponentProvider] refreshed: {} opponents from {} sources",
            len(opponents),
            len(self._sources),
        )

    async def get_programs_by_ids(self, ids: list[str]) -> list[OpponentProgram]:
        """Return OpponentProgram objects for the given IDs, refreshing cache on miss.

        Used by post-step hooks (e.g. CompositionInjectionHook) that may run with
        a stale or empty cache (the cache is populated by FetchOpponentIdsStage
        inside the DAG; the hook runs *between* DAG steps). On any miss we
        force a full _refresh_cache() — the archive is small enough that this
        is cheap, and avoids silent injection failures.
        """
        id_set = set(ids)
        hits = [o for o in self._cache if o.program_id in id_set]
        if len(hits) == len(id_set):
            return hits
        await self._refresh_cache()
        self._cache_time = time.monotonic()
        return [o for o in self._cache if o.program_id in id_set]

    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        """Return codes from the in-memory cache for the given IDs.

        No Redis round-trips — reads the cache populated by the most recent
        ``get_opponents()`` call (guaranteed fresh since FetchOpponentIdsStage
        runs before FetchOpponentResultsStage in the same DAG execution).

        IDs not found in the cache are silently skipped.
        """
        id_set = set(ids)
        return [o.code for o in self._cache if o.program_id in id_set]


class CellStratifiedRedisOpponentArchiveProvider(RedisOpponentArchiveProvider):
    """Role-specific top-K opponent selection, stratified by BD cell.

    Reads from the production archive schema written by ``RedisArchiveStorage``:

      - ``island_{island_id}:archive`` HASH — field=cell_field → program_id
      - ``{prefix}:program:{pid}`` STRING (JSON payload) — program data

    Sorts opponents by ``metrics[fitness_key]`` (role-specific: G samples D by
    ``mean_improvement``; D samples G by ``quality``) rather than the parent's
    hardcoded ``metrics.fitness``. Because ``RedisArchiveStorage`` enforces
    exactly one program per cell, the top-K-by-fitness_key view already yields
    K distinct cells; the cell_field tracking below defends that invariant
    against future schema changes.

    The ``fitness_key`` value rides on ``OpponentProgram.fitness`` so downstream
    consumers (fitness-proportional sampling, ``get_top_k``) rank by the
    role-specific metric without further plumbing.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        prefix: str,
        fitness_key: str,
        k: int = 3,
        higher_is_better: bool = True,
        island_id: str = "fitness_island",
        cache_ttl: float = 30.0,
        **_ignored: object,
    ):
        # Single-DB adapter for CellStratified (vs RedisOpponentArchiveProvider's multi-DB).
        # `**_ignored` swallows legacy Hydra keys (e.g. sources) that merge in from the
        # inherited `adversarial_coevo.yaml` pipeline config.
        sources = [{"db": db, "prefix": prefix}]
        super().__init__(
            host=host,
            port=port,
            sources=sources,
            island_id=island_id,
            cache_ttl=cache_ttl,
        )
        self._fitness_key = fitness_key
        self._k = k
        self._higher_is_better = higher_is_better
        self._db = db
        self._prefix = prefix
        # Populated by _refresh_cache: program_id -> cell_field (e.g. "7,5")
        self._cell_fields: dict[str, str] = {}
        # Tracks the elite-id set returned by the previous get_top_k call, so
        # HOF_ROTATE can fire on composition change. None = no prior fetch yet.
        self._last_hof_elite_ids: frozenset[str] | None = None

    def _get_redis(self, db: int) -> aioredis.Redis:  # type: ignore[type-arg]
        """Get Redis client, checking for test-injected _redis first."""
        if hasattr(self, "_redis"):
            return self._redis  # type: ignore[return-value]
        return super()._get_redis(db)

    async def _refresh_cache(self) -> None:
        """Override: sort by role-specific fitness_key, track cell_field per program.

        Schema exactly matches ``RedisArchiveStorage``:
          - ``island_{island_id}:archive`` HASH — cell_field → program_id
          - ``{prefix}:program:{pid}`` STRING — JSON payload with ``.code`` and
            ``.metrics[fitness_key]``.

        Programs missing ``metrics[fitness_key]`` are silently skipped — this
        happens during early-gen warmup where some eval stages have not yet
        emitted the role-specific metric for every archived program.
        """
        opponents: list[OpponentProgram] = []
        cell_map: dict[str, str] = {}
        archive_key = f"island_{self._island_id}:archive"

        for db, prefix in self._sources:
            try:
                r = self._get_redis(db)
                # HGETALL: field=cell_field -> program_id
                cell_to_pid = await r.hgetall(archive_key)
                if not cell_to_pid:
                    logger.debug(
                        "[CellStratifiedOpponentProvider] empty archive db={} key={}",
                        db,
                        archive_key,
                    )
                    continue

                items = list(cell_to_pid.items())
                pipe = r.pipeline(transaction=False)
                for _cell_field, pid in items:
                    pipe.get(f"{prefix}:program:{pid}")
                raw_programs = await pipe.execute()

                for (cell_field, pid), raw in zip(items, raw_programs):
                    if raw is None:
                        continue
                    try:
                        data = json.loads(raw)
                        code = data.get("code", "")
                        metrics = data.get("metrics", {})
                        fitness_value = metrics.get(self._fitness_key)
                        if fitness_value is None or code == "":
                            continue
                        fitness = float(fitness_value)
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            "[CellStratifiedOpponentProvider] parse error pid={} err={}",
                            pid,
                            e,
                        )
                        continue

                    opponents.append(
                        OpponentProgram(
                            program_id=pid,
                            code=code,
                            fitness=fitness,
                        )
                    )
                    cell_map[pid] = cell_field
            except Exception as exc:
                logger.warning(
                    "[CellStratifiedOpponentProvider] error reading db={}: {}", db, exc
                )

        self._cache = opponents
        self._cell_fields = cell_map
        logger.debug(
            "[CellStratifiedOpponentProvider] refreshed: {} opponents, "
            "{} distinct cells, fitness_key={}",
            len(opponents),
            len(set(cell_map.values())),
            self._fitness_key,
        )

    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        """Return up to k opponents, one elite per distinct BD cell.

        Sort: role-specific ``fitness_key`` descending (if ``higher_is_better``)
        with ``program_id`` as deterministic tiebreak.

        Distinct-cell constraint is enforced via ``cell_field`` lookup; with
        ``RedisArchiveStorage`` enforcing 1-per-cell, this is trivially satisfied
        but remains a defensive invariant.
        """
        now = time.monotonic()
        if not self._cache or (now - self._cache_time) > self._cache_ttl:
            await self._refresh_cache()
            self._cache_time = now

        effective_higher = self._higher_is_better and higher_is_better

        if not self._cache:
            emit_hof_fetch(
                label=f"db{self._db}:{self._prefix}",
                n_elites=0,
                fitness_key=self._fitness_key,
                k_requested=int(k),
                cells_populated=0,
            )
            return []

        if effective_higher:
            sorted_opponents = sorted(
                self._cache, key=lambda o: (-o.fitness, o.program_id)
            )
        else:
            sorted_opponents = sorted(
                self._cache, key=lambda o: (o.fitness, o.program_id)
            )

        picked: list[OpponentProgram] = []
        seen_cells: set[str] = set()
        for op in sorted_opponents:
            cell_field = self._cell_fields.get(op.program_id, op.program_id)
            if cell_field in seen_cells:
                continue
            picked.append(op)
            seen_cells.add(cell_field)
            emit_cell_pick(
                label=f"db{self._db}:{self._prefix}",
                cell_id=cell_field,
                program_id=op.program_id,
                fitness_key=self._fitness_key,
                fitness_value=float(op.fitness),
            )
            if len(picked) >= k:
                break

        emit_hof_fetch(
            label=f"db{self._db}:{self._prefix}",
            n_elites=len(picked),
            fitness_key=self._fitness_key,
            k_requested=int(k),
            cells_populated=len(seen_cells),
        )

        # HOF_ROTATE: emit when the elite-id set differs from the previous
        # non-empty fetch. Tracks composition change (not just size change),
        # so a 1-for-1 swap still counts as a rotation.
        new_elite_ids = frozenset(o.program_id for o in picked)
        if new_elite_ids and new_elite_ids != self._last_hof_elite_ids:
            old_size = (
                0 if self._last_hof_elite_ids is None else len(self._last_hof_elite_ids)
            )
            emit_hof_rotate(
                label=f"db{self._db}:{self._prefix}",
                old_hof_size=old_size,
                new_hof_size=len(new_elite_ids),
                fitness_key=self._fitness_key,
            )
            self._last_hof_elite_ids = new_elite_ids

        return picked
