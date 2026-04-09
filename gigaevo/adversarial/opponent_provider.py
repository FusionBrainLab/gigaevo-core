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

    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        """Return codes from the in-memory cache for the given IDs.

        No Redis round-trips — reads the cache populated by the most recent
        ``get_opponents()`` call (guaranteed fresh since FetchOpponentIdsStage
        runs before FetchOpponentResultsStage in the same DAG execution).

        IDs not found in the cache are silently skipped.
        """
        id_set = set(ids)
        return [o.code for o in self._cache if o.program_id in id_set]
