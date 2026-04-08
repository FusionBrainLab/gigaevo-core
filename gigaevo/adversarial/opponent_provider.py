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
from redis import asyncio as aioredis  # noqa: I001


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

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        """Fetch up to n opponents, fitness-proportional sampling."""
        now = time.monotonic()
        if not self._cache or (now - self._cache_time) > self._cache_ttl:
            await self._refresh_cache()
            self._cache_time = now

        if not self._cache:
            return []

        if len(self._cache) <= n:
            return list(self._cache)

        # Fitness-proportional sampling (shift to positive)
        fitnesses = [max(o.fitness, 0.0) for o in self._cache]
        total = sum(fitnesses)
        if total <= 0:
            return random.sample(self._cache, n)
        weights = [f / total for f in fitnesses]
        indices: set[int] = set()
        attempts = 0
        while len(indices) < n and attempts < n * 10:
            idx = random.choices(range(len(self._cache)), weights=weights, k=1)[0]
            indices.add(idx)
            attempts += 1
        return [self._cache[i] for i in indices]

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
