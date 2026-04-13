"""D-G improvement pair tracker for adversarial co-evolution.

Stores per-pair improvement deltas in Redis sorted sets. Each G program
has a sorted set mapping D program IDs to their improvement deltas.
GradientInPromptStage queries this to find the best D for a specific G.

Redis key pattern:
    {prefix}:dg_improvements:{g_id}  ->  sorted set (member=d_id, score=delta)

Only positive deltas (D actually improved G) are stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from redis import asyncio as aioredis


@dataclass
class DGImprovement:
    """A recorded improvement of G by D."""

    d_id: str
    g_id: str
    delta: float


class DGImprovementTracker:
    """Tracks which D programs improved which G programs and by how much.

    Uses Redis sorted sets keyed by G program ID. Each sorted set maps
    D program IDs to their improvement deltas (higher = better for D).

    Args:
        host: Redis host
        port: Redis port
        db: Redis DB number (should be G's DB since improvements are G-centric)
        prefix: Redis key prefix (should be G's prefix)
        ttl_seconds: TTL for improvement records (default: 86400 = 24 hours).
            Prevents unbounded growth as programs cycle out of the archive.
    """

    _KEY_TEMPLATE = "{prefix}:dg_improvements:{g_id}"

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        prefix: str,
        ttl_seconds: int = 86400,
    ):
        self._redis: aioredis.Redis = aioredis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, g_id: str) -> str:
        return self._KEY_TEMPLATE.format(prefix=self._prefix, g_id=g_id)

    async def record_improvement(self, d_id: str, g_id: str, delta: float) -> None:
        """Record that D improved G by delta. Only positive deltas are stored."""
        if delta <= 0:
            return
        key = self._key(g_id)
        await self._redis.zadd(key, {d_id: delta}, gt=True)
        await self._redis.expire(key, self._ttl)
        logger.debug("[DGTracker] recorded d={} g={} delta={:.6f}", d_id, g_id, delta)

    async def record_batch(self, pairs: list[tuple[str, str, float]]) -> int:
        """Record multiple D-G improvement pairs via Redis pipeline.

        Returns the number of positive pairs actually recorded.
        """
        positive = [(d, g, delta) for d, g, delta in pairs if delta > 0]
        if not positive:
            return 0
        pipe = self._redis.pipeline(transaction=False)
        for d_id, g_id, delta in positive:
            key = self._key(g_id)
            pipe.zadd(key, {d_id: delta}, gt=True)
            pipe.expire(key, self._ttl)
        await pipe.execute()
        logger.debug("[DGTracker] batch recorded {} positive pairs", len(positive))
        return len(positive)

    async def get_best_d_for_g(self, g_id: str) -> tuple[str, float] | None:
        """Return the D with the highest improvement delta for this G.

        Returns (d_id, delta) or None if no data exists.
        """
        key = self._key(g_id)
        results = await self._redis.zrevrange(key, 0, 0, withscores=True)
        if not results:
            return None
        d_id, delta = results[0]
        return (d_id, delta)

    async def get_top_d_for_g(self, g_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k D programs by improvement delta for this G."""
        key = self._key(g_id)
        results = await self._redis.zrevrange(key, 0, k - 1, withscores=True)
        return [(d_id, delta) for d_id, delta in results]

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.close()
