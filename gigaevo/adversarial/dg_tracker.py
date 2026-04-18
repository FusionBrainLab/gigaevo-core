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
import json

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
    _GLOBAL_PAIRS_KEY_TEMPLATE = "{prefix}:dg_best_pairs"
    _INJECTED_PAIRS_KEY_TEMPLATE = "{prefix}:dg_injected_pairs"
    # v3 inverted indices for MAP-Elites BD axes + shared-benchmark lineage.
    _D_WINS_KEY_TEMPLATE = "{prefix}:dg_d_wins:{d_id}"  # SET of g_ids D has beaten
    _G_RESISTED_KEY_TEMPLATE = (
        "{prefix}:dg_g_resisted:{g_id}"  # SET of d_ids G has resisted
    )
    _D_DELTA_KEY_TEMPLATE = (
        "{prefix}:dg_delta:{d_id}"  # HASH g_id -> raw delta (any sign)
    )

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

    def _global_pairs_key(self) -> str:
        return self._GLOBAL_PAIRS_KEY_TEMPLATE.format(prefix=self._prefix)

    def _injected_pairs_key(self) -> str:
        return self._INJECTED_PAIRS_KEY_TEMPLATE.format(prefix=self._prefix)

    def _d_wins_key(self, d_id: str) -> str:
        return self._D_WINS_KEY_TEMPLATE.format(prefix=self._prefix, d_id=d_id)

    def _g_resisted_key(self, g_id: str) -> str:
        return self._G_RESISTED_KEY_TEMPLATE.format(prefix=self._prefix, g_id=g_id)

    def _d_delta_key(self, d_id: str) -> str:
        return self._D_DELTA_KEY_TEMPLATE.format(prefix=self._prefix, d_id=d_id)

    async def is_pair_injected(self, d_id: str, g_id: str) -> bool:
        """Return True if (D, G) pair has already been composed and injected.

        Permanent dedup — no TTL, never repeat composition for the same pair.
        """
        member = f"{d_id}|{g_id}"
        return bool(await self._redis.sismember(self._injected_pairs_key(), member))

    async def mark_pair_injected(self, d_id: str, g_id: str) -> None:
        """Permanently mark (D, G) pair as injected. No TTL — never repeat."""
        member = f"{d_id}|{g_id}"
        await self._redis.sadd(self._injected_pairs_key(), member)
        logger.info(
            "[DGTracker] marked pair injected d={} g={} (permanent dedup)",
            d_id,
            g_id,
        )

    async def count_injected_pairs(self) -> int:
        """Return the number of permanently injected (D, G) pairs."""
        return int(await self._redis.scard(self._injected_pairs_key()))

    async def record_improvement(self, d_id: str, g_id: str, delta: float) -> None:
        """Record that D improved G by delta. Only positive deltas are stored.

        Dual-writes to both per-G sorted set and global best-pairs sorted set.
        """
        if delta <= 0:
            return
        key = self._key(g_id)
        await self._redis.zadd(key, {d_id: delta}, gt=True)
        await self._redis.expire(key, self._ttl)
        # Dual-write to global best-pairs sorted set with composite key "d_id|g_id"
        global_key = self._global_pairs_key()
        pair_member = f"{d_id}|{g_id}"
        await self._redis.zadd(global_key, {pair_member: delta}, gt=True)
        await self._redis.expire(global_key, self._ttl)
        logger.info(
            "[DGTracker] record_improvement d={} g={} delta={:.6f}",
            d_id,
            g_id,
            delta,
        )

    async def record_batch(
        self,
        pairs: list[tuple[str, str, float]],
        *,
        gen: int | None = None,
    ) -> int:
        """Record multiple D-G improvement pairs via a single Redis pipeline.

        Writes to FIVE key families in one round trip (v3):
          - Per-G sorted set ``dg_improvements:{g_id}``     (positive deltas only)
          - Global best-pairs sorted set ``dg_best_pairs``  (positive deltas only)
          - D-keyed SET ``dg_d_wins:{d_id}``                (positive deltas)
          - G-keyed SET ``dg_g_resisted:{g_id}``            (non-positive deltas)
          - D-keyed HASH ``dg_delta:{d_id}``                (every pair, any sign)

        The D-delta hash is the substrate for ``SharedBenchmarkLineageStage``
        (§3.5 Prong 2). The D-wins / G-resisted SETs are the BD y-axes.

        Emits a ``[TRACKER_WRITE]`` structured JSON log line so post-hoc log
        audit can reconstruct exactly what was persisted without re-reading
        Redis (§13 log-based verification contract).

        Returns the number of positive pairs (legacy contract — unchanged).
        """
        if not pairs:
            return 0

        pipe = self._redis.pipeline(transaction=False)
        global_key = self._global_pairs_key()
        global_members: dict[str, float] = {}

        d_wins: dict[str, set[str]] = {}
        g_resisted: dict[str, set[str]] = {}
        d_deltas: dict[str, dict[str, str]] = {}

        positive_count = 0
        for d_id, g_id, delta in pairs:
            d_val = float(delta)
            # Raw delta hash captures every pair, regardless of sign.
            d_deltas.setdefault(d_id, {})[g_id] = repr(d_val)
            if d_val > 0:
                positive_count += 1
                key = self._key(g_id)
                pipe.zadd(key, {d_id: d_val}, gt=True)
                pipe.expire(key, self._ttl)
                global_members[f"{d_id}|{g_id}"] = d_val
                d_wins.setdefault(d_id, set()).add(g_id)
            else:
                g_resisted.setdefault(g_id, set()).add(d_id)

        if global_members:
            pipe.zadd(global_key, global_members, gt=True)
            pipe.expire(global_key, self._ttl)

        for d_id, g_set in d_wins.items():
            key = self._d_wins_key(d_id)
            pipe.sadd(key, *g_set)
            pipe.expire(key, self._ttl)
        for g_id, d_set in g_resisted.items():
            key = self._g_resisted_key(g_id)
            pipe.sadd(key, *d_set)
            pipe.expire(key, self._ttl)
        for d_id, delta_map in d_deltas.items():
            key = self._d_delta_key(d_id)
            pipe.hset(key, mapping=delta_map)
            pipe.expire(key, self._ttl)

        await pipe.execute()

        payload = {
            "event": "TRACKER_WRITE",
            "gen": gen,
            "pairs_count": len(pairs),
            "positive_count": positive_count,
            "d_wins_added": sum(len(v) for v in d_wins.values()),
            "g_resisted_added": sum(len(v) for v in g_resisted.values()),
            "d_faced_added": sum(len(v) for v in d_deltas.values()),
        }
        logger.info("[TRACKER_WRITE] {}", json.dumps(payload))
        return positive_count

    async def count_g_beaten_by_d(self, d_id: str) -> int:
        """Number of distinct G programs this D has strictly improved (delta > 0).

        Source for D's BD y-axis ``tracker_coverage_count`` (§3.3).
        """
        return int(await self._redis.scard(self._d_wins_key(d_id)))

    async def count_d_resisted_by_g(self, g_id: str) -> int:
        """Number of distinct D programs this G has resisted (delta <= 0).

        Source for G's fallback BD y-axis ``g_tracker_coverage_count`` (§3.2).
        """
        return int(await self._redis.scard(self._g_resisted_key(g_id)))

    async def faced_by_d(self, d_id: str) -> set[str]:
        """Set of G program IDs this D has been evaluated against (any outcome).

        Substrate for ``SharedBenchmarkResolver`` intersection (§3.5 Prong 2).
        """
        keys = await self._redis.hkeys(self._d_delta_key(d_id))
        return set(keys)

    async def get_deltas_against(
        self, d_a: str, d_b: str, g_ids: list[str]
    ) -> list[tuple[float, float]]:
        """For each g_id, return (delta_a, delta_b). Pairs missing on either side are skipped.

        Consumed by ``SharedBenchmarkLineageStage`` to compute
        ``mean(delta_child) - mean(delta_parent)`` over the intersection
        of the two D's benchmark histories.
        """
        if not g_ids:
            return []
        g_list = list(g_ids)
        deltas_a = await self._redis.hmget(self._d_delta_key(d_a), g_list)
        deltas_b = await self._redis.hmget(self._d_delta_key(d_b), g_list)
        paired: list[tuple[float, float]] = []
        for da, db in zip(deltas_a, deltas_b):
            if da is None or db is None:
                continue
            paired.append((float(da), float(db)))
        return paired

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

    async def get_best_pairs(self, k: int = 5) -> list[tuple[str, str, float]]:
        """Return the top-k (D, G) pairs globally by improvement delta.

        Returns a list of (d_id, g_id, delta) tuples.
        Reads from the global best-pairs sorted set (dual-written during record).
        """
        global_key = self._global_pairs_key()
        results = await self._redis.zrevrange(global_key, 0, k - 1, withscores=True)
        pairs = []
        for member, delta in results:
            # member is formatted as "d_id|g_id"
            parts = member.split("|", 1)
            if len(parts) == 2:
                d_id, g_id = parts
                pairs.append((d_id, g_id, float(delta)))
        return pairs

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.close()
