"""D-G improvement pair tracker for adversarial co-evolution.

Stores per-pair improvement deltas in Redis sorted sets. Each G program
has a sorted set mapping D program IDs to their improvement deltas.
GradientInPromptStage queries this to find the best D for a specific G.

Redis key pattern:
    {prefix}:dg_improvements:{g_id}  ->  sorted set (member=d_id, score=delta)

Only positive deltas (D actually improved G) are stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import cast

from loguru import logger
from redis import asyncio as aioredis

from gigaevo.adversarial.structured_logging import emit_tracker_write


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
    # v4: full per-opponent metrics dict replaces the old scalar-delta hash.
    _D_METRICS_KEY_TEMPLATE = (
        "{prefix}:dg_metrics:{d_id}"  # HASH g_id -> JSON metrics dict
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

    def _d_metrics_key(self, d_id: str) -> str:
        return self._D_METRICS_KEY_TEMPLATE.format(prefix=self._prefix, d_id=d_id)

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

    async def record_metrics(
        self, d_id: str, g_id: str, metrics: dict[str, float]
    ) -> None:
        """Record a full per-opponent metrics dict for (D, G).

        Stores the metrics dict as JSON in the dg_metrics:{d_id} hash keyed
        by g_id. This is the v4 substrate — replaces the old scalar-delta hash.
        """
        key = self._d_metrics_key(d_id)
        await self._redis.hset(key, g_id, json.dumps(metrics))
        await self._redis.expire(key, self._ttl)

    async def metrics_by_d(self, d_id: str) -> dict[str, dict[str, float]]:
        """Return all per-G metrics dicts recorded for this D.

        Returns {g_id: {metric_name: value}} for every G this D has been
        evaluated against (via record_metrics).
        """
        raw = await self._redis.hgetall(self._d_metrics_key(d_id))
        return {g_id: json.loads(payload) for g_id, payload in raw.items()}

    async def record_batch(
        self,
        pairs: Sequence[tuple[str, str, Mapping[str, float]]],
        *,
        gen: int | None = None,
    ) -> int:
        """Record multiple D-G per-pair metric dicts via a single Redis pipeline.

        Writes to FIVE key families in one round trip:
          - Per-G sorted set ``dg_improvements:{g_id}``     (positive deltas only)
          - Global best-pairs sorted set ``dg_best_pairs``  (positive deltas only)
          - D-keyed SET ``dg_d_wins:{d_id}``                (positive deltas)
          - G-keyed SET ``dg_g_resisted:{g_id}``            (non-positive deltas)
          - D-keyed HASH ``dg_metrics:{d_id}``              (every pair, any sign)

        Schema-agnostic: the per-pair ``Mapping[str, float]`` is stored verbatim
        (JSON-serialised) in the ``dg_metrics`` hash. The only required key is
        ``"delta"``, which drives positive/non-positive routing into the
        ``dg_d_wins`` / ``dg_g_resisted`` inverted indices.

        The D-metrics hash is the substrate for ``SharedBenchmarkFilteredLineageStage``
        (§3.5 Prong 2). The D-wins / G-resisted SETs are the BD y-axes.

        Emits a ``[TRACKER_WRITE]`` structured JSON log line so post-hoc log
        audit can reconstruct exactly what was persisted without re-reading
        Redis (§13 log-based verification contract).

        Returns the number of positive pairs (delta > 0).
        """
        if not pairs:
            return 0

        pipe = self._redis.pipeline(transaction=False)
        global_key = self._global_pairs_key()
        global_members: dict[str, float] = {}

        d_wins: dict[str, set[str]] = {}
        g_resisted: dict[str, set[str]] = {}
        d_metrics: dict[str, dict[str, str]] = {}

        positive_count = 0
        for d_id, g_id, per_pair in pairs:
            # Store the dict verbatim (schema-agnostic). Only the "delta" field
            # is required — it drives positive/non-positive routing.
            record = dict(per_pair)
            d_val = float(record["delta"])
            d_metrics.setdefault(d_id, {})[g_id] = json.dumps(record)
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
            pipe.zadd(
                global_key,
                cast("Mapping[str | bytes, bytes | float | int | str]", global_members),
                gt=True,
            )
            pipe.expire(global_key, self._ttl)

        for d_id, g_set in d_wins.items():
            key = self._d_wins_key(d_id)
            pipe.sadd(key, *g_set)
            pipe.expire(key, self._ttl)
        for g_id, d_set in g_resisted.items():
            key = self._g_resisted_key(g_id)
            pipe.sadd(key, *d_set)
            pipe.expire(key, self._ttl)
        for d_id, metrics_map in d_metrics.items():
            key = self._d_metrics_key(d_id)
            pipe.hset(
                key,
                mapping=cast(
                    "Mapping[str | bytes, bytes | float | int | str]", metrics_map
                ),
            )
            pipe.expire(key, self._ttl)

        await pipe.execute()

        emit_tracker_write(
            pairs_count=len(pairs),
            positive_count=positive_count,
            d_wins_added=sum(len(v) for v in d_wins.values()),
            g_resisted_added=sum(len(v) for v in g_resisted.values()),
            d_faced_added=sum(len(v) for v in d_metrics.values()),
            gen=gen,
        )
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

        Substrate for ``SharedBenchmarkFilteredLineageStage`` intersection (§3.5 Prong 2).
        Reads from the v4 dg_metrics hash.
        """
        keys = await self._redis.hkeys(self._d_metrics_key(d_id))
        return set(keys)

    async def get_deltas_against(
        self, d_a: str, d_b: str, g_ids: list[str]
    ) -> list[tuple[float, float]]:
        """For each g_id, return (delta_a, delta_b). Pairs missing on either side are skipped.

        Consumed by ``SharedBenchmarkFilteredLineageStage`` to compute
        ``mean(delta_child) - mean(delta_parent)`` over the intersection
        of the two D's benchmark histories.
        Reads from the dg_metrics hash (``delta`` field).
        """
        if not g_ids:
            return []
        g_list = list(g_ids)
        raw_a = await self._redis.hmget(self._d_metrics_key(d_a), g_list)
        raw_b = await self._redis.hmget(self._d_metrics_key(d_b), g_list)
        paired: list[tuple[float, float]] = []
        for ra, rb in zip(raw_a, raw_b):
            if ra is None or rb is None:
                continue
            da = float(json.loads(ra)["delta"])
            db = float(json.loads(rb)["delta"])
            paired.append((da, db))
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
