"""Redis-coordinated least-loaded endpoint pool.

Multiple GigaEvo runs share a set of LLM server endpoints.  ``EndpointPool``
tracks in-flight request counts per endpoint in a Redis hash (DB 15) and
routes new requests to the least-loaded healthy endpoint.

Redis key schema (all on the configured DB, default 15)::

    llm_pool:{pool}:inflight            Hash  {endpoint → in-flight count}
    llm_pool:{pool}:cooldown:{url_hash} String with TTL (marks unhealthy)
    llm_pool:{pool}:stats:{url_hash}    Hash  {requests, errors, total_latency_ms}
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import random
import threading
import time
from typing import Any

from loguru import logger
import redis
import redis.asyncio as aioredis

_DEFAULT_REDIS_URL = "redis://localhost:6379/15"
_DEFAULT_COOLDOWN_SECS = 60


def _url_hash(url: str) -> str:
    """Short hash of an endpoint URL for use in Redis keys."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


class EndpointPool:
    """Redis-coordinated least-loaded endpoint selector.

    Each endpoint is identified by its full URL (e.g.
    ``http://localhost:8000/v1``).  In-flight counts are tracked in a
    shared Redis hash so that concurrent runs (separate processes) see each
    other's load.

    Provides both sync and async APIs — sync for ``MultiModelRouter._select()``
    (which is called from a sync context), async for ``ainvoke`` paths.
    """

    def __init__(
        self,
        pool_name: str,
        endpoints: list[str],
        redis_url: str = _DEFAULT_REDIS_URL,
        cooldown_secs: int = _DEFAULT_COOLDOWN_SECS,
    ) -> None:
        if not endpoints:
            raise ValueError("endpoints must be non-empty")
        self._pool_name = pool_name
        self._endpoints = list(endpoints)
        self._cooldown_secs = cooldown_secs
        self._redis_url = redis_url

        # Keys
        self._inflight_key = f"llm_pool:{pool_name}:inflight"

        # Local in-memory counters for fast-path selection (no Redis RT).
        # Redis is updated in background for cross-run visibility.
        self._local_inflight: dict[str, int] = {ep: 0 for ep in endpoints}
        self._local_cooldown: set[str] = set()
        self._lock = threading.Lock()

        # Lazy-init Redis clients (avoids connecting at import time)
        self._sync_redis: redis.Redis | None = None
        self._async_redis: aioredis.Redis | None = None

        logger.info(
            "[EndpointPool:{}] Initialized with {} endpoints",
            pool_name,
            len(endpoints),
        )

    # ------------------------------------------------------------------
    # Redis client helpers
    # ------------------------------------------------------------------

    def _get_sync(self) -> redis.Redis:
        if self._sync_redis is None:
            self._sync_redis = redis.Redis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._sync_redis

    def _get_async(self) -> aioredis.Redis:
        if self._async_redis is None:
            self._async_redis = aioredis.Redis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._async_redis

    def _cooldown_key(self, endpoint: str) -> str:
        return f"llm_pool:{self._pool_name}:cooldown:{_url_hash(endpoint)}"

    def _stats_key(self, endpoint: str) -> str:
        return f"llm_pool:{self._pool_name}:stats:{_url_hash(endpoint)}"

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def acquire(self) -> str:
        """Pick the least-loaded healthy endpoint and increment its count.

        Selection uses local in-memory state (zero Redis latency).  Redis is
        updated asynchronously for cross-run visibility.
        """
        endpoint = self._local_select()
        # Fire-and-forget Redis update for cross-run visibility
        r = self._get_async()
        await r.hincrby(self._inflight_key, endpoint, 1)
        return endpoint

    async def release(self, endpoint: str, latency_ms: float = 0.0) -> None:
        """Decrement in-flight count and update stats."""
        with self._lock:
            self._local_inflight[endpoint] = max(0, self._local_inflight[endpoint] - 1)
        r = self._get_async()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        stats_key = self._stats_key(endpoint)
        pipe.hincrby(stats_key, "requests", 1)
        pipe.hincrbyfloat(stats_key, "total_latency_ms", latency_ms)
        await pipe.execute()

    async def mark_unhealthy(self, endpoint: str) -> None:
        """Set a cooldown key so this endpoint is skipped for a while."""
        with self._lock:
            self._local_inflight[endpoint] = max(0, self._local_inflight[endpoint] - 1)
            self._local_cooldown.add(endpoint)
        r = self._get_async()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        pipe.set(self._cooldown_key(endpoint), "1", ex=self._cooldown_secs)
        pipe.hincrby(self._stats_key(endpoint), "errors", 1)
        await pipe.execute()

    async def get_stats(self) -> dict[str, dict[str, Any]]:
        """Read per-endpoint stats from Redis."""
        r = self._get_async()
        result: dict[str, dict[str, Any]] = {}
        inflight = await r.hgetall(self._inflight_key)
        for ep in self._endpoints:
            stats_raw = await r.hgetall(self._stats_key(ep))
            result[ep] = {
                "inflight": int(inflight.get(ep, 0)),
                "requests": int(stats_raw.get("requests", 0)),
                "errors": int(stats_raw.get("errors", 0)),
                "total_latency_ms": float(stats_raw.get("total_latency_ms", 0)),
                "healthy": not await r.exists(self._cooldown_key(ep)),
            }
        return result

    @asynccontextmanager
    async def use(self) -> AsyncIterator[str]:
        """Context manager: acquire on enter, release on exit."""
        endpoint = await self.acquire()
        t0 = time.perf_counter()
        try:
            yield endpoint
        except Exception:
            await self.mark_unhealthy(endpoint)
            raise
        else:
            latency_ms = (time.perf_counter() - t0) * 1000
            await self.release(endpoint, latency_ms)

    def _local_select(self) -> str:
        """Pick the least-loaded healthy endpoint using local state (no I/O)."""
        with self._lock:
            healthy = [ep for ep in self._endpoints if ep not in self._local_cooldown]
            if not healthy:
                healthy = list(self._endpoints)

            min_load = min(self._local_inflight.get(ep, 0) for ep in healthy)
            candidates = [
                ep for ep in healthy if self._local_inflight.get(ep, 0) == min_load
            ]
            endpoint = random.choice(candidates)
            self._local_inflight[endpoint] = self._local_inflight.get(endpoint, 0) + 1
            return endpoint

    async def _select_endpoint_async(self, r: aioredis.Redis) -> str:
        """Pick the least-loaded healthy endpoint (async).

        Uses a single pipelined round trip for HGETALL + all EXISTS checks.
        """
        pipe = r.pipeline(transaction=False)
        pipe.hgetall(self._inflight_key)
        for ep in self._endpoints:
            pipe.exists(self._cooldown_key(ep))
        results = await pipe.execute()

        inflight = results[0]  # dict from HGETALL
        cooldowns = results[1:]  # list of int (0 or 1) from EXISTS

        healthy = [ep for ep, cd in zip(self._endpoints, cooldowns) if not cd]

        if not healthy:
            logger.warning(
                "[EndpointPool:{}] All endpoints in cooldown — using all",
                self._pool_name,
            )
            healthy = list(self._endpoints)

        # Pick the one with fewest in-flight requests (random tiebreak)
        min_load = min(int(inflight.get(ep, 0)) for ep in healthy)
        candidates = [ep for ep in healthy if int(inflight.get(ep, 0)) == min_load]
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # Sync API (for MultiModelRouter._select which is sync)
    # ------------------------------------------------------------------

    def acquire_sync(self) -> str:
        """Sync version of acquire."""
        endpoint = self._local_select()
        r = self._get_sync()
        r.hincrby(self._inflight_key, endpoint, 1)
        return endpoint

    def release_sync(self, endpoint: str, latency_ms: float = 0.0) -> None:
        """Sync version of release."""
        with self._lock:
            self._local_inflight[endpoint] = max(0, self._local_inflight[endpoint] - 1)
        r = self._get_sync()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        stats_key = self._stats_key(endpoint)
        pipe.hincrby(stats_key, "requests", 1)
        pipe.hincrbyfloat(stats_key, "total_latency_ms", latency_ms)
        pipe.execute()

    def mark_unhealthy_sync(self, endpoint: str) -> None:
        """Sync version of mark_unhealthy."""
        with self._lock:
            self._local_inflight[endpoint] = max(0, self._local_inflight[endpoint] - 1)
            self._local_cooldown.add(endpoint)
        r = self._get_sync()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        pipe.set(self._cooldown_key(endpoint), "1", ex=self._cooldown_secs)
        pipe.hincrby(self._stats_key(endpoint), "errors", 1)
        pipe.execute()

    def _select_endpoint_sync(self, r: redis.Redis) -> str:
        """Pick the least-loaded healthy endpoint (sync).

        Uses a single pipelined round trip for HGETALL + all EXISTS checks.
        """
        pipe = r.pipeline(transaction=False)
        pipe.hgetall(self._inflight_key)
        for ep in self._endpoints:
            pipe.exists(self._cooldown_key(ep))
        results = pipe.execute()

        inflight = results[0]
        cooldowns = results[1:]

        healthy = [ep for ep, cd in zip(self._endpoints, cooldowns) if not cd]

        if not healthy:
            logger.warning(
                "[EndpointPool:{}] All endpoints in cooldown — using all",
                self._pool_name,
            )
            healthy = list(self._endpoints)

        min_load = min(int(inflight.get(ep, 0)) for ep in healthy)
        candidates = [ep for ep in healthy if int(inflight.get(ep, 0)) == min_load]
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close Redis connections."""
        if self._async_redis is not None:
            await self._async_redis.aclose()
            self._async_redis = None
        if self._sync_redis is not None:
            self._sync_redis.close()
            self._sync_redis = None
