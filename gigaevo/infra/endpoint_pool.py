"""Redis-coordinated least-loaded endpoint pool.

Multiple GigaEvo runs share a set of LLM server endpoints.  ``EndpointPool``
tracks in-flight request counts per endpoint in a Redis hash (DB 15) and
routes new requests to the least-loaded healthy endpoint.

All selection uses **sync Redis** (0.05ms per call on localhost) with a Lua
script for atomic read-select-increment.  This gives full cross-run visibility
with zero race conditions and negligible overhead vs 30-60s LLM calls.

Redis key schema (all on the configured DB, default 15)::

    llm_pool:{pool}:inflight            Hash  {endpoint → in-flight count}
    llm_pool:{pool}:cooldown:{url_hash} String with TTL (marks unhealthy)
    llm_pool:{pool}:stats:{url_hash}    Hash  {requests, errors, total_latency_ms}
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import time
from typing import Any

from loguru import logger
import redis
import redis.asyncio as aioredis

_DEFAULT_REDIS_URL = "redis://localhost:6379/15"
_DEFAULT_COOLDOWN_SECS = 60

# Lua script: atomically read inflight counts, filter cooldowns, pick
# least-loaded, HINCRBY +1, return the selected endpoint.
# KEYS[1] = inflight hash key
# ARGV = list of endpoint URLs followed by their cooldown keys
# Layout: ARGV[1..N] = endpoint URLs, ARGV[N+1..2N] = cooldown keys
# Lua script: atomically select the endpoint with fewest inflight requests.
# Pure least-connections routing — no EMA latency, no weighted scoring.
# Ties broken randomly.
#
# KEYS[1] = inflight hash key
# ARGV layout: [ep1..epN, cd_key1..cd_keyN]
_LUA_ACQUIRE = """
local inflight_key = KEYS[1]
local n = #ARGV / 2

local best_inflight = math.huge
local candidates = {}

for i = 1, n do
    local ep = ARGV[i]
    local cd_key = ARGV[n + i]
    if redis.call('EXISTS', cd_key) == 0 then
        local inflight = tonumber(redis.call('HGET', inflight_key, ep) or '0')
        if inflight < best_inflight then
            best_inflight = inflight
            candidates = {ep}
        elseif inflight == best_inflight then
            candidates[#candidates + 1] = ep
        end
    end
end

-- Fallback: if all in cooldown, ignore cooldowns
if #candidates == 0 then
    for i = 1, n do
        local ep = ARGV[i]
        local inflight = tonumber(redis.call('HGET', inflight_key, ep) or '0')
        if inflight < best_inflight then
            best_inflight = inflight
            candidates = {ep}
        elseif inflight == best_inflight then
            candidates[#candidates + 1] = ep
        end
    end
end

-- Random tiebreak among candidates, then atomically increment
local best_ep = candidates[math.random(#candidates)]
redis.call('HINCRBY', inflight_key, best_ep, 1)
return best_ep
"""

# Kept for API compatibility (ignored by Lua script — pure least-connections now)
_DEFAULT_LATENCY_WEIGHT = 1000.0


def _url_hash(url: str) -> str:
    """Short hash of an endpoint URL for use in Redis keys."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


class EndpointPool:
    """Redis-coordinated least-loaded endpoint selector.

    Selection is **atomic** via a Lua script: read global inflight counts →
    filter cooldowns → pick least-loaded → HINCRBY +1, all in a single Redis
    call (~0.05ms on localhost).  This guarantees cross-run visibility with no
    race conditions between concurrent acquire() calls.

    Provides both sync and async APIs.  The async ``acquire()`` delegates to
    sync Redis via the Lua script (safe because 0.05ms is negligible vs the
    30-60s LLM call that follows).
    """

    def __init__(
        self,
        pool_name: str,
        endpoints: list[str],
        redis_url: str = _DEFAULT_REDIS_URL,
        cooldown_secs: int = _DEFAULT_COOLDOWN_SECS,
        latency_weight: float = _DEFAULT_LATENCY_WEIGHT,
    ) -> None:
        if not endpoints:
            raise ValueError("endpoints must be non-empty")
        self._pool_name = pool_name
        self._endpoints = list(endpoints)
        self._cooldown_secs = cooldown_secs
        self._redis_url = redis_url

        # Keys
        self._inflight_key = f"llm_pool:{pool_name}:inflight"
        self._cooldown_keys = [self._cooldown_key(ep) for ep in endpoints]
        self._stats_keys = [self._stats_key(ep) for ep in endpoints]

        # Lua script args: [ep1..N, cd_key1..N]
        self._lua_argv = list(endpoints) + self._cooldown_keys

        # Lazy-init Redis clients
        self._sync_redis: redis.Redis | None = None
        self._async_redis: aioredis.Redis | None = None
        self._lua_sha: str | None = None  # cached SHA of the Lua script

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

    def _ensure_lua(self, r: redis.Redis) -> str:
        """Load the Lua script and cache its SHA."""
        if self._lua_sha is None:
            self._lua_sha = str(r.script_load(_LUA_ACQUIRE))
        assert self._lua_sha is not None  # guaranteed by branch above
        return self._lua_sha

    # ------------------------------------------------------------------
    # Core selection (sync — used by both sync and async paths)
    # ------------------------------------------------------------------

    def _acquire_via_lua(self) -> str:
        """Atomic least-loaded selection via Lua script (0.05ms)."""
        r = self._get_sync()
        sha = self._ensure_lua(r)
        try:
            return r.evalsha(sha, 1, self._inflight_key, *self._lua_argv)  # type: ignore[return-value]
        except redis.exceptions.NoScriptError:
            # Script evicted from cache — reload
            self._lua_sha = None
            sha = self._ensure_lua(r)
            return r.evalsha(sha, 1, self._inflight_key, *self._lua_argv)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def acquire(self) -> str:
        """Pick the least-loaded healthy endpoint and increment its count.

        Uses a Lua script for atomic cross-run-aware selection (~0.05ms).
        """
        return self._acquire_via_lua()

    async def release(self, endpoint: str, latency_ms: float = 0.0) -> None:
        """Decrement in-flight count and update stats."""
        r = self._get_async()
        stats_key = self._stats_key(endpoint)
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        pipe.hincrby(stats_key, "requests", 1)
        pipe.hincrbyfloat(stats_key, "total_latency_ms", latency_ms)
        await pipe.execute()

    async def mark_unhealthy(self, endpoint: str) -> None:
        """Decrement inflight, set cooldown, record error."""
        r = self._get_async()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        pipe.set(self._cooldown_key(endpoint), "1", ex=self._cooldown_secs)
        pipe.hincrby(self._stats_key(endpoint), "errors", 1)
        await pipe.execute()

    async def get_stats(self) -> dict[str, dict[str, Any]]:
        """Read per-endpoint stats from Redis."""
        r = self._get_async()
        pipe = r.pipeline(transaction=False)
        pipe.hgetall(self._inflight_key)
        for ep in self._endpoints:
            pipe.hgetall(self._stats_key(ep))
            pipe.exists(self._cooldown_key(ep))
        results = await pipe.execute()

        inflight = results[0]
        out: dict[str, dict[str, Any]] = {}
        for i, ep in enumerate(self._endpoints):
            stats_raw = results[1 + i * 2]
            cooldown = results[2 + i * 2]
            out[ep] = {
                "inflight": int(inflight.get(ep, 0)),
                "requests": int(stats_raw.get("requests", 0)),
                "errors": int(stats_raw.get("errors", 0)),
                "total_latency_ms": float(stats_raw.get("total_latency_ms", 0)),
                "healthy": not cooldown,
            }
        return out

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

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def acquire_sync(self) -> str:
        """Sync version of acquire (same Lua script)."""
        return self._acquire_via_lua()

    def release_sync(self, endpoint: str, latency_ms: float = 0.0) -> None:
        """Sync version of release."""
        r = self._get_sync()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        stats_key = self._stats_key(endpoint)
        pipe.hincrby(stats_key, "requests", 1)
        pipe.hincrbyfloat(stats_key, "total_latency_ms", latency_ms)
        pipe.execute()

    def mark_unhealthy_sync(self, endpoint: str) -> None:
        """Sync version of mark_unhealthy."""
        r = self._get_sync()
        pipe = r.pipeline(transaction=False)
        pipe.hincrby(self._inflight_key, endpoint, -1)
        pipe.set(self._cooldown_key(endpoint), "1", ex=self._cooldown_secs)
        pipe.hincrby(self._stats_key(endpoint), "errors", 1)
        pipe.execute()

    # ------------------------------------------------------------------
    # Time-series stats snapshot
    # ------------------------------------------------------------------

    def snapshot_stats(self, max_history: int = 10080) -> None:
        """Append a timestamped stats snapshot to a Redis list.

        Call periodically (e.g. from watchdog) to build a time-series of
        pool utilization.  Each entry is JSON::

            {"t": unix_ts, "endpoints": {"url": {"inflight": N, "requests": N, "errors": N}}}

        Key: ``llm_pool:{pool_name}:snapshots`` (trimmed to *max_history*).
        """
        import json as _json

        r = self._get_sync()
        inflight = r.hgetall(self._inflight_key)
        ep_data = {}
        for ep in self._endpoints:
            stats_raw = r.hgetall(self._stats_key(ep))
            ep_data[ep] = {
                "inflight": int(inflight.get(ep, 0)),  # type: ignore[union-attr]
                "requests": int(stats_raw.get("requests", 0)),  # type: ignore[union-attr]
                "errors": int(stats_raw.get("errors", 0)),  # type: ignore[union-attr]
            }
        entry = _json.dumps({"t": time.time(), "endpoints": ep_data})
        key = f"llm_pool:{self._pool_name}:snapshots"
        r.rpush(key, entry)
        r.ltrim(key, -max_history, -1)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close Redis connections."""
        if self._async_redis is not None:
            await self._async_redis.aclose()  # type: ignore[attr-defined]
            self._async_redis = None
        if self._sync_redis is not None:
            self._sync_redis.close()
            self._sync_redis = None
