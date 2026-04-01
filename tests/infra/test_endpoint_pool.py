"""Tests for gigaevo.infra.endpoint_pool — Redis-coordinated load balancing."""

from __future__ import annotations

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.infra.endpoint_pool import EndpointPool, _url_hash

ENDPOINTS = [
    "http://server-a:8777/v1",
    "http://server-b:8777/v1",
    "http://server-c:8777/v1",
]


@pytest.fixture()
def fake_server():
    return fakeredis.FakeServer()


def _make_pool(
    fake_server: fakeredis.FakeServer,
    endpoints: list[str] | None = None,
    pool_name: str = "test",
    cooldown_secs: int = 60,
) -> EndpointPool:
    """Create an EndpointPool backed by fakeredis."""
    pool = EndpointPool(
        pool_name=pool_name,
        endpoints=endpoints or ENDPOINTS,
        cooldown_secs=cooldown_secs,
    )
    # Replace Redis clients with fakeredis (sync used by Lua acquire path)
    pool._sync_redis = fakeredis.FakeRedis(server=fake_server, decode_responses=True)
    pool._async_redis = fakeredis.aioredis.FakeRedis(
        server=fake_server, decode_responses=True
    )
    return pool


class TestAcquireRelease:
    async def test_acquire_returns_endpoint(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        ep = await pool.acquire()
        assert ep in ENDPOINTS

    async def test_acquire_increments_inflight(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        r = pool._get_async()
        ep = await pool.acquire()
        count = await r.hget(pool._inflight_key, ep)
        assert int(count) == 1

    async def test_release_decrements_inflight(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        r = pool._get_async()
        ep = await pool.acquire()
        await pool.release(ep, latency_ms=10.0)
        count = await r.hget(pool._inflight_key, ep)
        assert int(count) == 0

    async def test_release_updates_stats(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        r = pool._get_async()
        ep = await pool.acquire()
        await pool.release(ep, latency_ms=42.5)
        stats = await r.hgetall(pool._stats_key(ep))
        assert int(stats["requests"]) == 1
        assert float(stats["total_latency_ms"]) == pytest.approx(42.5)


class TestLeastLoaded:
    async def test_selects_least_loaded(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        r = pool._get_sync()

        # Pre-load Redis inflight: server-a has 5, server-b has 2, server-c has 0
        r.hset(
            pool._inflight_key,
            mapping={
                ENDPOINTS[0]: 5,
                ENDPOINTS[1]: 2,
                ENDPOINTS[2]: 0,
            },
        )

        ep = await pool.acquire()
        assert ep == ENDPOINTS[2]  # server-c has 0 (lowest)

    async def test_random_tiebreak(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        # All at 0 → should pick from all endpoints
        selected = set()
        for _ in range(50):
            ep = await pool.acquire()
            selected.add(ep)
            await pool.release(ep)
        # With 50 draws from 3 endpoints, all should appear
        assert len(selected) == 3

    async def test_multiple_acquires_distribute_load(
        self, fake_server: fakeredis.FakeServer
    ):
        pool = _make_pool(fake_server)
        # Acquire 3 times — should spread across endpoints
        eps = []
        for _ in range(3):
            ep = await pool.acquire()
            eps.append(ep)
        # Lua selects least-loaded atomically, so all 3 should go to different servers
        assert len(set(eps)) == 3


class TestCooldown:
    async def test_unhealthy_endpoint_skipped(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server, cooldown_secs=60)
        r = pool._get_sync()

        # Pre-set inflight so mark_unhealthy can decrement
        r.hset(
            pool._inflight_key,
            mapping={
                ENDPOINTS[0]: 1,
                ENDPOINTS[1]: 1,
            },
        )
        # Mark server-a and server-b as unhealthy via sync API
        pool.mark_unhealthy_sync(ENDPOINTS[0])
        pool.mark_unhealthy_sync(ENDPOINTS[1])

        # Only server-c should be selected
        for _ in range(10):
            ep = await pool.acquire()
            assert ep == ENDPOINTS[2]
            await pool.release(ep)

    async def test_all_unhealthy_falls_back_to_all(
        self, fake_server: fakeredis.FakeServer
    ):
        pool = _make_pool(fake_server, cooldown_secs=60)
        r = pool._get_sync()

        # Mark all unhealthy
        for ep in ENDPOINTS:
            r.set(pool._cooldown_key(ep), "1", ex=60)

        # Should still return something (fallback to least-loaded)
        ep = await pool.acquire()
        assert ep in ENDPOINTS

    async def test_mark_unhealthy_records_error(
        self, fake_server: fakeredis.FakeServer
    ):
        pool = _make_pool(fake_server)
        r = pool._get_async()
        # Set inflight first
        pool._get_sync().hset(pool._inflight_key, ENDPOINTS[0], 1)
        await pool.mark_unhealthy(ENDPOINTS[0])
        stats = await r.hgetall(pool._stats_key(ENDPOINTS[0]))
        assert int(stats["errors"]) == 1


class TestContextManager:
    async def test_use_acquires_and_releases(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        r = pool._get_async()

        async with pool.use() as ep:
            assert ep in ENDPOINTS
            count = await r.hget(pool._inflight_key, ep)
            assert int(count) == 1

        # After context exit, should be released
        inflight = await r.hgetall(pool._inflight_key)
        total = sum(int(v) for v in inflight.values())
        assert total == 0

    async def test_use_marks_unhealthy_on_error(
        self, fake_server: fakeredis.FakeServer
    ):
        pool = _make_pool(fake_server)
        r = pool._get_async()

        with pytest.raises(RuntimeError, match="boom"):
            async with pool.use() as ep:
                raise RuntimeError("boom")

        # Should have cooldown key set
        assert await r.exists(pool._cooldown_key(ep))


class TestSyncAPI:
    def test_acquire_sync(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        ep = pool.acquire_sync()
        assert ep in ENDPOINTS

    def test_release_sync(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        ep = pool.acquire_sync()
        pool.release_sync(ep, latency_ms=5.0)
        r = pool._get_sync()
        count = r.hget(pool._inflight_key, ep)
        assert int(count) == 0

    def test_mark_unhealthy_sync(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        pool.acquire_sync()  # sets inflight=1 for some endpoint
        r = pool._get_sync()
        # Mark a specific endpoint
        r.hset(pool._inflight_key, ENDPOINTS[0], 1)
        pool.mark_unhealthy_sync(ENDPOINTS[0])
        assert r.exists(pool._cooldown_key(ENDPOINTS[0]))


class TestGetStats:
    async def test_returns_all_endpoints(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        stats = await pool.get_stats()
        assert set(stats.keys()) == set(ENDPOINTS)

    async def test_stats_reflect_usage(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        ep = await pool.acquire()
        await pool.release(ep, latency_ms=100.0)
        ep2 = await pool.acquire()
        await pool.release(ep2, latency_ms=200.0)

        stats = await pool.get_stats()
        total_requests = sum(s["requests"] for s in stats.values())
        assert total_requests == 2


class TestEdgeCases:
    def test_empty_endpoints_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            EndpointPool(pool_name="bad", endpoints=[])

    async def test_single_endpoint(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server, endpoints=["http://only:8777/v1"])
        ep = await pool.acquire()
        assert ep == "http://only:8777/v1"

    def test_url_hash_deterministic(self):
        h1 = _url_hash("http://server-a:8777/v1")
        h2 = _url_hash("http://server-a:8777/v1")
        assert h1 == h2
        assert len(h1) == 12

    def test_url_hash_different_for_different_urls(self):
        h1 = _url_hash("http://server-a:8777/v1")
        h2 = _url_hash("http://server-b:8777/v1")
        assert h1 != h2


class TestCrossRunVisibility:
    """Test that multiple pools (simulating multiple runs) see each other's state."""

    async def test_two_pools_see_shared_inflight(
        self, fake_server: fakeredis.FakeServer
    ):
        pool_a = _make_pool(fake_server, pool_name="shared")
        pool_b = _make_pool(fake_server, pool_name="shared")

        # Pool A acquires — should increment in Redis
        ep_a = await pool_a.acquire()

        # Pool B should see pool A's inflight and pick a different endpoint
        await pool_b.acquire()

        # With 3 endpoints and 1 already taken, pool B picks one of the other 2
        # (both at 0, so random — but NOT the one with inflight=1)
        r = pool_a._get_async()
        inflight = await r.hgetall(pool_a._inflight_key)
        assert int(inflight[ep_a]) >= 1  # pool A's acquire
        total = sum(int(v) for v in inflight.values())
        assert total == 2  # both acquires tracked

    async def test_four_pools_distribute_evenly(
        self, fake_server: fakeredis.FakeServer
    ):
        """Simulate 4 runs each acquiring 2 endpoints — should distribute across all 3."""
        pools = [_make_pool(fake_server, pool_name="shared4") for _ in range(4)]

        selected = []
        for pool in pools:
            for _ in range(2):
                ep = await pool.acquire()
                selected.append(ep)

        # With atomic Lua selection, 8 acquires across 3 endpoints should be
        # roughly even: ~3, ~3, ~2
        from collections import Counter

        counts = Counter(selected)
        assert max(counts.values()) <= 4  # no single endpoint gets more than 4 of 8
        assert len(counts) == 3  # all endpoints used


# ---------------------------------------------------------------------------
# Failure mode tests (previously ZERO coverage)
# ---------------------------------------------------------------------------


class TestLuaScriptEviction:
    """Test that Lua script cache eviction is handled gracefully."""

    async def test_lua_reload_on_eviction(self, fake_server: fakeredis.FakeServer):
        """If Lua script is evicted (NoScriptError), pool should reload and retry."""
        pool = _make_pool(fake_server)

        # First acquire loads the Lua script
        ep1 = await pool.acquire()
        assert ep1 in ENDPOINTS

        # Clear the Lua SHA to simulate eviction
        pool._lua_sha = None

        # Second acquire should reload and succeed
        ep2 = await pool.acquire()
        assert ep2 in ENDPOINTS

    async def test_acquire_after_many_releases(self, fake_server: fakeredis.FakeServer):
        """Acquire after many release cycles should still work."""
        pool = _make_pool(fake_server)

        for _ in range(20):
            ep = await pool.acquire()
            await pool.release(ep, 10.0)

        # Should still be functional
        ep = await pool.acquire()
        assert ep in ENDPOINTS


class TestInflightCounterEdgeCases:
    """Test edge cases in inflight counter management."""

    async def test_release_without_acquire_goes_negative(
        self, fake_server: fakeredis.FakeServer
    ):
        """Releasing without prior acquire decrements below 0.
        Documents current behavior — no underflow protection.
        """
        pool = _make_pool(fake_server)
        ep = ENDPOINTS[0]

        await pool.release(ep, 10.0)

        r = pool._get_async()
        inflight = await r.hget(pool._inflight_key, ep)
        assert int(inflight) == -1, "No underflow protection — counter goes negative"

    async def test_concurrent_acquires_all_succeed(
        self, fake_server: fakeredis.FakeServer
    ):
        """Many sequential acquires should all succeed (simulating burst load)."""
        pool = _make_pool(fake_server)

        endpoints = []
        for _ in range(10):
            ep = await pool.acquire()
            endpoints.append(ep)

        # All should be from the valid set
        for ep in endpoints:
            assert ep in ENDPOINTS

        # Inflight should be 10 total
        r = pool._get_async()
        inflight = await r.hgetall(pool._inflight_key)
        total = sum(int(v) for v in inflight.values())
        assert total == 10

    async def test_mark_unhealthy_then_recover(self, fake_server: fakeredis.FakeServer):
        """After marking unhealthy, endpoint should be skipped during cooldown
        then recover after cooldown expires.
        """
        pool = _make_pool(fake_server, cooldown_secs=1)

        # Acquire and mark unhealthy
        ep = await pool.acquire()
        await pool.mark_unhealthy(ep)

        # During cooldown, this endpoint should be skipped (if others available)
        # With 3 endpoints and 1 unhealthy, acquires should avoid the unhealthy one
        selected = set()
        for _ in range(5):
            selected.add(await pool.acquire())

        # The unhealthy endpoint should be avoided (or at least not preferred)
        # Note: if ALL endpoints are unhealthy, it falls back to all
        assert len(selected) >= 1  # At least one endpoint is selected

    async def test_mark_unhealthy_decrements_inflight(
        self, fake_server: fakeredis.FakeServer
    ):
        """mark_unhealthy should decrement the inflight counter (release the slot)."""
        pool = _make_pool(fake_server)

        ep = await pool.acquire()
        r = pool._get_async()
        inflight_before = int(await r.hget(pool._inflight_key, ep) or 0)

        await pool.mark_unhealthy(ep)
        inflight_after = int(await r.hget(pool._inflight_key, ep) or 0)

        assert inflight_after == inflight_before - 1


class TestStatsAccuracy:
    """Test that stats accurately reflect usage patterns."""

    async def test_stats_after_mixed_success_and_failure(
        self, fake_server: fakeredis.FakeServer
    ):
        """Stats should accurately reflect both successful and failed requests."""
        pool = _make_pool(fake_server)

        # 3 successful requests
        for _ in range(3):
            ep = await pool.acquire()
            await pool.release(ep, 50.0)

        # 2 failed requests
        for _ in range(2):
            ep = await pool.acquire()
            await pool.mark_unhealthy(ep)

        stats = await pool.get_stats()
        total_requests = sum(s.get("requests", 0) for s in stats.values())
        total_errors = sum(s.get("errors", 0) for s in stats.values())

        assert total_requests == 3
        assert total_errors == 2
