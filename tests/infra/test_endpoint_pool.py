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
    # Replace Redis clients with fakeredis
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
        ep = await pool.acquire()
        r = pool._get_async()
        count = await r.hget(pool._inflight_key, ep)
        assert int(count) == 1

    async def test_release_decrements_inflight(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        ep = await pool.acquire()
        await pool.release(ep, latency_ms=10.0)
        r = pool._get_async()
        count = await r.hget(pool._inflight_key, ep)
        assert int(count) == 0

    async def test_release_updates_stats(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        ep = await pool.acquire()
        await pool.release(ep, latency_ms=42.5)
        r = pool._get_async()
        stats = await r.hgetall(pool._stats_key(ep))
        assert int(stats["requests"]) == 1
        assert float(stats["total_latency_ms"]) == pytest.approx(42.5)


class TestLeastLoaded:
    async def test_selects_least_loaded(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        r = pool._get_async()

        # Pre-load: server-a has 5, server-b has 2, server-c has 0
        await r.hset(
            pool._inflight_key,
            mapping={
                ENDPOINTS[0]: 5,
                ENDPOINTS[1]: 2,
                ENDPOINTS[2]: 0,
            },
        )

        ep = await pool.acquire()
        assert ep == ENDPOINTS[2]  # server-c has 0

    async def test_random_tiebreak(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server)
        # All at 0 → should pick randomly from all
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
        # All 3 endpoints should be used (each has 0 initially, then 1)
        assert len(set(eps)) == 3


class TestCooldown:
    async def test_unhealthy_endpoint_skipped(self, fake_server: fakeredis.FakeServer):
        pool = _make_pool(fake_server, cooldown_secs=60)

        # Mark server-a and server-b as unhealthy
        # First acquire them so mark_unhealthy can decrement
        await pool._get_async().hset(
            pool._inflight_key,
            mapping={
                ENDPOINTS[0]: 1,
                ENDPOINTS[1]: 1,
            },
        )
        await pool.mark_unhealthy(ENDPOINTS[0])
        await pool.mark_unhealthy(ENDPOINTS[1])

        # Only server-c should be selected
        for _ in range(10):
            ep = await pool.acquire()
            assert ep == ENDPOINTS[2]
            await pool.release(ep)

    async def test_all_unhealthy_falls_back_to_all(
        self, fake_server: fakeredis.FakeServer
    ):
        pool = _make_pool(fake_server, cooldown_secs=60)
        r = pool._get_async()

        # Mark all unhealthy
        for ep in ENDPOINTS:
            await r.set(pool._cooldown_key(ep), "1", ex=60)

        # Should still return something (fallback to all)
        ep = await pool.acquire()
        assert ep in ENDPOINTS

    async def test_mark_unhealthy_records_error(
        self, fake_server: fakeredis.FakeServer
    ):
        pool = _make_pool(fake_server)
        r = pool._get_async()
        await r.hset(pool._inflight_key, ENDPOINTS[0], 1)
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
        r = pool._get_sync()
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
