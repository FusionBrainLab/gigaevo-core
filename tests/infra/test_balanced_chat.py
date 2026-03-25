"""Tests for gigaevo.infra.balanced_chat — load-balanced ChatOpenAI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fakeredis
import fakeredis.aioredis
from langchain_core.messages import AIMessage
import pytest

from gigaevo.infra.balanced_chat import BalancedChatOpenAI, _BalancedStructuredOutput
from gigaevo.infra.pool_metrics import PoolMetricsTracker

ENDPOINTS = [
    "http://server-a:8777/v1",
    "http://server-b:8777/v1",
]


def _patch_pool(balanced: BalancedChatOpenAI, server: fakeredis.FakeServer) -> None:
    """Replace the pool's Redis clients with fakeredis."""
    balanced._pool._sync_redis = fakeredis.FakeRedis(
        server=server, decode_responses=True
    )
    balanced._pool._async_redis = fakeredis.aioredis.FakeRedis(
        server=server, decode_responses=True
    )


@pytest.fixture()
def fake_server():
    return fakeredis.FakeServer()


@pytest.fixture()
def balanced(fake_server: fakeredis.FakeServer) -> BalancedChatOpenAI:
    """Create BalancedChatOpenAI with mocked ChatOpenAI clients."""
    b = BalancedChatOpenAI(
        endpoints=ENDPOINTS,
        pool_name="test",
        model="test-model",
        api_key="fake-key",
    )
    _patch_pool(b, fake_server)

    # Mock the underlying ChatOpenAI clients
    for ep in ENDPOINTS:
        mock_client = MagicMock()
        mock_client.invoke.return_value = AIMessage(content="hello")
        mock_client.ainvoke = AsyncMock(return_value=AIMessage(content="hello async"))
        mock_client.with_structured_output.return_value = MagicMock()
        b._clients[ep] = mock_client

    return b


class TestBalancedInvoke:
    def test_invoke_delegates_to_endpoint(self, balanced: BalancedChatOpenAI):
        result = balanced.invoke("test prompt")
        assert result.content == "hello"

        # One of the clients should have been called
        called = [
            ep for ep, client in balanced._clients.items() if client.invoke.called
        ]
        assert len(called) == 1

    async def test_ainvoke_delegates_to_endpoint(self, balanced: BalancedChatOpenAI):
        result = await balanced.ainvoke("test prompt")
        assert result.content == "hello async"

        called = [
            ep for ep, client in balanced._clients.items() if client.ainvoke.called
        ]
        assert len(called) == 1

    def test_invoke_distributes_load(self, balanced: BalancedChatOpenAI):
        """Multiple calls should distribute across endpoints."""
        for _ in range(10):
            balanced.invoke("test")

        total_calls = sum(
            client.invoke.call_count for client in balanced._clients.values()
        )
        assert total_calls == 10

        # Both endpoints should have been used
        used = [
            ep
            for ep, client in balanced._clients.items()
            if client.invoke.call_count > 0
        ]
        assert len(used) == 2


class TestFailover:
    def test_invoke_marks_unhealthy_on_error(
        self, balanced: BalancedChatOpenAI, fake_server: fakeredis.FakeServer
    ):
        # Make ALL endpoints fail so whichever is selected triggers cooldown
        for ep in ENDPOINTS:
            balanced._clients[ep].invoke.side_effect = ConnectionError("down")

        with pytest.raises(ConnectionError):
            balanced.invoke("test")

        # At least one endpoint should be in cooldown
        r = balanced._pool._get_sync()
        any_cooldown = any(
            r.exists(balanced._pool._cooldown_key(ep)) for ep in ENDPOINTS
        )
        assert any_cooldown

    async def test_ainvoke_marks_unhealthy_on_error(
        self, balanced: BalancedChatOpenAI, fake_server: fakeredis.FakeServer
    ):
        for ep in ENDPOINTS:
            balanced._clients[ep].ainvoke.side_effect = ConnectionError("down")

        with pytest.raises(ConnectionError):
            await balanced.ainvoke("test")

        r = balanced._pool._get_async()
        cooldowns = [
            await r.exists(balanced._pool._cooldown_key(ep)) for ep in ENDPOINTS
        ]
        assert any(cooldowns)


class TestStructuredOutput:
    def test_with_structured_output_returns_balanced_wrapper(
        self, balanced: BalancedChatOpenAI
    ):
        result = balanced.with_structured_output(dict)
        assert isinstance(result, _BalancedStructuredOutput)

    def test_structured_invoke_delegates(self, balanced: BalancedChatOpenAI):
        # Set up structured output mocks
        for ep in ENDPOINTS:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {"parsed": {"key": "value"}, "raw": None}
            balanced._clients[ep].with_structured_output.return_value = mock_chain

        structured = balanced.with_structured_output(dict)
        result = structured.invoke("test")
        assert result is not None


class TestMetrics:
    def test_metrics_recorded_on_success(self, balanced: BalancedChatOpenAI):
        balanced.invoke("test")
        assert len(balanced._metrics.cumulative) > 0

        # Check at least one endpoint has a request recorded
        total = sum(v["requests"] for v in balanced._metrics.cumulative.values())
        assert total == 1

    def test_metrics_recorded_on_failure(self, balanced: BalancedChatOpenAI):
        balanced._clients[ENDPOINTS[0]].invoke.side_effect = RuntimeError("fail")
        balanced._clients[ENDPOINTS[1]].invoke.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError):
            balanced.invoke("test")

        total_errors = sum(v["errors"] for v in balanced._metrics.cumulative.values())
        assert total_errors == 1


class TestPoolMetricsTracker:
    def test_record_writes_to_writer(self):
        writer = MagicMock()
        writer.bind.return_value = writer
        tracker = PoolMetricsTracker(pool_name="test", writer=writer)

        tracker.record("http://server-a:8777/v1", 42.0, success=True)

        assert writer.scalar.called
        assert tracker.cumulative["server-a_8777"]["requests"] == 1

    def test_record_tracks_errors(self):
        tracker = PoolMetricsTracker(pool_name="test", writer=MagicMock())
        tracker.record("http://server-a:8777/v1", 100.0, success=False)
        assert tracker.cumulative["server-a_8777"]["errors"] == 1

    def test_no_writer_still_tracks_cumulative(self):
        tracker = PoolMetricsTracker(pool_name="test", writer=None)
        tracker.record("http://server-a:8777/v1", 42.0, success=True)
        # Cumulative stats tracked even without writer (for get_stats)
        assert tracker.cumulative["server-a_8777"]["requests"] == 1
