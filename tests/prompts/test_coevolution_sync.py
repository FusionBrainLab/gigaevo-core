"""Tests for gigaevo.prompts.coevolution.sync — MainRunSyncHook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.engine.snapshot import EngineSnapshot
from gigaevo.prompts.coevolution.sync import MainRunSyncHook


def _snap(programs_processed: int | None) -> str | None:
    """Encode a programs_processed value as the engine:snapshot JSON blob.

    ``None`` is passed through (represents a missing Redis key).
    """
    if programs_processed is None:
        return None
    return EngineSnapshot(
        total_mutants=programs_processed,
        programs_processed=programs_processed,
    ).model_dump_json()


def _encode(v):
    """Encode int/None/str fixtures into snapshot JSON (or pass through)."""
    if v is None or isinstance(v, str):
        return v
    return _snap(v)


class TestMainRunSyncHookInit:
    def test_stores_config_single_source(self):
        hook = MainRunSyncHook(
            host="redis.example.com",
            port=6380,
            db=3,
            prefix="chains/synthetic",
            timeout=1000.0,
            poll_interval=2.0,
        )
        assert hook._host == "redis.example.com"
        assert hook._port == 6380
        assert hook._sources == [(3, "chains/synthetic")]
        assert hook._timeout == 1000.0
        assert hook._poll_interval == 2.0
        assert hook._last_main_progress == -1

    def test_stores_config_multi_source(self):
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "chains/synthetic"},
                {"db": 5, "prefix": "chains/synthetic"},
                {"db": 8, "prefix": "chains/synthetic"},
            ],
        )
        assert hook._sources == [
            (4, "chains/synthetic"),
            (5, "chains/synthetic"),
            (8, "chains/synthetic"),
        ]

    def test_defaults(self):
        hook = MainRunSyncHook(host="localhost", port=6379, db=0, prefix="test")
        assert hook._timeout == 7200.0
        assert hook._poll_interval == 5.0

    def test_requires_db_prefix_or_sources(self):
        with pytest.raises(ValueError, match="requires either"):
            MainRunSyncHook(host="localhost", port=6379)


class TestMainRunSyncHookCall:
    @pytest.mark.asyncio
    async def test_proceeds_immediately_when_progress_advanced(self):
        """If the main run is already at programs_processed > -1, proceed immediately."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value=_encode(5))
        hook._redis_clients[0] = mock_redis

        await hook()

        mock_redis.hget.assert_called_once_with("test:run_state", "engine:snapshot")
        assert hook._last_main_progress == 5

    @pytest.mark.asyncio
    async def test_waits_until_progress_advances(self):
        """If main run hasn't advanced, poll until it does."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        hook._last_main_progress = 3
        mock_redis.hget = AsyncMock(side_effect=[_encode(3), _encode(3), _encode(4)])
        hook._redis_clients[0] = mock_redis

        await hook()

        assert mock_redis.hget.call_count == 3
        assert hook._last_main_progress == 4

    @pytest.mark.asyncio
    async def test_timeout_proceeds_without_advancement(self):
        """If main run doesn't advance within timeout, proceed anyway."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            timeout=0.05,
            poll_interval=0.01,
        )
        hook._last_main_progress = 10
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value=_encode(10))
        hook._redis_clients[0] = mock_redis

        await hook()

        assert hook._last_main_progress == 10

    @pytest.mark.asyncio
    async def test_handles_none_from_redis(self):
        """If the key doesn't exist in Redis, treat programs_processed as 0."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value=None)
        hook._redis_clients[0] = mock_redis

        await hook()

        assert hook._last_main_progress == 0

    @pytest.mark.asyncio
    async def test_tracks_progress_across_calls(self):
        """Multiple calls should track the advancing programs_processed."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        hook._redis_clients[0] = mock_redis

        mock_redis.hget = AsyncMock(return_value=_encode(0))
        await hook()
        assert hook._last_main_progress == 0

        mock_redis.hget = AsyncMock(return_value=_encode(1))
        await hook()
        assert hook._last_main_progress == 1

        mock_redis.hget = AsyncMock(return_value=_encode(3))
        await hook()
        assert hook._last_main_progress == 3

    @pytest.mark.asyncio
    async def test_correct_redis_key_construction(self):
        """Verify the Redis key is built from the prefix."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="chains/synthetic",
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value=_encode(1))
        hook._redis_clients[0] = mock_redis

        await hook()

        mock_redis.hget.assert_called_with(
            "chains/synthetic:run_state", "engine:snapshot"
        )

    @pytest.mark.asyncio
    async def test_multi_source_waits_for_min_progress(self):
        """With multiple sources, waits for the minimum programs_processed to advance."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "chains/synthetic"},
                {"db": 5, "prefix": "chains/synthetic"},
            ],
            poll_interval=0.01,
        )
        mock_r4 = AsyncMock()
        mock_r5 = AsyncMock()
        hook._redis_clients[4] = mock_r4
        hook._redis_clients[5] = mock_r5

        # DB4 at progress 3, DB5 at progress 1 → min=1 > -1 → proceed
        mock_r4.hget = AsyncMock(return_value=_encode(3))
        mock_r5.hget = AsyncMock(return_value=_encode(1))

        await hook()
        assert hook._last_main_progress == 1

    @pytest.mark.asyncio
    async def test_multi_source_blocks_until_slowest_advances(self):
        """Must wait until ALL sources advance past last_main_progress."""
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "p"},
                {"db": 5, "prefix": "p"},
            ],
            poll_interval=0.01,
        )
        hook._last_main_progress = 2
        mock_r4 = AsyncMock()
        mock_r5 = AsyncMock()
        hook._redis_clients[4] = mock_r4
        hook._redis_clients[5] = mock_r5

        # First poll: DB4=5, DB5=2 → min=2, not > 2 → wait
        # Second poll: DB4=5, DB5=3 → min=3 > 2 → proceed
        mock_r4.hget = AsyncMock(side_effect=[_encode(5), _encode(5)])
        mock_r5.hget = AsyncMock(side_effect=[_encode(2), _encode(3)])

        await hook()
        assert hook._last_main_progress == 3


class TestMainRunSyncHookGetRedis:
    def test_lazy_creates_redis_on_first_call(self):
        hook = MainRunSyncHook(host="localhost", port=6379, db=5, prefix="test")
        assert hook._redis_clients == {}

        with patch("gigaevo.prompts.coevolution.sync.aioredis.Redis") as mock_redis_cls:
            mock_client = MagicMock()
            mock_redis_cls.return_value = mock_client

            result = hook._get_redis(5)

            mock_redis_cls.assert_called_once_with(
                host="localhost",
                port=6379,
                db=5,
                decode_responses=True,
            )
            assert result is mock_client
            assert hook._redis_clients[5] is mock_client

    def test_reuses_existing_redis(self):
        hook = MainRunSyncHook(host="localhost", port=6379, db=0, prefix="test")
        sentinel = MagicMock()
        hook._redis_clients[0] = sentinel

        result = hook._get_redis(0)
        assert result is sentinel
