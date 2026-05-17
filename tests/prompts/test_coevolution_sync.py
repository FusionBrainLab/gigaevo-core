"""Tests for gigaevo.prompts.coevolution.sync — MainRunSyncHook.

DataPlane handles are injected via ``dataplanes=`` and stubbed with an
``AsyncMock`` exposing :meth:`raw_hash_get`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.dataplane import Ok
from gigaevo.prompts.coevolution.sync import MainRunSyncHook


def _stub_dataplane(*, hget_returns) -> AsyncMock:
    """Stand-in for DataPlane.raw_hash_get; wraps values in :class:`Ok`."""
    dp = AsyncMock()
    if isinstance(hget_returns, list):
        dp.raw_hash_get = AsyncMock(side_effect=[Ok(v) for v in hget_returns])
    else:
        dp.raw_hash_get = AsyncMock(return_value=Ok(hget_returns))
    return dp


class TestMainRunSyncHookInit:
    def test_stores_config_single_source(self):
        hook = MainRunSyncHook(
            host="redis.example.com",
            port=6380,
            db=3,
            prefix="chains/hotpotqa",
            timeout=1000.0,
            poll_interval=2.0,
        )
        assert hook._host == "redis.example.com"
        assert hook._port == 6380
        assert hook._sources == [(3, "chains/hotpotqa")]
        assert hook._timeout == 1000.0
        assert hook._poll_interval == 2.0
        assert hook._last_main_gen == -1

    def test_stores_config_multi_source(self):
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "chains/hotpotqa"},
                {"db": 5, "prefix": "chains/hotpotqa"},
                {"db": 8, "prefix": "chains/hotpotqa"},
            ],
        )
        assert hook._sources == [
            (4, "chains/hotpotqa"),
            (5, "chains/hotpotqa"),
            (8, "chains/hotpotqa"),
        ]

    def test_defaults(self):
        hook = MainRunSyncHook(host="localhost", port=6379, db=0, prefix="test")
        assert hook._timeout == 7200.0
        assert hook._poll_interval == 5.0

    def test_requires_db_prefix_or_sources(self):
        with pytest.raises(ValueError, match="requires either"):
            MainRunSyncHook(host="localhost", port=6379)

    def test_dataplanes_length_must_match_sources(self):
        with pytest.raises(ValueError, match="dataplanes length"):
            MainRunSyncHook(
                host="localhost",
                port=6379,
                sources=[
                    {"db": 4, "prefix": "p"},
                    {"db": 5, "prefix": "p"},
                ],
                dataplanes=[AsyncMock()],
            )


class TestMainRunSyncHookCall:
    @pytest.mark.asyncio
    async def test_proceeds_immediately_when_gen_advanced(self):
        """If the main run is already at gen > -1, proceed immediately."""
        dp = _stub_dataplane(hget_returns="5")
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
            dataplanes=[dp],
        )

        await hook()

        dp.raw_hash_get.assert_called_once_with(
            "test:run_state", "engine:total_generations"
        )
        assert hook._last_main_gen == 5

    @pytest.mark.asyncio
    async def test_waits_until_gen_advances(self):
        """If main run hasn't advanced, poll until it does."""
        dp = _stub_dataplane(hget_returns=["3", "3", "4"])
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
            dataplanes=[dp],
        )
        hook._last_main_gen = 3

        await hook()

        assert dp.raw_hash_get.call_count == 3
        assert hook._last_main_gen == 4

    @pytest.mark.asyncio
    async def test_timeout_proceeds_without_advancement(self):
        """If main run doesn't advance within timeout, proceed anyway."""
        dp = _stub_dataplane(hget_returns="10")
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            timeout=0.05,
            poll_interval=0.01,
            dataplanes=[dp],
        )
        hook._last_main_gen = 10

        await hook()

        assert hook._last_main_gen == 10

    @pytest.mark.asyncio
    async def test_handles_none_from_redis(self):
        """If the key doesn't exist, treat gen as 0."""
        dp = _stub_dataplane(hget_returns=None)
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
            dataplanes=[dp],
        )

        await hook()

        assert hook._last_main_gen == 0

    @pytest.mark.asyncio
    async def test_tracks_generation_across_calls(self):
        """Multiple calls should track the advancing generation."""
        dp = AsyncMock()
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="test",
            poll_interval=0.01,
            dataplanes=[dp],
        )

        dp.raw_hash_get = AsyncMock(return_value=Ok("0"))
        await hook()
        assert hook._last_main_gen == 0

        dp.raw_hash_get = AsyncMock(return_value=Ok("1"))
        await hook()
        assert hook._last_main_gen == 1

        dp.raw_hash_get = AsyncMock(return_value=Ok("3"))
        await hook()
        assert hook._last_main_gen == 3

    @pytest.mark.asyncio
    async def test_correct_redis_key_construction(self):
        """Verify the hash key is built from the source prefix."""
        dp = _stub_dataplane(hget_returns="1")
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            db=0,
            prefix="chains/hotpotqa",
            poll_interval=0.01,
            dataplanes=[dp],
        )

        await hook()

        dp.raw_hash_get.assert_called_with(
            "chains/hotpotqa:run_state", "engine:total_generations"
        )

    @pytest.mark.asyncio
    async def test_multi_source_waits_for_min_gen(self):
        """With multiple sources, waits for the minimum gen to advance."""
        dp4 = _stub_dataplane(hget_returns="3")
        dp5 = _stub_dataplane(hget_returns="1")
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "chains/hotpotqa"},
                {"db": 5, "prefix": "chains/hotpotqa"},
            ],
            poll_interval=0.01,
            dataplanes=[dp4, dp5],
        )

        # DB4 at gen 3, DB5 at gen 1 → min=1 > -1 → proceed
        await hook()
        assert hook._last_main_gen == 1

    @pytest.mark.asyncio
    async def test_multi_source_blocks_until_slowest_advances(self):
        """Must wait until ALL sources advance past last_main_gen."""
        dp4 = _stub_dataplane(hget_returns=["5", "5"])
        dp5 = _stub_dataplane(hget_returns=["2", "3"])
        hook = MainRunSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "p"},
                {"db": 5, "prefix": "p"},
            ],
            poll_interval=0.01,
            dataplanes=[dp4, dp5],
        )
        hook._last_main_gen = 2

        # First poll: DB4=5, DB5=2 → min=2, not > 2 → wait
        # Second poll: DB4=5, DB5=3 → min=3 > 2 → proceed
        await hook()
        assert hook._last_main_gen == 3
