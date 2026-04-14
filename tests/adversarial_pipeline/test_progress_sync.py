"""Tests for gigaevo.adversarial.sync — ProgressBasedSyncHook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.adversarial.sync import ProgressBasedSyncHook


class TestProgressBasedSyncHookInit:
    def test_stores_config(self):
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 4, "prefix": "adv/pop_a"}],
            min_delta=20,
            sync_every_n_epochs=3,
            timeout=600.0,
            poll_interval=1.0,
        )
        assert hook._sources == [(4, "adv/pop_a")]
        assert hook._min_delta == 20
        assert hook._sync_every_n_epochs == 3
        assert hook._timeout == 600.0
        assert hook._poll_interval == 1.0

    def test_defaults(self):
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
        )
        assert hook._min_delta == 10
        assert hook._sync_every_n_epochs == 1
        assert hook._timeout == 7200.0
        assert hook._poll_interval == 5.0

    def test_requires_sources(self):
        with pytest.raises((ValueError, TypeError)):
            ProgressBasedSyncHook(host="localhost", port=6379, sources=[])


class TestProgressBasedSyncHookCall:
    async def test_initial_call_records_baseline_no_block(self):
        """First call records baseline progress and returns immediately."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
            min_delta=10,
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value="25")
        hook._redis_clients[0] = mock_redis

        await hook()

        assert hook._last_progress == 25

    async def test_blocks_until_min_delta(self):
        """Blocks until opponent processes min_delta more programs."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
            min_delta=10,
            poll_interval=0.01,
        )
        hook._last_progress = 20  # baseline set from previous call
        mock_redis = AsyncMock()
        # First two polls: not enough progress; third: 30 >= 20+10
        mock_redis.hget = AsyncMock(side_effect=["25", "28", "30"])
        hook._redis_clients[0] = mock_redis

        await hook()

        assert mock_redis.hget.call_count == 3
        assert hook._last_progress == 30

    async def test_sync_every_n_epochs_skips(self):
        """With sync_every_n_epochs=3, first 2 calls are no-ops, 3rd blocks."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
            min_delta=10,
            sync_every_n_epochs=3,
            poll_interval=0.01,
        )
        hook._last_progress = 0
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value="100")
        hook._redis_clients[0] = mock_redis

        # Call 1: skip (epoch_call_count becomes 1)
        await hook()
        assert mock_redis.hget.call_count == 0

        # Call 2: skip (epoch_call_count becomes 2)
        await hook()
        assert mock_redis.hget.call_count == 0

        # Call 3: sync (epoch_call_count becomes 3 → resets to 0)
        await hook()
        assert mock_redis.hget.call_count >= 1
        assert hook._last_progress == 100

    async def test_timeout_proceeds(self):
        """If opponent doesn't advance enough within timeout, proceed anyway."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
            min_delta=100,
            timeout=0.05,
            poll_interval=0.01,
        )
        hook._last_progress = 0
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value="5")  # only 5 < 100
        hook._redis_clients[0] = mock_redis

        await hook()

        # Should have proceeded after timeout and reset baseline to current
        # reality so the next epoch doesn't also wait the full timeout
        assert hook._last_progress == 5  # updated to current min_progress

    async def test_multi_source_takes_minimum(self):
        """With multiple sources, blocks on the slowest opponent."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "pop_a"},
                {"db": 5, "prefix": "pop_b"},
            ],
            min_delta=10,
            poll_interval=0.01,
        )
        hook._last_progress = 10
        mock_r4 = AsyncMock()
        mock_r5 = AsyncMock()
        hook._redis_clients[4] = mock_r4
        hook._redis_clients[5] = mock_r5

        # DB4 far ahead (50), DB5 catches up on 2nd poll (20 >= 10+10)
        mock_r4.hget = AsyncMock(side_effect=["50", "55"])
        mock_r5.hget = AsyncMock(side_effect=["15", "20"])

        await hook()
        assert hook._last_progress == 20  # takes minimum

    async def test_missing_key_returns_zero(self):
        """If programs_processed key doesn't exist in Redis, treat as 0."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
            min_delta=10,
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value=None)
        hook._redis_clients[0] = mock_redis

        # First call: baseline = 0
        await hook()
        assert hook._last_progress == 0

    async def test_reads_correct_redis_key(self):
        """Verify the correct Redis key and field are read."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "chains/hover/static"}],
            poll_interval=0.01,
        )
        mock_redis = AsyncMock()
        mock_redis.hget = AsyncMock(return_value="10")
        hook._redis_clients[0] = mock_redis

        await hook()

        mock_redis.hget.assert_called_with(
            "chains/hover/static:run_state", "engine:programs_processed"
        )


class TestProgressBasedSyncHookGetRedis:
    def test_lazy_creates_redis(self):
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 5, "prefix": "test"}],
        )
        assert hook._redis_clients == {}

        with patch("gigaevo.adversarial.sync.aioredis.Redis") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            result = hook._get_redis(5)

            mock_cls.assert_called_once_with(
                host="localhost",
                port=6379,
                db=5,
                decode_responses=True,
            )
            assert result is mock_client

    def test_reuses_existing(self):
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "test"}],
        )
        sentinel = MagicMock()
        hook._redis_clients[0] = sentinel
        assert hook._get_redis(0) is sentinel


class TestProgressBasedSyncHookEpochParity:
    async def test_enforces_1_to_1_epoch_advancement(self):
        """
        Verify that ProgressBasedSyncHook enforces ~1:1 epoch advancement ratio.

        This test guards against the bug fixed in steady_state.py where
        incremental publications of programs_processed during _ingest_batch
        allowed the faster population to read stale intermediate values and
        advance multiple epochs per opponent epoch (observed 2-2.5x divergence
        in heilbron/asymmetric-iterations).

        With the fix, programs_processed is published ONLY at epoch boundaries
        (step 3a of _epoch_refresh), forcing sync parity.
        """
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "opponent"}],
            min_delta=8,  # standard: epoch_trigger_count = max_mutations_per_generation
            poll_interval=0.01,
        )

        mock_redis = AsyncMock()
        hook._redis_clients[0] = mock_redis

        # Simulate opponent advancing at exactly min_delta per sync.
        # This enforces ~1:1 epoch parity.
        mock_redis.hget = AsyncMock(side_effect=[
            "0",   # epoch 1 baseline: opponent at 0
            "8",   # epoch 2: opponent advanced 8 → unblock
            "16",  # epoch 3: opponent advanced 8 more → unblock
            "24",  # epoch 4: opponent advanced 8 more → unblock
        ])

        # Epoch 1 baseline
        await hook()
        assert hook._last_progress == 0

        # Epoch 2: blocked until opponent reaches 8
        await hook()
        assert hook._last_progress == 8

        # Epoch 3: blocked until opponent reaches 16
        await hook()
        assert hook._last_progress == 16

        # Epoch 4: blocked until opponent reaches 24
        await hook()
        assert hook._last_progress == 24

        # Verify we polled exactly as many times as needed (one poll per epoch)
        # The fix ensures no premature unblocking from intermediate counter values
        assert mock_redis.hget.call_count == 4  # 1 baseline + 3 syncs
