"""Tests for gigaevo.adversarial.sync — ProgressBasedSyncHook (drift-cap semantics)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
import warnings

import pytest

from gigaevo.adversarial.sync import ProgressBasedSyncHook


def _make_hook(
    *,
    own_db: int = 1,
    own_prefix: str = "test/own",
    sources: list[dict[str, int | str]] | None = None,
    drift_cap: int | None = 10,
    **overrides,
) -> ProgressBasedSyncHook:
    if sources is None:
        sources = [{"db": 2, "prefix": "test/opp"}]
    return ProgressBasedSyncHook(
        host="localhost",
        port=6379,
        sources=sources,
        own_db=own_db,
        own_prefix=own_prefix,
        drift_cap=drift_cap,
        poll_interval=0.01,
        **overrides,
    )


def _install_mock(hook: ProgressBasedSyncHook, db: int, return_values) -> AsyncMock:
    mock_redis = AsyncMock()
    if isinstance(return_values, list):
        mock_redis.hget = AsyncMock(side_effect=return_values)
    else:
        mock_redis.hget = AsyncMock(return_value=return_values)
    hook._redis_clients[db] = mock_redis
    return mock_redis


class TestProgressBasedSyncHookInit:
    def test_stores_config(self):
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 4, "prefix": "adv/pop_a"}],
            own_db=3,
            own_prefix="adv/pop_b",
            drift_cap=20,
            sync_every_n_epochs=3,
            timeout=600.0,
            poll_interval=1.0,
        )
        assert hook._sources == [(4, "adv/pop_a")]
        assert hook._own_db == 3
        assert hook._own_prefix == "adv/pop_b"
        assert hook._drift_cap == 20
        assert hook._sync_every_n_epochs == 3
        assert hook._timeout == 600.0
        assert hook._poll_interval == 1.0

    def test_drift_cap_is_required(self):
        """drift_cap (or alias min_delta) must be explicit — no silent default."""
        with pytest.raises(ValueError, match="drift_cap"):
            ProgressBasedSyncHook(
                host="localhost",
                port=6379,
                sources=[{"db": 0, "prefix": "opp"}],
                own_db=1,
                own_prefix="own",
            )

    def test_drift_cap_rejects_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            ProgressBasedSyncHook(
                host="localhost",
                port=6379,
                sources=[{"db": 0, "prefix": "opp"}],
                own_db=1,
                own_prefix="own",
                drift_cap=-1,
            )

    def test_requires_sources(self):
        with pytest.raises((ValueError, TypeError)):
            ProgressBasedSyncHook(
                host="localhost",
                port=6379,
                sources=[],
                own_db=1,
                own_prefix="own",
                drift_cap=8,
            )

    def test_min_delta_is_deprecated_alias(self):
        """Passing min_delta (no drift_cap) is accepted with a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            hook = ProgressBasedSyncHook(
                host="localhost",
                port=6379,
                sources=[{"db": 0, "prefix": "opp"}],
                own_db=1,
                own_prefix="own",
                min_delta=8,
            )
        assert hook._drift_cap == 8
        assert any(
            issubclass(w.category, DeprecationWarning) and "min_delta" in str(w.message)
            for w in warned
        ), "Expected a DeprecationWarning mentioning min_delta"

    def test_drift_cap_wins_over_min_delta_when_both_given(self):
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[{"db": 0, "prefix": "opp"}],
            own_db=1,
            own_prefix="own",
            drift_cap=5,
            min_delta=99,
        )
        assert hook._drift_cap == 5


class TestProgressBasedSyncHookFirstCall:
    async def test_first_call_logs_baseline_no_block(self):
        """First call reads own+opponent progress, logs baseline, does not block."""
        hook = _make_hook(drift_cap=10)
        # Own is very far ahead — would normally block, but first-call skips drift check
        _install_mock(hook, 1, "500")  # own
        _install_mock(hook, 2, "0")  # opp

        start = time.monotonic()
        await hook()
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"First call should not block; elapsed={elapsed:.3f}s"
        assert hook._first_call is False


class TestProgressBasedSyncHookDriftCap:
    """The core contract: leader waits, follower proceeds, no deadlock possible."""

    async def test_within_cap_proceeds_immediately(self):
        """When |own - opp| <= drift_cap, hook returns without polling-loop wait."""
        hook = _make_hook(drift_cap=10)
        hook._first_call = False  # skip baseline
        _install_mock(hook, 1, "25")  # own
        _install_mock(hook, 2, "20")  # opp, drift = +5 within cap

        start = time.monotonic()
        await hook()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1, f"Within-cap should return fast; elapsed={elapsed:.3f}s"

    async def test_equal_progress_both_proceed(self):
        """Both sides at identical progress => drift=0 <= cap; no block on either side."""
        hook = _make_hook(drift_cap=10)
        hook._first_call = False
        _install_mock(hook, 1, "42")
        _install_mock(hook, 2, "42")

        start = time.monotonic()
        await hook()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1

    async def test_follower_proceeds_immediately_when_behind(self):
        """The 'behind' side's wait condition is always false -> never blocks."""
        hook = _make_hook(drift_cap=10)
        hook._first_call = False
        _install_mock(hook, 1, "5")  # own, far behind
        _install_mock(hook, 2, "100")  # opp, far ahead

        start = time.monotonic()
        await hook()
        elapsed = time.monotonic() - start

        # drift = 5 - 100 = -95 <= 10 -> proceed
        assert elapsed < 0.1, (
            f"Behind side must never block (deadlock-free invariant); "
            f"elapsed={elapsed:.3f}s"
        )

    async def test_leader_blocks_until_follower_catches_up(self):
        """The 'ahead' side waits until drift drops back within cap."""
        hook = _make_hook(drift_cap=10)
        hook._first_call = False
        # own stays at 30; opp advances 10 -> 15 -> 22 (catches up on third poll)
        own_mock = _install_mock(hook, 1, "30")
        _install_mock(hook, 2, ["10", "15", "22"])  # drift: 20, 15, 8 (<=10 -> unblock)

        await hook()

        # own_mock unused reference to keep it alive
        del own_mock

    async def test_k3_1_replay_is_deadlock_free(self):
        """Regression for KF-07.

        Exact numbers from experiments/heilbron/k5-budget-v3/04_issues_log.md:
            G_progress = 319, D_progress = 283, cap = 8.
        Under prior symmetric semantics both sides blocked for ~6962s.
        Under drift-cap: G (leader) blocks until D catches up; D (follower)
        proceeds freely on its pre-step hook.
        """
        # D's hook (follower: D is at 283, opp G at 319 -> drift = -36 <= 8 -> proceed)
        d_hook = _make_hook(
            own_db=10,
            own_prefix="pop_b",
            sources=[{"db": 11, "prefix": "pop_a"}],
            drift_cap=8,
        )
        d_hook._first_call = False
        _install_mock(d_hook, 10, "283")  # own = D
        _install_mock(d_hook, 11, "319")  # opp = G

        start = time.monotonic()
        await d_hook()
        d_elapsed = time.monotonic() - start
        assert d_elapsed < 0.1, (
            f"D (behind) should never block; elapsed={d_elapsed:.3f}s"
        )

        # G's hook (leader: G at 319, opp D at 283 -> drift = +36 > 8 -> wait)
        # Simulate D advancing to 311 after 2 polls (drift drops to 8 == cap).
        g_hook = _make_hook(
            own_db=11,
            own_prefix="pop_a",
            sources=[{"db": 10, "prefix": "pop_b"}],
            drift_cap=8,
        )
        g_hook._first_call = False
        _install_mock(g_hook, 11, "319")  # own = G stays at 319 while waiting
        _install_mock(g_hook, 10, ["283", "290", "311"])  # opp advances

        start = time.monotonic()
        await g_hook()
        g_elapsed = time.monotonic() - start
        assert g_elapsed < 1.0, (
            f"G should unblock quickly once D catches up; elapsed={g_elapsed:.3f}s"
        )


class TestProgressBasedSyncHookEpochSkip:
    async def test_sync_every_n_epochs_skips(self):
        """With sync_every_n_epochs=3, first 2 calls are no-ops, 3rd does check."""
        hook = _make_hook(drift_cap=10, sync_every_n_epochs=3)
        hook._first_call = False
        own_mock = _install_mock(hook, 1, "100")
        opp_mock = _install_mock(hook, 2, "100")

        # Call 1: skip
        await hook()
        assert own_mock.hget.call_count == 0
        assert opp_mock.hget.call_count == 0

        # Call 2: skip
        await hook()
        assert own_mock.hget.call_count == 0
        assert opp_mock.hget.call_count == 0

        # Call 3: sync
        await hook()
        assert own_mock.hget.call_count >= 1
        assert opp_mock.hget.call_count >= 1


class TestProgressBasedSyncHookTimeout:
    async def test_timeout_proceeds(self):
        """If waiting for opponent to catch up exceeds timeout, proceed anyway."""
        hook = _make_hook(drift_cap=10, timeout=0.05)
        hook._first_call = False
        _install_mock(hook, 1, "100")  # own
        _install_mock(hook, 2, "0")  # opp stays at 0 forever -> drift 100 > 10

        start = time.monotonic()
        await hook()
        elapsed = time.monotonic() - start

        # Should have proceeded after timeout (not hung)
        assert elapsed < 1.0, f"Timeout should bound wait; elapsed={elapsed:.3f}s"
        assert elapsed >= 0.05


class TestProgressBasedSyncHookMultiSource:
    async def test_multi_source_takes_minimum(self):
        """With multiple opponents, blocks on the slowest (min) opponent progress."""
        hook = ProgressBasedSyncHook(
            host="localhost",
            port=6379,
            sources=[
                {"db": 4, "prefix": "pop_a"},
                {"db": 5, "prefix": "pop_b"},
            ],
            own_db=1,
            own_prefix="own",
            drift_cap=10,
            poll_interval=0.01,
        )
        hook._first_call = False
        _install_mock(hook, 1, "30")  # own
        _install_mock(hook, 4, "50")  # opp A far ahead
        _install_mock(hook, 5, ["15", "25"])  # opp B catches up on 2nd poll

        await hook()
        # drift on 2nd poll = 30 - min(50, 25) = 30 - 25 = 5 <= 10 -> proceed


class TestProgressBasedSyncHookRedisKey:
    async def test_missing_key_returns_zero(self):
        """If programs_processed key doesn't exist in Redis, treat as 0."""
        hook = _make_hook(drift_cap=10)
        _install_mock(hook, 1, None)
        _install_mock(hook, 2, None)

        # First call reads own + opp and logs baseline (no block)
        await hook()

    async def test_reads_correct_redis_key(self):
        """Verify the correct Redis key and field are read for own and opponent."""
        hook = _make_hook(
            own_db=1,
            own_prefix="chains/hover/own",
            sources=[{"db": 2, "prefix": "chains/hover/opp"}],
            drift_cap=10,
        )
        own_mock = _install_mock(hook, 1, "10")
        opp_mock = _install_mock(hook, 2, "8")

        await hook()  # first-call baseline

        own_mock.hget.assert_any_call(
            "chains/hover/own:run_state", "engine:programs_processed"
        )
        opp_mock.hget.assert_any_call(
            "chains/hover/opp:run_state", "engine:programs_processed"
        )


class TestProgressBasedSyncHookGetRedis:
    def test_lazy_creates_redis(self):
        hook = _make_hook()
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
        hook = _make_hook()
        sentinel = MagicMock()
        hook._redis_clients[0] = sentinel
        assert hook._get_redis(0) is sentinel
