from __future__ import annotations

import fakeredis
import pytest

from gigaevo.utils.trackers.backends.redis import RedisMetricsBackend
from gigaevo.utils.trackers.configs import RedisMetricsConfig


@pytest.fixture
def backend() -> RedisMetricsBackend:
    cfg = RedisMetricsConfig(
        redis_url="redis://127.0.0.1:6379/0",
        key_prefix="test_metrics",
        store_history=True,
        max_history_per_metric=1000,
    )
    b = RedisMetricsBackend(cfg)
    b._client = fakeredis.FakeRedis(decode_responses=True)
    return b


class TestClearSeriesBufferRace:
    def test_clear_series_drops_pending_buffered_writes_for_tag(
        self, backend: RedisMetricsBackend
    ) -> None:
        backend.write_scalar("foo", 1.0, step=0, wall_time=1.0)
        backend.write_scalar("foo", 2.0, step=1, wall_time=2.0)

        backend.clear_series("foo")
        backend.flush()

        assert backend.get_history("foo") == []

    def test_clear_series_preserves_buffered_writes_for_other_tags(
        self, backend: RedisMetricsBackend
    ) -> None:
        backend.write_scalar("foo", 1.0, step=0, wall_time=1.0)
        backend.write_scalar("bar", 2.0, step=1, wall_time=2.0)

        backend.clear_series("foo")
        backend.flush()

        foo_hist = backend.get_history("foo")
        bar_hist = backend.get_history("bar")
        assert foo_hist == []
        assert len(bar_hist) == 1
        assert bar_hist[0]["v"] == 2.0
        assert bar_hist[0]["s"] == 1

    def test_clear_series_deletes_already_persisted_entries(
        self, backend: RedisMetricsBackend
    ) -> None:
        backend.write_scalar("foo", 1.0, step=0, wall_time=1.0)
        backend.flush()
        assert len(backend.get_history("foo")) == 1

        backend.clear_series("foo")

        assert backend.get_history("foo") == []

    def test_clear_series_no_client_is_noop(self) -> None:
        cfg = RedisMetricsConfig(redis_url="redis://x", key_prefix="np")
        b = RedisMetricsBackend(cfg)
        assert b._client is None
        b.clear_series("foo")
