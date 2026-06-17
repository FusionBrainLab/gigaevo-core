"""Tests for Track B4: Redis event counters + read helper.

Track B4 has two sides:
  * Emission (this module): every ``emit(event)`` also INCRs a minute-bucketed
    Redis counter ``{prefix}:events:{event_name}:{minute_bucket}``. Configured
    once per process via ``configure_event_counters``; never raises.
  * Readback (this module): ``count_events_in_window`` sums recent buckets for
    one event, and ``collect_event_window_counts`` does it for many events in
    one round-trip — feeds the AlertDetector ``EVENT_RATE_ZERO`` predicate.

No Redis server is required: we stub the minimal interface AlertDetector uses
(``incr``, ``expire``, ``mget``) via ``unittest.mock.MagicMock``.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock

from omegaconf import OmegaConf
import pytest

from gigaevo.monitoring import emit as emit_mod
from gigaevo.monitoring.emit import (
    configure_event_counters,
    configure_event_counters_from_cfg,
    emit,
    reset_event_counters,
)
from gigaevo.monitoring.events import BaseEvent
from gigaevo.monitoring.redis_queries import (
    collect_event_window_counts,
    count_events_in_window,
)


class _PingEvent(BaseEvent):
    event: ClassVar[str] = "__B4_PING__"
    description: ClassVar[str] = "Track B4 counter test event"
    health_question: ClassVar[str] = "?"

    note: str = ""


@pytest.fixture(autouse=True)
def _reset_counters():
    """Ensure module-level counter state is clean between tests."""
    reset_event_counters()
    yield
    reset_event_counters()


# ---------------------------------------------------------------------------
# emit() INCRs the counter when configured
# ---------------------------------------------------------------------------


class TestEmitIncrementsCounter:
    def test_emit_incrs_minute_bucket_key(self):
        client = MagicMock()
        configure_event_counters(client=client, prefix="test_prefix")

        emit(_PingEvent(note="one"))

        assert client.incr.called, "expected a Redis INCR on emit"
        key = client.incr.call_args.args[0]
        assert key.startswith("test_prefix:events:__B4_PING__:")
        # Minute bucket suffix is a non-negative integer.
        suffix = key.rsplit(":", 1)[-1]
        assert suffix.isdigit()
        assert int(suffix) >= 0

    def test_emit_sets_ttl_on_counter_key(self):
        client = MagicMock()
        configure_event_counters(client=client, prefix="p")

        emit(_PingEvent())

        # We set a TTL so old buckets don't accumulate forever.
        assert client.expire.called, "expected EXPIRE to bound counter lifetime"
        ttl_seconds = client.expire.call_args.args[1]
        assert ttl_seconds >= 3600, "TTL must cover at least the recent window"

    def test_emit_without_configure_does_not_touch_redis(self):
        # Purposely do NOT call configure_event_counters — default state.
        # emit() must remain a pure loguru-only operation.
        emit(_PingEvent(note="orphan"))  # should not raise

    def test_emit_survives_redis_errors(self):
        class _FlakeyClient:
            def incr(self, *a, **kw):
                raise RuntimeError("redis down")

            def expire(self, *a, **kw):
                raise RuntimeError("redis down")

        configure_event_counters(client=_FlakeyClient(), prefix="p")
        # Must not propagate — emit is on the hot path.
        emit(_PingEvent(note="still emitted"))

    def test_reset_clears_configuration(self):
        client = MagicMock()
        configure_event_counters(client=client, prefix="p")
        reset_event_counters()

        emit(_PingEvent())

        assert not client.incr.called


class TestConfigureEventCountersIgnoresEmptyPrefix:
    def test_empty_prefix_disables_counters(self):
        client = MagicMock()
        configure_event_counters(client=client, prefix="")

        emit(_PingEvent())

        assert not client.incr.called, "empty prefix must be a no-op"


# ---------------------------------------------------------------------------
# configure_event_counters_from_cfg only wires Redis for Redis-backed runs.
# Minute-bucket counters live in Redis and are read by the watchdog; a
# disk-storage run is Redis-free by design, so it must NOT construct a client
# or write counters into whatever DB the leftover ``redis`` node points at.
# ---------------------------------------------------------------------------

_REDIS_TARGET = "gigaevo.database.redis_program_storage.RedisProgramStorage"
_DISK_TARGET = "gigaevo.database.disk_program_storage.DiskProgramStorage"


def _cfg(target: str, *, prefix: str = "exp", db: int = 0):
    return OmegaConf.create(
        {
            "program_storage": {
                "_target_": target,
                "config": {"key_prefix": prefix},
            },
            "redis": {"host": "localhost", "port": 6379, "db": db},
        }
    )


class TestConfigureFromCfgRespectsStorageBackend:
    def test_disk_storage_builds_no_redis_client(self, monkeypatch):
        import redis

        ctor = MagicMock(side_effect=AssertionError("disk run must not touch Redis"))
        monkeypatch.setattr(redis, "Redis", ctor)

        configure_event_counters_from_cfg(_cfg(_DISK_TARGET))

        assert not ctor.called, "disk run must not construct a Redis client"
        assert emit_mod._event_redis is None
        # And emit() stays Redis-free afterwards.
        emit(_PingEvent(note="disk"))

    def test_redis_storage_wires_counters_to_configured_db(self, monkeypatch):
        import redis

        fake_client = MagicMock()
        ctor = MagicMock(return_value=fake_client)
        monkeypatch.setattr(redis, "Redis", ctor)

        configure_event_counters_from_cfg(_cfg(_REDIS_TARGET, prefix="exp", db=7))

        ctor.assert_called_once_with(host="localhost", port=6379, db=7)
        emit(_PingEvent(note="redis"))
        assert fake_client.incr.called, "redis-backed run must INCR counters"
        assert emit_mod._event_prefix == "exp"

    def test_redis_storage_with_empty_prefix_is_noop(self, monkeypatch):
        import redis

        ctor = MagicMock(side_effect=AssertionError("no prefix → nowhere to write"))
        monkeypatch.setattr(redis, "Redis", ctor)

        configure_event_counters_from_cfg(_cfg(_REDIS_TARGET, prefix=""))

        assert not ctor.called
        assert emit_mod._event_redis is None


# ---------------------------------------------------------------------------
# Read helpers sum minute buckets
# ---------------------------------------------------------------------------


class TestCountEventsInWindow:
    def test_sums_recent_buckets(self):
        # Fake Redis: MGET returns a value per bucket key. Bytes mimic driver.
        client = MagicMock()
        client.mget.return_value = [b"3", b"1", None, b"5"]

        total = count_events_in_window(
            client,
            prefix="p",
            event_name="HOF_ROTATE",
            window_minutes=4,
            now_minute=100,
        )

        # Keys should be the 4 most recent buckets: 97,98,99,100.
        keys = client.mget.call_args.args[0]
        assert keys == [
            "p:events:HOF_ROTATE:97",
            "p:events:HOF_ROTATE:98",
            "p:events:HOF_ROTATE:99",
            "p:events:HOF_ROTATE:100",
        ]
        assert total == 3 + 1 + 0 + 5

    def test_zero_when_all_buckets_empty(self):
        client = MagicMock()
        client.mget.return_value = [None, None, None]

        total = count_events_in_window(
            client, prefix="p", event_name="X", window_minutes=3, now_minute=50
        )
        assert total == 0

    def test_zero_when_redis_errors(self):
        class _Flakey:
            def mget(self, *_a, **_kw):
                raise RuntimeError("redis down")

        # Read path must also be resilient.
        total = count_events_in_window(
            _Flakey(), prefix="p", event_name="X", window_minutes=3
        )
        assert total == 0


class TestCollectEventWindowCounts:
    def test_collects_all_event_names_in_one_mget(self):
        client = MagicMock()
        # 2 events × 3 buckets = 6 keys. Return values positionally.
        client.mget.return_value = [
            b"1",
            b"0",
            b"2",  # ev A: total 3
            None,
            None,
            None,  # ev B: total 0
        ]

        counts = collect_event_window_counts(
            client,
            prefix="p",
            event_names=["A", "B"],
            window_minutes=3,
            now_minute=10,
        )

        assert counts == {"A": 3, "B": 0}
        # One call — pipelined.
        assert client.mget.call_count == 1


# ---------------------------------------------------------------------------
# Emit path sets `decode_responses=False` friendly — mget returns bytes.
# The counter helper must tolerate both bytes and str.
# ---------------------------------------------------------------------------


class TestCounterValueParsing:
    def test_accepts_bytes_and_str_and_int(self):
        client = MagicMock()
        client.mget.return_value = [b"2", "3", 4, None]

        total = count_events_in_window(
            client, prefix="p", event_name="X", window_minutes=4, now_minute=100
        )
        assert total == 9


def test_module_level_state_is_process_scoped():
    # Sanity check the private names exist — test fails loud if refactored
    # without updating callers.
    assert hasattr(emit_mod, "_event_redis")
    assert hasattr(emit_mod, "_event_prefix")
