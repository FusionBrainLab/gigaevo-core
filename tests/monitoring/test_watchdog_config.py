"""Tests for WatchdogConfig frozen dataclass."""

from __future__ import annotations

import pytest

from gigaevo.monitoring.watchdog_config import WatchdogConfig


class TestWatchdogConfigConstruction:
    """Test WatchdogConfig creation and defaults."""

    def test_default_values(self):
        cfg = WatchdogConfig()
        assert cfg.poll_interval_s == 3600
        assert cfg.max_restarts == 5
        assert cfg.restart_cooldown_s == 60
        assert cfg.heartbeat_ttl_multiplier == 3
        assert cfg.max_plot_files == 50
        assert cfg.stagnation_gens == 10
        assert cfg.model_drift_check is True
        assert cfg.redis_host == "localhost"
        assert cfg.redis_port == 6379

    def test_custom_values(self):
        cfg = WatchdogConfig(
            poll_interval_s=1800,
            max_restarts=3,
            heartbeat_ttl_multiplier=2,
            stagnation_gens=5,
        )
        assert cfg.poll_interval_s == 1800
        assert cfg.max_restarts == 3
        assert cfg.heartbeat_ttl_multiplier == 2
        assert cfg.stagnation_gens == 5

    def test_frozen(self):
        cfg = WatchdogConfig()
        with pytest.raises(AttributeError):
            cfg.poll_interval_s = 999  # type: ignore[misc]


class TestWatchdogConfigHeartbeatTTL:
    """Test heartbeat_ttl_s computed property."""

    def test_default_heartbeat_ttl(self):
        cfg = WatchdogConfig()  # poll=3600, multiplier=3
        assert cfg.heartbeat_ttl_s == 10800

    def test_custom_heartbeat_ttl(self):
        cfg = WatchdogConfig(poll_interval_s=600, heartbeat_ttl_multiplier=2)
        assert cfg.heartbeat_ttl_s == 1200
