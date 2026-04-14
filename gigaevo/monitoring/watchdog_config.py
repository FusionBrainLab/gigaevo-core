"""Configuration for the WatchdogEngine.

Frozen dataclass -- all values set at construction time.
heartbeat_ttl_s is derived from poll_interval_s * heartbeat_ttl_multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatchdogConfig:
    """Immutable configuration for WatchdogEngine.

    Attributes:
        poll_interval_s: Seconds between monitoring cycles.
        max_restarts: Maximum outer restart attempts before giving up.
        restart_cooldown_s: Seconds to wait between restart attempts.
        heartbeat_ttl_multiplier: Heartbeat TTL = poll_interval_s * this.
        max_plot_files: Maximum plot files to retain before cleanup.
        stagnation_gens: Consecutive gens without frontier improvement to trigger alert.
        model_drift_check: Whether to verify model identity each cycle.
        redis_host: Redis server hostname.
        redis_port: Redis server port.
    """

    poll_interval_s: int = 3600
    max_restarts: int = 5
    restart_cooldown_s: int = 60
    heartbeat_ttl_multiplier: int = 3
    max_plot_files: int = 50
    stagnation_gens: int = 10
    model_drift_check: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    plot_retries: int = 3
    plot_retry_delay_s: int = 30
    rolling_comment_threshold_hours: int = 24
    checkpoint_milestones: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0)

    @property
    def heartbeat_ttl_s(self) -> int:
        """Heartbeat TTL in seconds = poll_interval_s * heartbeat_ttl_multiplier."""
        return self.poll_interval_s * self.heartbeat_ttl_multiplier
