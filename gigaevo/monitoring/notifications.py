"""Channel-neutral notification data model and ABC.

StatusUpdate carries all data for a notification cycle.
NotificationChannel defines the async contract for delivery channels.
Formatters render StatusUpdate into channel-specific markup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gigaevo.monitoring.alerts import Alert, AlertSeverity
from gigaevo.monitoring.snapshot import RunSnapshot


@dataclass(frozen=True)
class PlotAttachment:
    """A plot file to attach to notifications.

    Attributes:
        path: Local filesystem path to the PNG/PDF/SVG file.
        caption: Short description for the plot (used as Telegram photo caption
                 and PR comment alt text).
    """

    path: Path
    caption: str = ""


@dataclass(frozen=True)
class StatusUpdate:
    """Channel-neutral data for one notification cycle.

    Immutable. Produced by the watchdog engine, consumed by
    NotificationChannel implementations and their formatters.

    Attributes:
        experiment_name: Human-readable experiment identifier (task/name).
        snapshots: Current state of all monitored runs.
        alerts: Alerts raised in this cycle (already cooldown-filtered).
        plots: Plot files generated this cycle.
        max_generations: Target generation count from experiment.yaml (None in run mode).
        timestamp: When this update was collected.
    """

    experiment_name: str
    snapshots: list[RunSnapshot] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    plots: list[PlotAttachment] = field(default_factory=list)
    max_generations: int | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    @property
    def has_plots(self) -> bool:
        return len(self.plots) > 0

    @property
    def run_count(self) -> int:
        return len(self.snapshots)


class NotificationChannel(ABC):
    """Abstract base for notification delivery channels.

    All methods are async to support non-blocking I/O (httpx, subprocess).
    Return True on success, False on failure. Never raise on delivery errors --
    the dispatcher handles failure tracking.
    """

    @abstractmethod
    async def send_status(self, update: StatusUpdate) -> bool:
        """Send a full status update (table + plots + alerts)."""
        ...

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send a single alert notification."""
        ...

    @abstractmethod
    async def check_health(self) -> bool:
        """Probe channel health. Returns True if the channel is reachable."""
        ...
