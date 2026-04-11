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


# ── Formatters ───────────────────────────────────────────────────────────────


def format_status_table_markdown(snapshots: list[RunSnapshot]) -> str:
    """Render snapshots as a GitHub-flavored markdown table.

    Columns match tools/status.py output: Run, DB, Gen, Fitness,
    Invalid%, Val dur(s), Keys, PID, Status.
    """
    if not snapshots:
        return "_No runs to display._"

    header = "| Run | DB | Gen | Fitness | Invalid% | Val dur(s) | Keys | PID | Status |"
    sep = "|-----|-----|-----|---------|----------|------------|------|-----|--------|"
    rows = [header, sep]

    for snap in snapshots:
        run = snap.run_spec.label
        db = str(snap.run_spec.db)
        gen = str(snap.generation) if snap.generation is not None else "-"
        fitness = _format_fitness(snap.metrics.get("fitness"))
        invalid = _format_invalid_rate(snap.invalid_rate)
        val_dur = _format_validator_duration(snap.validator_mean_s, snap.validator_max_s)
        keys = str(snap.total_keys) if snap.total_keys is not None else "-"
        pid = str(snap.pid) if snap.pid is not None else "-"
        status = _format_pid_status(snap.pid, snap.pid_alive)
        rows.append(
            f"| {run} | {db} | {gen} | {fitness} | {invalid} "
            f"| {val_dur} | {keys} | {pid} | {status} |"
        )

    return "\n".join(rows)


def format_status_table_telegram(snapshots: list[RunSnapshot]) -> str:
    """Render snapshots as a monospace table for Telegram (HTML parse_mode).

    Wrapped in <pre> tags for monospace rendering. Same columns and data
    as format_status_table_markdown (NOT-06 compliance).
    """
    if not snapshots:
        return "<pre>No runs to display.</pre>"

    headers = ["Run", "DB", "Gen", "Fitness", "Inv%", "Val(s)", "Keys", "PID", "Status"]
    table_rows: list[list[str]] = []
    for snap in snapshots:
        table_rows.append([
            snap.run_spec.label,
            str(snap.run_spec.db),
            str(snap.generation) if snap.generation is not None else "-",
            _format_fitness(snap.metrics.get("fitness")),
            _format_invalid_rate(snap.invalid_rate),
            _format_validator_duration(snap.validator_mean_s, snap.validator_max_s),
            str(snap.total_keys) if snap.total_keys is not None else "-",
            str(snap.pid) if snap.pid is not None else "-",
            _format_pid_status(snap.pid, snap.pid_alive),
        ])

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    lines = [_pad_row(headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in table_rows:
        lines.append(_pad_row(row))

    return "<pre>\n" + "\n".join(lines) + "\n</pre>"


def format_alert_message(alert: Alert) -> str:
    """Render a single alert as a human-readable string with severity prefix."""
    severity_prefix = {
        AlertSeverity.INFO: "INFO",
        AlertSeverity.WARN: "WARNING",
        AlertSeverity.ERROR: "ERROR",
    }
    prefix = severity_prefix.get(alert.severity, str(alert.severity).upper())
    return f"[{prefix}] {alert.alert_type.value}: {alert.message}"


# ── Private formatting helpers ───────────────────────────────────────────────


def _format_fitness(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_invalid_rate(rate: float | None) -> str:
    if rate is None:
        return "-"
    return f"{rate * 100:.0f}%"


def _format_validator_duration(
    mean: float | None, max_val: float | None
) -> str:
    if mean is None and max_val is None:
        return "-"
    mean_s = f"{mean:.0f}" if mean is not None else "?"
    max_s = f"{max_val:.0f}" if max_val is not None else "?"
    return f"{mean_s}/{max_s}"


def _format_pid_status(pid: int | None, pid_alive: bool | None) -> str:
    if pid is None:
        return "-"
    if pid_alive is True:
        return "ALIVE"
    if pid_alive is False:
        return "DEAD"
    return "?"
