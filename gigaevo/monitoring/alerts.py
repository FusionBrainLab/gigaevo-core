"""Alert detection for experiment health monitoring.

Analyzes RunSnapshot sequences to detect stalls, crashes, high invalidity,
and completion. Multi-signal detection prevents false alarms (P-WD-01).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger

from gigaevo.monitoring.snapshot import RunSnapshot


class AlertType(StrEnum):
    """Types of alerts the detector can raise."""

    STALL = "stall"
    CRASH = "crash"
    HIGH_INVALIDITY = "high_invalidity"
    COMPLETION = "completion"
    LOW_THROUGHPUT = "low_throughput"


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Alert:
    """An immutable alert raised by the AlertDetector.

    Attributes:
        alert_type: Category of the alert.
        severity: How urgent this alert is.
        run_label: Which run this alert is about (or "experiment" for global alerts).
        message: Human-readable summary.
        details: Optional structured data for programmatic consumers.
    """

    alert_type: AlertType
    severity: AlertSeverity
    run_label: str
    message: str
    details: dict | None = field(default=None)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.alert_type}: {self.run_label} -- {self.message}"


class AlertDetector:
    """Detects experiment health issues from RunSnapshot sequences.

    Multi-signal detection prevents false alarms (P-WD-01). Cooldown
    prevents alert floods. Stateful: tracks previous snapshots and
    alert history between calls.

    Usage:
        detector = AlertDetector(max_generations=50)
        alerts = detector.check(current_snapshots)
        # ... next cycle ...
        alerts = detector.check(current_snapshots)
    """

    def __init__(
        self,
        max_generations: int | None = None,
        invalidity_threshold: float = 0.75,
        invalidity_min_generation: int = 3,
        cooldown_cycles: int = 2,
    ):
        self._max_generations = max_generations
        self._invalidity_threshold = invalidity_threshold
        self._invalidity_min_gen = invalidity_min_generation
        self._cooldown_cycles = cooldown_cycles

        # State between calls
        self._previous_snapshots: dict[str, RunSnapshot] = {}
        # Cooldown tracking: (alert_type, run_label) -> cycles_remaining
        self._cooldowns: dict[tuple[str, str], int] = {}
        self._cycle_count = 0

    def check(self, snapshots: list[RunSnapshot]) -> list[Alert]:
        """Check all snapshots for alerts.

        Args:
            snapshots: Current cycle's RunSnapshots.

        Returns:
            List of Alert objects. May be empty if no issues detected
            or all alerts are in cooldown.
        """
        self._cycle_count += 1
        raw_alerts: list[Alert] = []

        for snap in snapshots:
            label = snap.run_spec.label
            log = logger.bind(component="alerts", run=label)

            # --- Stall detection (multi-signal) ---
            prev = self._previous_snapshots.get(label)
            if prev is not None and snap.is_stalled(prev):
                log.warning(
                    f"Stall detected: gen={snap.generation}, "
                    f"running={snap.running_programs}, total={snap.total_programs}"
                )
                raw_alerts.append(
                    Alert(
                        alert_type=AlertType.STALL,
                        severity=AlertSeverity.WARN,
                        run_label=label,
                        message=(
                            f"Run {label} stalled at gen {snap.generation}: "
                            f"no generation advancement, no running programs, "
                            f"no new submissions."
                        ),
                        details={
                            "generation": snap.generation,
                            "running_programs": snap.running_programs,
                            "total_programs": snap.total_programs,
                        },
                    )
                )

            # --- Crash detection ---
            if snap.pid is not None and snap.pid_alive is False:
                log.error(f"Crash detected: PID {snap.pid} is not alive")
                raw_alerts.append(
                    Alert(
                        alert_type=AlertType.CRASH,
                        severity=AlertSeverity.ERROR,
                        run_label=label,
                        message=f"Run {label} process (PID {snap.pid}) is not alive.",
                        details={"pid": snap.pid},
                    )
                )

            # --- High invalidity ---
            inv_rate = snap.invalid_rate
            if (
                inv_rate is not None
                and inv_rate > self._invalidity_threshold
                and snap.generation is not None
                and snap.generation >= self._invalidity_min_gen
            ):
                pct = inv_rate * 100
                log.warning(f"High invalidity: {pct:.0f}% at gen {snap.generation}")
                raw_alerts.append(
                    Alert(
                        alert_type=AlertType.HIGH_INVALIDITY,
                        severity=AlertSeverity.WARN,
                        run_label=label,
                        message=(
                            f"Run {label}: {pct:.0f}% invalid programs at gen "
                            f"{snap.generation} -- stage_timeout is likely too "
                            f"short for this eval workload."
                        ),
                        details={
                            "invalid_rate": inv_rate,
                            "generation": snap.generation,
                            "total_programs": snap.total_programs,
                            "valid_programs": snap.valid_programs,
                        },
                    )
                )

        # --- Completion detection (global, not per-run) ---
        if self._max_generations is not None and snapshots:
            all_complete = all(
                snap.generation is not None and snap.generation >= self._max_generations
                for snap in snapshots
            )
            if all_complete:
                gen_summary = ", ".join(
                    f"{s.run_spec.label}={s.generation}" for s in snapshots
                )
                logger.bind(component="alerts").info(
                    f"Experiment complete: all runs at "
                    f"max_generations={self._max_generations}"
                )
                raw_alerts.append(
                    Alert(
                        alert_type=AlertType.COMPLETION,
                        severity=AlertSeverity.INFO,
                        run_label="experiment",
                        message=(
                            f"All {len(snapshots)} runs have reached "
                            f"max_generations ({self._max_generations}): "
                            f"{gen_summary}."
                        ),
                        details={
                            "max_generations": self._max_generations,
                            "run_generations": {
                                s.run_spec.label: s.generation for s in snapshots
                            },
                        },
                    )
                )

        # --- Apply cooldowns ---
        alerts = self._apply_cooldowns(raw_alerts)

        # --- Update state for next cycle ---
        for snap in snapshots:
            self._previous_snapshots[snap.run_spec.label] = snap

        return alerts

    def _apply_cooldowns(self, raw_alerts: list[Alert]) -> list[Alert]:
        """Filter alerts through cooldown tracker.

        Each (alert_type, run_label) pair gets a cooldown counter.
        On each call: first filter alerts (suppress if counter > 0),
        then decrement only pre-existing counters (NOT newly set ones).

        Semantics: cooldown_cycles=N means the alert fires, then is
        suppressed for the next N consecutive check() calls, then
        fires again on call N+2.
        Example with cooldown_cycles=2:
          Call 1: fires (counter set to 2)
          Call 2: suppressed (counter 2 -> 1)
          Call 3: suppressed (counter 1 -> 0, deleted)
          Call 4: fires again
        """
        # 1. Filter alerts against current cooldowns, track newly set keys
        emitted: list[Alert] = []
        new_keys: set[tuple[str, str]] = set()
        for alert in raw_alerts:
            key = (alert.alert_type.value, alert.run_label)
            if key in self._cooldowns:
                logger.bind(component="alerts").debug(
                    f"Suppressed {alert.alert_type} for {alert.run_label} "
                    f"(cooldown: {self._cooldowns[key]} cycles remaining)"
                )
                continue
            emitted.append(alert)
            self._cooldowns[key] = self._cooldown_cycles
            new_keys.add(key)

        # 2. Decrement only pre-existing cooldowns (skip newly set ones)
        expired_keys = []
        for key in list(self._cooldowns.keys()):
            if key in new_keys:
                continue  # newly set this cycle -- start decrementing next cycle
            self._cooldowns[key] -= 1
            if self._cooldowns[key] <= 0:
                expired_keys.append(key)
        for key in expired_keys:
            del self._cooldowns[key]

        return emitted
