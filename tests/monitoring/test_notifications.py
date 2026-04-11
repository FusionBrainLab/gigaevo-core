"""Tests for gigaevo.monitoring.notifications — data model, ABC, and formatters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gigaevo.monitoring import Alert, AlertSeverity, AlertType, RunSnapshot
from gigaevo.monitoring.run_spec import RunSpec

from gigaevo.monitoring.notifications import (
    NotificationChannel,
    PlotAttachment,
    StatusUpdate,
)


# ── Factories ────────────────────────────────────────────────────────────────


def _make_snapshot(
    label: str = "A",
    db: int = 1,
    generation: int | None = 5,
    fitness: float | None = 0.762,
    invalid_rate_inputs: tuple[int, int] | None = (100, 80),
    val_mean: float | None = 639.0,
    val_max: float | None = 980.0,
    keys: int | None = 157,
    pid: int | None = 49341,
    pid_alive: bool | None = True,
) -> RunSnapshot:
    """Factory for test RunSnapshots."""
    total, valid = invalid_rate_inputs if invalid_rate_inputs else (None, None)
    return RunSnapshot(
        run_spec=RunSpec(prefix="chains/test/static", db=db, label=label),
        generation=generation,
        metrics={"fitness": fitness},
        total_programs=total,
        valid_programs=valid,
        validator_mean_s=val_mean,
        validator_max_s=val_max,
        total_keys=keys,
        pid=pid,
        pid_alive=pid_alive,
    )


def _make_alert(
    alert_type: AlertType = AlertType.STALL,
    severity: AlertSeverity = AlertSeverity.WARN,
    run_label: str = "A",
    message: str = "Run A stalled at gen 5",
    details: dict | None = None,
) -> Alert:
    return Alert(
        alert_type=alert_type,
        severity=severity,
        run_label=run_label,
        message=message,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PlotAttachment tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotAttachment:
    def test_construct_with_fields(self) -> None:
        pa = PlotAttachment(path=Path("/tmp/plot.png"), caption="Fitness curves")
        assert pa.path == Path("/tmp/plot.png")
        assert pa.caption == "Fitness curves"

    def test_frozen(self) -> None:
        pa = PlotAttachment(path=Path("/tmp/plot.png"), caption="Fitness curves")
        with pytest.raises(AttributeError):
            pa.path = Path("/other")  # type: ignore[misc]

    def test_caption_defaults_to_empty(self) -> None:
        pa = PlotAttachment(path=Path("/tmp/plot.png"))
        assert pa.caption == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. StatusUpdate construction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusUpdateConstruction:
    def test_construct_full(self) -> None:
        snap = _make_snapshot()
        alert = _make_alert()
        plot = PlotAttachment(path=Path("/tmp/p.png"), caption="c")
        ts = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)

        update = StatusUpdate(
            experiment_name="hover/push",
            snapshots=[snap],
            alerts=[alert],
            plots=[plot],
            max_generations=50,
            timestamp=ts,
        )

        assert update.experiment_name == "hover/push"
        assert update.snapshots == [snap]
        assert update.alerts == [alert]
        assert update.plots == [plot]
        assert update.max_generations == 50
        assert update.timestamp == ts

    def test_construct_empty_lists(self) -> None:
        update = StatusUpdate(experiment_name="test/x")
        assert update.snapshots == []
        assert update.alerts == []
        assert update.plots == []

    def test_frozen(self) -> None:
        update = StatusUpdate(experiment_name="test/x")
        with pytest.raises(AttributeError):
            update.experiment_name = "other"  # type: ignore[misc]

    def test_max_generations_none(self) -> None:
        update = StatusUpdate(experiment_name="test/x", max_generations=None)
        assert update.max_generations is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. StatusUpdate convenience properties tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusUpdateProperties:
    def test_has_alerts_true(self) -> None:
        update = StatusUpdate(
            experiment_name="test/x",
            alerts=[_make_alert()],
        )
        assert update.has_alerts is True

    def test_has_alerts_false(self) -> None:
        update = StatusUpdate(experiment_name="test/x", alerts=[])
        assert update.has_alerts is False

    def test_has_plots_true(self) -> None:
        update = StatusUpdate(
            experiment_name="test/x",
            plots=[PlotAttachment(path=Path("/tmp/p.png"))],
        )
        assert update.has_plots is True

    def test_has_plots_false(self) -> None:
        update = StatusUpdate(experiment_name="test/x", plots=[])
        assert update.has_plots is False

    def test_run_count(self) -> None:
        update = StatusUpdate(
            experiment_name="test/x",
            snapshots=[_make_snapshot(label="A"), _make_snapshot(label="B")],
        )
        assert update.run_count == 2

    def test_run_count_empty(self) -> None:
        update = StatusUpdate(experiment_name="test/x")
        assert update.run_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NotificationChannel ABC enforcement tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotificationChannelABC:
    def test_direct_instantiation_raises(self) -> None:
        with pytest.raises(TypeError):
            NotificationChannel()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(NotificationChannel):
            async def send_status(self, update: StatusUpdate) -> bool:
                return True

            async def send_alert(self, alert: Alert) -> bool:
                return True

            async def check_health(self) -> bool:
                return True

        ch = Complete()
        assert isinstance(ch, NotificationChannel)

    def test_missing_send_status_raises(self) -> None:
        class Partial(NotificationChannel):
            async def send_alert(self, alert: Alert) -> bool:
                return True

            async def check_health(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_missing_send_alert_raises(self) -> None:
        class Partial(NotificationChannel):
            async def send_status(self, update: StatusUpdate) -> bool:
                return True

            async def check_health(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_missing_check_health_raises(self) -> None:
        class Partial(NotificationChannel):
            async def send_status(self, update: StatusUpdate) -> bool:
                return True

            async def send_alert(self, alert: Alert) -> bool:
                return True

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NotificationChannel method signature tests (async)
# ═══════════════════════════════════════════════════════════════════════════════


class FakeChannel(NotificationChannel):
    """Test double that records calls and returns True."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send_status(self, update: StatusUpdate) -> bool:
        self.calls.append("send_status")
        return True

    async def send_alert(self, alert: Alert) -> bool:
        self.calls.append("send_alert")
        return True

    async def check_health(self) -> bool:
        self.calls.append("check_health")
        return True


class TestNotificationChannelAsync:
    @pytest.mark.asyncio
    async def test_send_status_returns_bool(self) -> None:
        ch = FakeChannel()
        update = StatusUpdate(experiment_name="test/x")
        result = await ch.send_status(update)
        assert result is True
        assert "send_status" in ch.calls

    @pytest.mark.asyncio
    async def test_send_alert_returns_bool(self) -> None:
        ch = FakeChannel()
        alert = _make_alert()
        result = await ch.send_alert(alert)
        assert result is True
        assert "send_alert" in ch.calls

    @pytest.mark.asyncio
    async def test_check_health_returns_bool(self) -> None:
        ch = FakeChannel()
        result = await ch.check_health()
        assert result is True
        assert "check_health" in ch.calls
