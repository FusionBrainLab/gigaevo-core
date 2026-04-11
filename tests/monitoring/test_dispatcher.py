"""Tests for gigaevo.monitoring.dispatcher — NotificationDispatcher fan-out."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gigaevo.monitoring.alerts import Alert, AlertSeverity, AlertType
from gigaevo.monitoring.dispatcher import DispatchResult, NotificationDispatcher
from gigaevo.monitoring.notifications import NotificationChannel, StatusUpdate
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot

# ── Test infrastructure ─────────────────────────────────────────────────────


class FakeChannel(NotificationChannel):
    """Test double that records calls and returns configurable results."""

    def __init__(self, name: str = "fake", *, succeed: bool = True):
        self.name = name
        self._succeed = succeed
        self.send_status_calls: list[StatusUpdate] = []
        self.send_alert_calls: list[Alert] = []
        self.health_check_calls: int = 0

    async def send_status(self, update: StatusUpdate) -> bool:
        self.send_status_calls.append(update)
        return self._succeed

    async def send_alert(self, alert: Alert) -> bool:
        self.send_alert_calls.append(alert)
        return self._succeed

    async def check_health(self) -> bool:
        self.health_check_calls += 1
        return self._succeed


def _make_update(
    experiment_name: str = "test/exp",
    n_snapshots: int = 2,
    alerts: list[Alert] | None = None,
    max_generations: int | None = None,
) -> StatusUpdate:
    snapshots = [
        RunSnapshot(
            run_spec=RunSpec(prefix="prefix", db=i, label=f"R{i}"),
            generation=10 + i,
            metrics={"fitness": 0.7 + i * 0.01},
        )
        for i in range(n_snapshots)
    ]
    return StatusUpdate(
        experiment_name=experiment_name,
        snapshots=snapshots,
        alerts=alerts or [],
        max_generations=max_generations,
        timestamp=datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DispatchResult construction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchResult:
    def test_fields_accessible(self) -> None:
        result = DispatchResult(
            channel_results={"telegram": True, "github_pr": True},
            alerts_sent=2,
            alerts_suppressed=0,
        )
        assert result.channel_results == {"telegram": True, "github_pr": True}
        assert result.alerts_sent == 2
        assert result.alerts_suppressed == 0

    def test_frozen(self) -> None:
        result = DispatchResult(
            channel_results={"telegram": True},
            alerts_sent=0,
            alerts_suppressed=0,
        )
        with pytest.raises(AttributeError):
            result.alerts_sent = 5  # type: ignore[misc]

    def test_all_succeeded_true(self) -> None:
        result = DispatchResult(
            channel_results={"telegram": True, "github_pr": True},
            alerts_sent=0,
            alerts_suppressed=0,
        )
        assert result.all_succeeded is True

    def test_all_succeeded_false_when_one_fails(self) -> None:
        result = DispatchResult(
            channel_results={"telegram": True, "github_pr": False},
            alerts_sent=0,
            alerts_suppressed=0,
        )
        assert result.all_succeeded is False

    def test_any_failed_true(self) -> None:
        result = DispatchResult(
            channel_results={"telegram": True, "github_pr": False},
            alerts_sent=0,
            alerts_suppressed=0,
        )
        assert result.any_failed is True

    def test_any_failed_false_when_all_succeed(self) -> None:
        result = DispatchResult(
            channel_results={"telegram": True, "github_pr": True},
            alerts_sent=0,
            alerts_suppressed=0,
        )
        assert result.any_failed is False

    def test_empty_channel_results_all_succeeded_vacuously_true(self) -> None:
        result = DispatchResult()
        assert result.all_succeeded is True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NotificationDispatcher construction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotificationDispatcherConstruction:
    def test_creates_instance(self) -> None:
        ch1 = FakeChannel("ch1")
        ch2 = FakeChannel("ch2")
        dispatcher = NotificationDispatcher(channels=[ch1, ch2])
        assert isinstance(dispatcher, NotificationDispatcher)

    def test_channels_count(self) -> None:
        ch1 = FakeChannel("ch1")
        ch2 = FakeChannel("ch2")
        dispatcher = NotificationDispatcher(channels=[ch1, ch2])
        assert len(dispatcher.channels) == 2

    def test_empty_channels_valid(self) -> None:
        dispatcher = NotificationDispatcher(channels=[])
        assert len(dispatcher.channels) == 0


class FakeAlertFailChannel(NotificationChannel):
    """Test double that succeeds on send_status but fails on send_alert."""

    def __init__(self, name: str = "alert_fail"):
        self.name = name
        self.send_status_calls: list[StatusUpdate] = []
        self.send_alert_calls: list[Alert] = []

    async def send_status(self, update: StatusUpdate) -> bool:
        self.send_status_calls.append(update)
        return True

    async def send_alert(self, alert: Alert) -> bool:
        self.send_alert_calls.append(alert)
        return False

    async def check_health(self) -> bool:
        return True


def _make_alert(
    alert_type: AlertType = AlertType.STALL,
    severity: AlertSeverity = AlertSeverity.WARN,
    run_label: str = "R0",
    message: str = "test alert",
) -> Alert:
    return Alert(
        alert_type=alert_type,
        severity=severity,
        run_label=run_label,
        message=message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fan-out dispatch tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchFanOut:
    async def test_both_channels_succeed(self) -> None:
        ch1 = FakeChannel("ch1")
        ch2 = FakeChannel("ch2")
        dispatcher = NotificationDispatcher(channels=[ch1, ch2])
        update = _make_update()

        result = await dispatcher.dispatch(update)

        assert result.all_succeeded is True
        assert len(ch1.send_status_calls) == 1
        assert len(ch2.send_status_calls) == 1
        # Both channels receive the SAME StatusUpdate object (identity)
        assert ch1.send_status_calls[0] is ch2.send_status_calls[0]

    async def test_one_channel_fails(self) -> None:
        ch_ok = FakeChannel("ok", succeed=True)
        ch_fail = FakeChannel("fail", succeed=False)
        dispatcher = NotificationDispatcher(channels=[ch_ok, ch_fail])
        update = _make_update()

        result = await dispatcher.dispatch(update)

        assert result.any_failed is True
        # Channel A still received the update (failure isolation)
        assert len(ch_ok.send_status_calls) == 1

    async def test_all_channels_fail(self) -> None:
        ch1 = FakeChannel("ch1", succeed=False)
        ch2 = FakeChannel("ch2", succeed=False)
        dispatcher = NotificationDispatcher(channels=[ch1, ch2])
        update = _make_update()

        result = await dispatcher.dispatch(update)

        assert result.all_succeeded is False
        assert result.any_failed is True

    async def test_no_channels(self) -> None:
        dispatcher = NotificationDispatcher(channels=[])
        update = _make_update()

        result = await dispatcher.dispatch(update)

        assert result.all_succeeded is True
        assert result.channel_results == {}

    async def test_with_alerts(self) -> None:
        ch1 = FakeChannel("ch1")
        ch2 = FakeChannel("ch2")
        alerts = [
            _make_alert(AlertType.STALL, AlertSeverity.WARN, "R0", "stall alert"),
            _make_alert(AlertType.CRASH, AlertSeverity.ERROR, "R1", "crash alert"),
        ]
        update = _make_update(alerts=alerts)
        dispatcher = NotificationDispatcher(channels=[ch1, ch2])

        result = await dispatcher.dispatch(update)

        assert result.alerts_sent == 2
        assert len(ch1.send_alert_calls) == 2
        assert len(ch2.send_alert_calls) == 2

    async def test_alert_delivery_failure_tracking(self) -> None:
        ch_ok = FakeChannel("ok")
        ch_alert_fail = FakeAlertFailChannel("alert_fail")
        alert = _make_alert()
        update = _make_update(alerts=[alert])
        dispatcher = NotificationDispatcher(channels=[ch_ok, ch_alert_fail])

        result = await dispatcher.dispatch(update)

        # Alert was sent to at least one channel
        assert result.alerts_sent == 1
        # One channel failed to deliver
        assert result.alerts_suppressed == 1

    async def test_channels_receive_identical_data(self) -> None:
        """NOT-06 compliance: both channels get the same data."""
        ch1 = FakeChannel("ch1")
        ch2 = FakeChannel("ch2")
        alert = _make_alert(message="test identical")
        update = _make_update(n_snapshots=3, alerts=[alert])
        dispatcher = NotificationDispatcher(channels=[ch1, ch2])

        await dispatcher.dispatch(update)

        u1 = ch1.send_status_calls[0]
        u2 = ch2.send_status_calls[0]
        assert u1.experiment_name == u2.experiment_name
        assert len(u1.snapshots) == len(u2.snapshots) == 3
        assert u1.snapshots[0].generation == u2.snapshots[0].generation
        assert u1.snapshots[0].metrics == u2.snapshots[0].metrics
        assert u1.alerts[0].message == u2.alerts[0].message
