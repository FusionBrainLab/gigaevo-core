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
