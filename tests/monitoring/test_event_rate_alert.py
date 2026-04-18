"""Tests for Track B4: EVENT_RATE_ZERO AlertDetector predicate.

The watchdog fires one generic alert when a registered canonical event with
``expected_after_gen > 0`` has zero count in the recent window AND the run's
generation has passed that threshold. One predicate, all events.

The predicate reads event counts off ``RunSnapshot.event_window_counts`` so
``AlertDetector`` does not need a Redis connection — ``collect_snapshot``
populates that dict.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from gigaevo.monitoring.alerts import (
    AlertDetector,
    AlertSeverity,
    AlertType,
)
from gigaevo.monitoring.events import BaseEvent
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot


class _TestHofRotate(BaseEvent):
    """Stand-in for HOF_ROTATE: expected to emit by gen >= 2."""

    event: ClassVar[str] = "__B4_HOF_ROTATE__"
    description: ClassVar[str] = "test"
    health_question: ClassVar[str] = "Is the archive rotating?"
    expected_after_gen: ClassVar[int] = 2


class _TestGenerationBoundary(BaseEvent):
    """Stand-in for GENERATION_BOUNDARY: expected_after_gen=0 → never flagged."""

    event: ClassVar[str] = "__B4_GEN__"
    description: ClassVar[str] = "test"
    health_question: ClassVar[str] = "?"
    expected_after_gen: ClassVar[int] = 0


def _snap(
    *,
    label: str = "A",
    generation: int = 5,
    event_counts: dict[str, int] | None = None,
) -> RunSnapshot:
    return RunSnapshot(
        run_spec=RunSpec(prefix="p", db=3, label=label),
        generation=generation,
        event_window_counts=event_counts,
    )


class TestEventRateZeroPredicate:
    def test_fires_when_gen_past_threshold_and_count_zero(self):
        detector = AlertDetector()
        # gen=5, threshold=2, count=0 → ALERT
        snap = _snap(generation=5, event_counts={"__B4_HOF_ROTATE__": 0})
        alerts = detector.check([snap])

        rate_zero = [a for a in alerts if a.alert_type == AlertType.EVENT_RATE_ZERO]
        assert len(rate_zero) == 1
        alert = rate_zero[0]
        assert alert.severity == AlertSeverity.WARN
        assert alert.run_label == "A"
        # Health question from the registry surfaces in the message body.
        assert "__B4_HOF_ROTATE__" in alert.message
        # Details carry enough structured info for downstream consumers.
        assert alert.details is not None
        assert alert.details.get("event_name") == "__B4_HOF_ROTATE__"
        assert alert.details.get("generation") == 5
        assert alert.details.get("expected_after_gen") == 2

    def test_silent_when_gen_below_threshold(self):
        detector = AlertDetector()
        # gen=1, threshold=2 → still too early; no alert.
        snap = _snap(generation=1, event_counts={"__B4_HOF_ROTATE__": 0})
        alerts = detector.check([snap])

        assert not any(a.alert_type == AlertType.EVENT_RATE_ZERO for a in alerts)

    def test_silent_when_count_is_positive(self):
        detector = AlertDetector()
        snap = _snap(generation=5, event_counts={"__B4_HOF_ROTATE__": 7})
        alerts = detector.check([snap])

        assert not any(a.alert_type == AlertType.EVENT_RATE_ZERO for a in alerts)

    def test_silent_when_expected_after_gen_is_zero(self):
        """Events with expected_after_gen=0 never fire — they're either emitted
        from the very start or intentionally on-demand. The watchdog must not
        spam for GENERATION_BOUNDARY-class events."""
        detector = AlertDetector()
        snap = _snap(generation=5, event_counts={"__B4_GEN__": 0})
        alerts = detector.check([snap])

        assert not any(
            a.alert_type == AlertType.EVENT_RATE_ZERO and "__B4_GEN__" in a.message
            for a in alerts
        )

    def test_silent_when_event_counts_is_none(self):
        """Backward compatibility: snapshots without event counts collected
        (e.g. Redis unreachable at collect time) must not trigger."""
        detector = AlertDetector()
        snap = _snap(generation=5, event_counts=None)
        alerts = detector.check([snap])

        assert not any(a.alert_type == AlertType.EVENT_RATE_ZERO for a in alerts)

    def test_cooldown_suppresses_repeat_alerts(self):
        """Once an EVENT_RATE_ZERO alert fires, the standard cooldown logic
        suppresses it for N cycles — otherwise the watchdog would spam
        notifications for every poll while the condition persists."""
        detector = AlertDetector(cooldown_cycles=2)
        snap = _snap(generation=5, event_counts={"__B4_HOF_ROTATE__": 0})

        first = detector.check([snap])
        second = detector.check([snap])

        first_rate = [a for a in first if a.alert_type == AlertType.EVENT_RATE_ZERO]
        second_rate = [a for a in second if a.alert_type == AlertType.EVENT_RATE_ZERO]
        assert len(first_rate) == 1
        assert len(second_rate) == 0, "cooldown should suppress immediate repeats"

    def test_fires_per_run_independently(self):
        detector = AlertDetector()
        a = _snap(label="A", generation=5, event_counts={"__B4_HOF_ROTATE__": 0})
        b = _snap(label="B", generation=5, event_counts={"__B4_HOF_ROTATE__": 3})
        alerts = detector.check([a, b])

        rate = [x for x in alerts if x.alert_type == AlertType.EVENT_RATE_ZERO]
        assert len(rate) == 1
        assert rate[0].run_label == "A"


@pytest.mark.parametrize(
    "gen,threshold,count,should_fire",
    [
        (0, 2, 0, False),  # before threshold
        (1, 2, 0, False),  # still before
        (2, 2, 0, True),  # exactly at threshold
        (5, 2, 0, True),  # well past
        (5, 2, 1, False),  # seen at least once
        (5, 0, 0, False),  # threshold=0 never fires
    ],
)
def test_predicate_matrix(gen, threshold, count, should_fire, monkeypatch):
    """Tabular spec for the predicate across edge cases."""

    # Register a scratch event with the parametric threshold.
    class _Ev(BaseEvent):
        event: ClassVar[str] = f"__B4_TMP_{gen}_{threshold}_{count}__"
        description: ClassVar[str] = "matrix"
        health_question: ClassVar[str] = "?"
        expected_after_gen: ClassVar[int] = threshold

    detector = AlertDetector()
    snap = _snap(generation=gen, event_counts={_Ev.event: count})
    alerts = detector.check([snap])

    rate = [
        a
        for a in alerts
        if a.alert_type == AlertType.EVENT_RATE_ZERO
        and a.details is not None
        and a.details.get("event_name") == _Ev.event
    ]
    assert bool(rate) == should_fire
