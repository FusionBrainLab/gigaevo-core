"""Tests for AlertDetector with multi-signal stall detection."""

from __future__ import annotations

import pytest

from gigaevo.monitoring.alerts import Alert, AlertDetector, AlertSeverity, AlertType
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot


def make_snapshot(
    label: str = "O",
    prefix: str = "chains/hotpotqa/static",
    db: int = 4,
    generation: int | None = 10,
    fitness: float | None = 0.65,
    total_programs: int | None = 100,
    valid_programs: int | None = 90,
    running_programs: int | None = 2,
    queued_programs: int | None = 0,
    done_programs: int | None = 98,
    pid: int | None = None,
    pid_alive: bool | None = None,
    error: str | None = None,
) -> RunSnapshot:
    metrics = {"fitness": fitness} if fitness is not None else {}
    return RunSnapshot(
        run_spec=RunSpec(prefix=prefix, db=db, label=label),
        generation=generation,
        metrics=metrics,
        total_programs=total_programs,
        valid_programs=valid_programs,
        running_programs=running_programs,
        queued_programs=queued_programs,
        done_programs=done_programs,
        pid=pid,
        pid_alive=pid_alive,
        error=error,
    )


# ---------------------------------------------------------------------------
# 1. Stall detection tests
# ---------------------------------------------------------------------------


class TestStallDetection:
    def test_stall_detected_multi_signal(self):
        """Previous: gen=10, running=2, total=100. Current: gen=10, running=0, total=100.
        Two consecutive snapshots with no progress -> STALL alert with WARN severity."""
        detector = AlertDetector()
        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        curr = make_snapshot(generation=10, running_programs=0, total_programs=100)

        # First call sets previous state
        detector.check([prev])
        # Second call detects stall
        alerts = detector.check([curr])

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.alert_type == AlertType.STALL
        assert alert.severity == AlertSeverity.WARN
        assert alert.run_label == "O"

    def test_no_stall_when_generation_advances(self):
        """Previous: gen=10. Current: gen=11. No stall even if running=0."""
        detector = AlertDetector()
        prev = make_snapshot(generation=10, running_programs=0, total_programs=100)
        curr = make_snapshot(generation=11, running_programs=0, total_programs=100)

        detector.check([prev])
        alerts = detector.check([curr])

        stall_alerts = [a for a in alerts if a.alert_type == AlertType.STALL]
        assert len(stall_alerts) == 0

    def test_no_stall_when_programs_still_running(self):
        """Previous: gen=10, running=2, total=100. Current: gen=10, running=2, total=100.
        Running programs > 0 means work is in progress, not stalled."""
        detector = AlertDetector()
        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        curr = make_snapshot(generation=10, running_programs=2, total_programs=100)

        detector.check([prev])
        alerts = detector.check([curr])

        stall_alerts = [a for a in alerts if a.alert_type == AlertType.STALL]
        assert len(stall_alerts) == 0

    def test_no_stall_when_new_programs_submitted(self):
        """Previous: gen=10, total=100. Current: gen=10, running=0, total=105.
        New programs submitted (total increased) means progress."""
        detector = AlertDetector()
        prev = make_snapshot(generation=10, running_programs=0, total_programs=100)
        curr = make_snapshot(generation=10, running_programs=0, total_programs=105)

        detector.check([prev])
        alerts = detector.check([curr])

        stall_alerts = [a for a in alerts if a.alert_type == AlertType.STALL]
        assert len(stall_alerts) == 0

    def test_no_stall_on_first_check(self):
        """No previous snapshots -> no stall detection possible."""
        detector = AlertDetector()
        curr = make_snapshot(generation=10, running_programs=0, total_programs=100)

        alerts = detector.check([curr])

        stall_alerts = [a for a in alerts if a.alert_type == AlertType.STALL]
        assert len(stall_alerts) == 0

    def test_stall_requires_all_signals(self):
        """Only gen unchanged is not enough. Only running=0 is not enough.
        Only total unchanged is not enough. Must be all three."""
        detector = AlertDetector()

        # gen unchanged, running > 0, total unchanged -> no stall
        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        curr = make_snapshot(generation=10, running_programs=2, total_programs=100)
        detector.check([prev])
        alerts = detector.check([curr])
        assert not any(a.alert_type == AlertType.STALL for a in alerts)

        # gen unchanged, running=0, total increased -> no stall
        detector2 = AlertDetector()
        prev2 = make_snapshot(generation=10, running_programs=0, total_programs=100)
        curr2 = make_snapshot(generation=10, running_programs=0, total_programs=110)
        detector2.check([prev2])
        alerts2 = detector2.check([curr2])
        assert not any(a.alert_type == AlertType.STALL for a in alerts2)

        # gen advances, running=0, total unchanged -> no stall
        detector3 = AlertDetector()
        prev3 = make_snapshot(generation=10, running_programs=0, total_programs=100)
        curr3 = make_snapshot(generation=11, running_programs=0, total_programs=100)
        detector3.check([prev3])
        alerts3 = detector3.check([curr3])
        assert not any(a.alert_type == AlertType.STALL for a in alerts3)


# ---------------------------------------------------------------------------
# 2. Crash detection tests
# ---------------------------------------------------------------------------


class TestCrashDetection:
    def test_crash_detected_pid_dead(self):
        """Snapshot with pid=12345, pid_alive=False -> CRASH alert with ERROR severity."""
        detector = AlertDetector()
        snap = make_snapshot(pid=12345, pid_alive=False)

        alerts = detector.check([snap])

        crash_alerts = [a for a in alerts if a.alert_type == AlertType.CRASH]
        assert len(crash_alerts) == 1
        assert crash_alerts[0].severity == AlertSeverity.ERROR
        assert crash_alerts[0].run_label == "O"

    def test_no_crash_when_pid_alive(self):
        """pid=12345, pid_alive=True -> no crash alert."""
        detector = AlertDetector()
        snap = make_snapshot(pid=12345, pid_alive=True)

        alerts = detector.check([snap])

        crash_alerts = [a for a in alerts if a.alert_type == AlertType.CRASH]
        assert len(crash_alerts) == 0

    def test_no_crash_when_no_pid(self):
        """pid=None -> no crash detection (PID not tracked)."""
        detector = AlertDetector()
        snap = make_snapshot(pid=None)

        alerts = detector.check([snap])

        crash_alerts = [a for a in alerts if a.alert_type == AlertType.CRASH]
        assert len(crash_alerts) == 0


# ---------------------------------------------------------------------------
# 3. High-invalidity tests
# ---------------------------------------------------------------------------


class TestHighInvalidity:
    def test_high_invalidity_detected(self):
        """total=100, valid=20 (80% invalid), generation=5 -> HIGH_INVALIDITY alert."""
        detector = AlertDetector()
        snap = make_snapshot(total_programs=100, valid_programs=20, generation=5)

        alerts = detector.check([snap])

        inv_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_INVALIDITY]
        assert len(inv_alerts) == 1
        assert inv_alerts[0].severity == AlertSeverity.WARN
        assert "stage_timeout" in inv_alerts[0].message.lower()

    def test_no_high_invalidity_below_threshold(self):
        """total=100, valid=30 (70% invalid, below 75% default) -> no alert."""
        detector = AlertDetector()
        snap = make_snapshot(total_programs=100, valid_programs=30, generation=5)

        alerts = detector.check([snap])

        inv_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_INVALIDITY]
        assert len(inv_alerts) == 0

    def test_no_high_invalidity_early_generations(self):
        """total=100, valid=10 (90% invalid), generation=1 -> no alert (too early)."""
        detector = AlertDetector()
        snap = make_snapshot(total_programs=100, valid_programs=10, generation=1)

        alerts = detector.check([snap])

        inv_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_INVALIDITY]
        assert len(inv_alerts) == 0

    def test_no_high_invalidity_when_no_data(self):
        """total=None or valid=None -> no alert."""
        detector = AlertDetector()
        snap_no_total = make_snapshot(
            total_programs=None, valid_programs=90, generation=5
        )
        snap_no_valid = make_snapshot(
            total_programs=100, valid_programs=None, generation=5
        )

        alerts1 = detector.check([snap_no_total])
        alerts2 = detector.check([snap_no_valid])

        inv_alerts = [
            a for a in alerts1 + alerts2 if a.alert_type == AlertType.HIGH_INVALIDITY
        ]
        assert len(inv_alerts) == 0

    def test_custom_invalidity_threshold(self):
        """Configure detector with invalidity_threshold=0.5.
        total=100, valid=40 (60% invalid) -> alert."""
        detector = AlertDetector(invalidity_threshold=0.5)
        snap = make_snapshot(total_programs=100, valid_programs=40, generation=5)

        alerts = detector.check([snap])

        inv_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_INVALIDITY]
        assert len(inv_alerts) == 1


# ---------------------------------------------------------------------------
# 4. Completion detection tests
# ---------------------------------------------------------------------------


class TestCompletionDetection:
    def test_completion_detected_all_at_max(self):
        """3 runs, all at generation=50, max_generations=50 -> COMPLETION alert."""
        detector = AlertDetector(max_generations=50)
        snaps = [
            make_snapshot(label="A", generation=50),
            make_snapshot(label="B", generation=50),
            make_snapshot(label="C", generation=50),
        ]

        alerts = detector.check(snaps)

        comp_alerts = [a for a in alerts if a.alert_type == AlertType.COMPLETION]
        assert len(comp_alerts) == 1
        assert comp_alerts[0].severity == AlertSeverity.INFO

    def test_completion_detected_beyond_max(self):
        """Run at generation=55, max=50 -> still counts as complete."""
        detector = AlertDetector(max_generations=50)
        snaps = [make_snapshot(generation=55)]

        alerts = detector.check(snaps)

        comp_alerts = [a for a in alerts if a.alert_type == AlertType.COMPLETION]
        assert len(comp_alerts) == 1

    def test_no_completion_when_some_not_done(self):
        """3 runs, 2 at gen=50, 1 at gen=40, max=50 -> no completion."""
        detector = AlertDetector(max_generations=50)
        snaps = [
            make_snapshot(label="A", generation=50),
            make_snapshot(label="B", generation=50),
            make_snapshot(label="C", generation=40),
        ]

        alerts = detector.check(snaps)

        comp_alerts = [a for a in alerts if a.alert_type == AlertType.COMPLETION]
        assert len(comp_alerts) == 0

    def test_no_completion_when_generation_none(self):
        """A run with generation=None -> not complete."""
        detector = AlertDetector(max_generations=50)
        snaps = [
            make_snapshot(label="A", generation=50),
            make_snapshot(label="B", generation=None),
        ]

        alerts = detector.check(snaps)

        comp_alerts = [a for a in alerts if a.alert_type == AlertType.COMPLETION]
        assert len(comp_alerts) == 0


# ---------------------------------------------------------------------------
# 5. Cooldown tests
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_suppresses_duplicate_alert(self):
        """First call returns STALL. Next 2 calls suppressed. Fourth fires again."""
        detector = AlertDetector(cooldown_cycles=2)
        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        stall = make_snapshot(generation=10, running_programs=0, total_programs=100)

        # Set up previous state
        detector.check([prev])

        # Call 1: STALL fires
        alerts1 = detector.check([stall])
        assert len([a for a in alerts1 if a.alert_type == AlertType.STALL]) == 1

        # Call 2: suppressed (cooldown 2 -> 1)
        alerts2 = detector.check([stall])
        assert len([a for a in alerts2 if a.alert_type == AlertType.STALL]) == 0

        # Call 3: suppressed (cooldown 1 -> 0, deleted)
        alerts3 = detector.check([stall])
        assert len([a for a in alerts3 if a.alert_type == AlertType.STALL]) == 0

        # Call 4: fires again
        alerts4 = detector.check([stall])
        assert len([a for a in alerts4 if a.alert_type == AlertType.STALL]) == 1

    def test_cooldown_per_run_label(self):
        """STALL for run 'O' is suppressed, but STALL for run 'R' is not."""
        detector = AlertDetector(cooldown_cycles=2)

        prev_o = make_snapshot(
            label="O", generation=10, running_programs=2, total_programs=100
        )
        stall_o = make_snapshot(
            label="O", generation=10, running_programs=0, total_programs=100
        )
        prev_r = make_snapshot(
            label="R", generation=10, running_programs=2, total_programs=100
        )
        stall_r = make_snapshot(
            label="R", generation=10, running_programs=0, total_programs=100
        )

        # Set up previous for both
        detector.check([prev_o, prev_r])

        # Both stall: both fire
        alerts1 = detector.check([stall_o, stall_r])
        stall_alerts1 = [a for a in alerts1 if a.alert_type == AlertType.STALL]
        assert len(stall_alerts1) == 2

        # Next call: both suppressed (in cooldown)
        alerts2 = detector.check([stall_o, stall_r])
        stall_alerts2 = [a for a in alerts2 if a.alert_type == AlertType.STALL]
        assert len(stall_alerts2) == 0

    def test_cooldown_per_alert_type(self):
        """STALL for run 'O' is suppressed, but HIGH_INVALIDITY for run 'O' is not."""
        detector = AlertDetector(cooldown_cycles=2)

        # Set up previous for stall detection
        prev = make_snapshot(
            label="O",
            generation=10,
            running_programs=2,
            total_programs=100,
            valid_programs=90,
        )
        # Current: stalled AND high invalidity
        curr = make_snapshot(
            label="O",
            generation=10,
            running_programs=0,
            total_programs=100,
            valid_programs=20,  # 80% invalid
        )

        detector.check([prev])

        # Both fire on first check
        alerts1 = detector.check([curr])
        stall1 = [a for a in alerts1 if a.alert_type == AlertType.STALL]
        inv1 = [a for a in alerts1 if a.alert_type == AlertType.HIGH_INVALIDITY]
        assert len(stall1) == 1
        assert len(inv1) == 1

        # Second check: both suppressed (same cooldown cycle)
        alerts2 = detector.check([curr])
        stall2 = [a for a in alerts2 if a.alert_type == AlertType.STALL]
        inv2 = [a for a in alerts2 if a.alert_type == AlertType.HIGH_INVALIDITY]
        assert len(stall2) == 0
        assert len(inv2) == 0

    def test_cooldown_configurable(self):
        """AlertDetector(cooldown_cycles=5) suppresses for 5 cycles."""
        detector = AlertDetector(cooldown_cycles=5)
        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        stall = make_snapshot(generation=10, running_programs=0, total_programs=100)

        detector.check([prev])

        # Fire on first
        alerts1 = detector.check([stall])
        assert len([a for a in alerts1 if a.alert_type == AlertType.STALL]) == 1

        # Suppressed for next 5 calls
        for i in range(5):
            alerts = detector.check([stall])
            assert len([a for a in alerts if a.alert_type == AlertType.STALL]) == 0, (
                f"Expected suppression on call {i + 2}, but alert fired"
            )

        # Fires again on call 7 (1 + 5 suppressed + 1 fire)
        alerts_final = detector.check([stall])
        assert len([a for a in alerts_final if a.alert_type == AlertType.STALL]) == 1

    def test_cooldown_cycles_semantics(self):
        """cooldown_cycles=3: fire, suppress, suppress, suppress, fire."""
        detector = AlertDetector(cooldown_cycles=3)
        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        stall = make_snapshot(generation=10, running_programs=0, total_programs=100)

        detector.check([prev])

        # Call 1: fires
        assert (
            len([a for a in detector.check([stall]) if a.alert_type == AlertType.STALL])
            == 1
        )
        # Call 2: suppress
        assert (
            len([a for a in detector.check([stall]) if a.alert_type == AlertType.STALL])
            == 0
        )
        # Call 3: suppress
        assert (
            len([a for a in detector.check([stall]) if a.alert_type == AlertType.STALL])
            == 0
        )
        # Call 4: suppress
        assert (
            len([a for a in detector.check([stall]) if a.alert_type == AlertType.STALL])
            == 0
        )
        # Call 5: fires again
        assert (
            len([a for a in detector.check([stall]) if a.alert_type == AlertType.STALL])
            == 1
        )


# ---------------------------------------------------------------------------
# 6. Multiple alerts in one check
# ---------------------------------------------------------------------------


class TestMultipleAlerts:
    def test_multiple_alerts_from_single_check(self):
        """One run is stalled AND has high invalidity -> two alerts returned."""
        detector = AlertDetector()
        prev = make_snapshot(
            generation=10,
            running_programs=2,
            total_programs=100,
            valid_programs=90,
        )
        curr = make_snapshot(
            generation=10,
            running_programs=0,
            total_programs=100,
            valid_programs=20,
        )

        detector.check([prev])
        alerts = detector.check([curr])

        alert_types = {a.alert_type for a in alerts}
        assert AlertType.STALL in alert_types
        assert AlertType.HIGH_INVALIDITY in alert_types

    def test_alerts_from_multiple_runs(self):
        """Two runs: one stalled, one crashed -> two alerts (one per run)."""
        detector = AlertDetector()
        prev_stall = make_snapshot(
            label="O",
            generation=10,
            running_programs=2,
            total_programs=100,
        )
        curr_stall = make_snapshot(
            label="O",
            generation=10,
            running_programs=0,
            total_programs=100,
        )
        crash = make_snapshot(label="R", pid=12345, pid_alive=False)

        detector.check([prev_stall, make_snapshot(label="R")])
        alerts = detector.check([curr_stall, crash])

        labels = {a.run_label for a in alerts}
        assert "O" in labels
        assert "R" in labels
        alert_types = {a.alert_type for a in alerts}
        assert AlertType.STALL in alert_types
        assert AlertType.CRASH in alert_types


# ---------------------------------------------------------------------------
# 7. Alert dataclass tests
# ---------------------------------------------------------------------------


class TestAlertDataclass:
    def test_alert_is_frozen(self):
        """Alert is immutable."""
        alert = Alert(
            alert_type=AlertType.STALL,
            severity=AlertSeverity.WARN,
            run_label="O",
            message="test",
        )
        with pytest.raises(AttributeError):
            alert.message = "changed"  # type: ignore[misc]

    def test_alert_str_representation(self):
        """Alert has meaningful __str__ representation."""
        alert = Alert(
            alert_type=AlertType.STALL,
            severity=AlertSeverity.WARN,
            run_label="O",
            message="Run O stalled at gen 10",
        )
        s = str(alert)
        assert "WARN" in s
        assert "stall" in s
        assert "O" in s

    def test_alert_type_is_enum(self):
        """AlertType values are strings via StrEnum."""
        assert AlertType.STALL == "stall"
        assert AlertType.CRASH == "crash"
        assert AlertType.HIGH_INVALIDITY == "high_invalidity"
        assert AlertType.COMPLETION == "completion"
        assert AlertType.LOW_THROUGHPUT == "low_throughput"
        assert AlertType.MODEL_DRIFT == "model_drift"

    def test_alert_severity_is_enum(self):
        """AlertSeverity values are strings via StrEnum."""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARN == "warn"
        assert AlertSeverity.ERROR == "error"


# ---------------------------------------------------------------------------
# 8. Edge case tests (Task 4)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_stall_not_triggered_when_generation_is_none(self):
        """Both current and previous have generation=None -> no stall.
        Prevents false stall alarms on runs that haven't started."""
        detector = AlertDetector()
        prev = make_snapshot(generation=None, running_programs=0, total_programs=0)
        curr = make_snapshot(generation=None, running_programs=0, total_programs=0)

        detector.check([prev])
        alerts = detector.check([curr])

        stall_alerts = [a for a in alerts if a.alert_type == AlertType.STALL]
        assert len(stall_alerts) == 0

    def test_detector_resets_after_progress(self):
        """Cycle 1: stall. Cycle 2: progress. Cycle 3: stall returns.
        Stall in cycle 3 fires (cooldown from cycle 1 has expired)."""
        detector = AlertDetector(cooldown_cycles=1)

        prev = make_snapshot(generation=10, running_programs=2, total_programs=100)
        stall = make_snapshot(generation=10, running_programs=0, total_programs=100)
        progress = make_snapshot(generation=11, running_programs=2, total_programs=110)
        stall_again = make_snapshot(
            generation=11, running_programs=0, total_programs=110
        )

        # Setup
        detector.check([prev])
        # Cycle 1: stall fires
        alerts1 = detector.check([stall])
        assert any(a.alert_type == AlertType.STALL for a in alerts1)
        # Cycle 2: progress (cooldown 1 -> 0, deleted)
        alerts2 = detector.check([progress])
        assert not any(a.alert_type == AlertType.STALL for a in alerts2)
        # Cycle 3: stall again -> fires (cooldown expired)
        alerts3 = detector.check([stall_again])
        assert any(a.alert_type == AlertType.STALL for a in alerts3)

    def test_high_invalidity_at_exactly_threshold(self):
        """75% invalid at threshold=0.75 -> NOT triggered (strictly greater than).
        At 75.1% -> triggered."""
        detector = AlertDetector(invalidity_threshold=0.75)

        # Exactly at threshold: 25 valid / 100 total = 0.75 invalid -> NOT triggered
        snap_at = make_snapshot(total_programs=100, valid_programs=25, generation=5)
        alerts_at = detector.check([snap_at])
        assert not any(a.alert_type == AlertType.HIGH_INVALIDITY for a in alerts_at)

        # Just above: 24 valid / 100 total = 0.76 invalid -> triggered
        detector2 = AlertDetector(invalidity_threshold=0.75)
        snap_above = make_snapshot(total_programs=100, valid_programs=24, generation=5)
        alerts_above = detector2.check([snap_above])
        assert any(a.alert_type == AlertType.HIGH_INVALIDITY for a in alerts_above)

    def test_completion_with_zero_runs(self):
        """Empty snapshots list with max_generations set -> no completion alert."""
        detector = AlertDetector(max_generations=50)
        alerts = detector.check([])
        assert not any(a.alert_type == AlertType.COMPLETION for a in alerts)

    def test_crash_and_stall_same_run(self):
        """A run that is both stalled AND crashed -> both alerts emitted."""
        detector = AlertDetector()
        prev = make_snapshot(
            generation=10,
            running_programs=2,
            total_programs=100,
            pid=12345,
            pid_alive=True,
        )
        curr = make_snapshot(
            generation=10,
            running_programs=0,
            total_programs=100,
            pid=12345,
            pid_alive=False,
        )

        detector.check([prev])
        alerts = detector.check([curr])

        alert_types = {a.alert_type for a in alerts}
        assert AlertType.STALL in alert_types
        assert AlertType.CRASH in alert_types

    def test_error_snapshot_produces_no_false_alerts(self):
        """A snapshot with error and all fields None -> no stall/crash/invalidity."""
        detector = AlertDetector()
        snap = make_snapshot(
            generation=None,
            total_programs=None,
            valid_programs=None,
            running_programs=None,
            pid=None,
            error="connection refused",
        )

        alerts = detector.check([snap])

        # Should not produce stall, crash, or invalidity alerts
        problematic = [
            a
            for a in alerts
            if a.alert_type
            in (AlertType.STALL, AlertType.CRASH, AlertType.HIGH_INVALIDITY)
        ]
        assert len(problematic) == 0


# ---------------------------------------------------------------------------
# 9. Full lifecycle integration test (Task 4)
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_full_lifecycle(self):
        """Simulate 6 cycles of an experiment with cooldown_cycles=2.

        Cycle 1: All runs healthy -> no alerts
        Cycle 2: One run stalls -> STALL alert (cooldown set to 2)
        Cycle 3: Same run still stalled -> STALL suppressed (2 -> 1)
        Cycle 4: Same run still stalled -> STALL suppressed (1 -> 0, deleted)
        Cycle 5: All runs reach max_gen -> COMPLETION + STALL resumes
        Cycle 6: Completion suppressed (cooldown)
        """
        detector = AlertDetector(max_generations=50, cooldown_cycles=2)

        # Cycle 1: healthy
        healthy_a = make_snapshot(
            label="A", generation=10, running_programs=2, total_programs=100
        )
        healthy_b = make_snapshot(
            label="B", generation=10, running_programs=2, total_programs=100
        )
        alerts1 = detector.check([healthy_a, healthy_b])
        assert len(alerts1) == 0

        # Cycle 2: A stalls
        stall_a = make_snapshot(
            label="A", generation=10, running_programs=0, total_programs=100
        )
        healthy_b2 = make_snapshot(
            label="B", generation=11, running_programs=2, total_programs=110
        )
        alerts2 = detector.check([stall_a, healthy_b2])
        stalls2 = [a for a in alerts2 if a.alert_type == AlertType.STALL]
        assert len(stalls2) == 1
        assert stalls2[0].run_label == "A"

        # Cycle 3: A still stalled -> suppressed
        healthy_b3 = make_snapshot(
            label="B", generation=12, running_programs=2, total_programs=120
        )
        alerts3 = detector.check([stall_a, healthy_b3])
        stalls3 = [a for a in alerts3 if a.alert_type == AlertType.STALL]
        assert len(stalls3) == 0

        # Cycle 4: A still stalled -> suppressed (last suppression)
        healthy_b4 = make_snapshot(
            label="B", generation=13, running_programs=2, total_programs=130
        )
        alerts4 = detector.check([stall_a, healthy_b4])
        stalls4 = [a for a in alerts4 if a.alert_type == AlertType.STALL]
        assert len(stalls4) == 0

        # Cycle 5: all reach max_gen -> COMPLETION + STALL resumes for A
        done_a = make_snapshot(
            label="A", generation=50, running_programs=0, total_programs=100
        )
        done_b = make_snapshot(
            label="B", generation=50, running_programs=2, total_programs=200
        )
        alerts5 = detector.check([done_a, done_b])
        comp5 = [a for a in alerts5 if a.alert_type == AlertType.COMPLETION]
        assert len(comp5) == 1

        # Cycle 6: completion suppressed
        alerts6 = detector.check([done_a, done_b])
        comp6 = [a for a in alerts6 if a.alert_type == AlertType.COMPLETION]
        assert len(comp6) == 0


# ---------------------------------------------------------------------------
# 10. ModelDriftRule tests
# ---------------------------------------------------------------------------


class TestModelDriftRule:
    def test_model_found_returns_none(self):
        """When expected model is in the /models response, returns None."""
        from unittest.mock import patch, MagicMock
        import json

        from gigaevo.monitoring.alerts import ModelDriftRule

        rule = ModelDriftRule(timeout=5)
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": [{"id": "gpt-4"}, {"id": "claude-3"}]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = rule.check("R1", "http://localhost:4000/v1", "gpt-4")

        assert result is None

    def test_model_not_found_returns_alert(self):
        """When expected model is NOT in /models response, returns MODEL_DRIFT alert."""
        from unittest.mock import patch, MagicMock
        import json

        from gigaevo.monitoring.alerts import ModelDriftRule

        rule = ModelDriftRule(timeout=5)
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": [{"id": "gpt-4"}, {"id": "claude-3"}]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = rule.check("R1", "http://localhost:4000/v1", "missing-model")

        assert result is not None
        assert result.alert_type == AlertType.MODEL_DRIFT
        assert result.severity == AlertSeverity.WARN
        assert "missing-model" in result.message
        assert result.run_label == "R1"

    def test_connection_error_returns_alert(self):
        """When /models endpoint is unreachable, returns MODEL_DRIFT alert."""
        from unittest.mock import patch

        from gigaevo.monitoring.alerts import ModelDriftRule

        rule = ModelDriftRule(timeout=1)

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = rule.check("R1", "http://unreachable:4000/v1", "gpt-4")

        assert result is not None
        assert result.alert_type == AlertType.MODEL_DRIFT
        assert "refused" in result.message
