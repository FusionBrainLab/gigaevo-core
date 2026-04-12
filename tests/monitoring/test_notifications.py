"""Tests for gigaevo.monitoring.notifications — data model, ABC, and formatters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gigaevo.monitoring import Alert, AlertSeverity, AlertType, RunSnapshot
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    PlotAttachment,
    StatusUpdate,
    format_alert_message,
    format_status_table_markdown,
    format_status_table_telegram,
)
from gigaevo.monitoring.run_spec import RunSpec

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
        ts = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

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


# ═══════════════════════════════════════════════════════════════════════════════
# 6. format_status_table_markdown tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatStatusTableMarkdown:
    def test_single_snapshot_produces_table(self) -> None:
        md = format_status_table_markdown([_make_snapshot()])
        lines = md.strip().split("\n")
        assert len(lines) == 3  # header, separator, 1 data row
        assert "|" in lines[0]
        assert "---" in lines[1]
        assert "|" in lines[2]

    def test_multiple_snapshots_produce_rows(self) -> None:
        snaps = [
            _make_snapshot(label="A", db=1),
            _make_snapshot(label="B", db=2),
            _make_snapshot(label="C", db=3),
        ]
        md = format_status_table_markdown(snaps)
        lines = md.strip().split("\n")
        assert len(lines) == 5  # header + separator + 3 data rows
        for line in lines[2:]:
            col_count = line.count("|")
            assert col_count == lines[0].count("|")

    def test_columns_present(self) -> None:
        md = format_status_table_markdown([_make_snapshot()])
        header = md.split("\n")[0]
        for col in [
            "Run",
            "DB",
            "Gen",
            "Fitness",
            "Invalid%",
            "Val dur(s)",
            "Keys",
            "PID",
            "Status",
        ]:
            assert col in header

    def test_fitness_formatting(self) -> None:
        md = format_status_table_markdown([_make_snapshot(fitness=0.762)])
        assert "0.7620" in md

    def test_fitness_formatting_small_value(self) -> None:
        md = format_status_table_markdown([_make_snapshot(fitness=0.03450)])
        assert "0.03450" in md

    def test_invalid_rate_formatting(self) -> None:
        md = format_status_table_markdown(
            [_make_snapshot(invalid_rate_inputs=(100, 80))]
        )
        assert "20%" in md

    def test_pid_alive_status(self) -> None:
        md = format_status_table_markdown([_make_snapshot(pid=49341, pid_alive=True)])
        assert "ALIVE" in md

    def test_pid_dead_status(self) -> None:
        md = format_status_table_markdown([_make_snapshot(pid=49341, pid_alive=False)])
        assert "DEAD" in md

    def test_pid_unknown(self) -> None:
        md = format_status_table_markdown([_make_snapshot(pid=None, pid_alive=None)])
        data_row = md.strip().split("\n")[2]
        cells = [c.strip() for c in data_row.split("|") if c.strip()]
        # PID column (index 7) and Status column (index 8) should be "-"
        assert cells[7] == "-"
        assert cells[8] == "-"

    def test_missing_generation(self) -> None:
        md = format_status_table_markdown([_make_snapshot(generation=None)])
        data_row = md.strip().split("\n")[2]
        cells = [c.strip() for c in data_row.split("|") if c.strip()]
        assert cells[2] == "-"  # Gen column

    def test_missing_fitness(self) -> None:
        md = format_status_table_markdown([_make_snapshot(fitness=None)])
        data_row = md.strip().split("\n")[2]
        cells = [c.strip() for c in data_row.split("|") if c.strip()]
        assert cells[3] == "-"  # Fitness column

    def test_validator_duration_format(self) -> None:
        md = format_status_table_markdown(
            [_make_snapshot(val_mean=639.0, val_max=980.0)]
        )
        assert "639/980" in md

    def test_empty_snapshots(self) -> None:
        md = format_status_table_markdown([])
        assert "No runs" in md


# ═══════════════════════════════════════════════════════════════════════════════
# 7. format_status_table_telegram tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatStatusTableTelegram:
    def test_wrapped_in_pre_tags(self) -> None:
        tg = format_status_table_telegram([_make_snapshot()])
        assert tg.startswith("<pre>")
        assert tg.endswith("</pre>")

    def test_contains_same_columns(self) -> None:
        tg = format_status_table_telegram([_make_snapshot()])
        for col in ["Run", "DB", "Gen", "Fitness", "Keys", "PID", "Status"]:
            assert col in tg

    def test_same_data_values(self) -> None:
        tg = format_status_table_telegram([_make_snapshot(fitness=0.762)])
        assert "0.7620" in tg
        assert "ALIVE" in tg

    def test_monospace_alignment(self) -> None:
        snaps = [
            _make_snapshot(label="A", db=1),
            _make_snapshot(label="BB", db=22),
        ]
        tg = format_status_table_telegram(snaps)
        # Strip <pre> tags and get data lines
        inner = tg.replace("<pre>\n", "").replace("\n</pre>", "")
        lines = inner.split("\n")
        # All non-separator lines should have the same length
        data_lines = [row for row in lines if not all(c in "-  " for c in row)]
        lengths = [len(row) for row in data_lines]
        assert len(set(lengths)) == 1, f"Unequal line lengths: {lengths}"

    def test_empty_snapshots(self) -> None:
        tg = format_status_table_telegram([])
        assert "<pre>" in tg
        assert "No runs" in tg


# ═══════════════════════════════════════════════════════════════════════════════
# 8. format_alert_message tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatAlertMessage:
    def test_info_severity(self) -> None:
        msg = format_alert_message(
            _make_alert(severity=AlertSeverity.INFO, message="All done")
        )
        assert msg.startswith("[INFO]")

    def test_warn_severity(self) -> None:
        msg = format_alert_message(
            _make_alert(severity=AlertSeverity.WARN, message="Stalled")
        )
        assert msg.startswith("[WARNING]")

    def test_error_severity(self) -> None:
        msg = format_alert_message(
            _make_alert(severity=AlertSeverity.ERROR, message="Crashed")
        )
        assert msg.startswith("[ERROR]")

    def test_contains_type_and_label_and_message(self) -> None:
        msg = format_alert_message(
            _make_alert(
                alert_type=AlertType.STALL,
                run_label="X",
                message="Run X stalled at gen 10",
            )
        )
        assert "stall" in msg
        assert "Run X stalled at gen 10" in msg

    def test_stall_includes_generation(self) -> None:
        msg = format_alert_message(
            _make_alert(
                alert_type=AlertType.STALL,
                message="Run A stalled at gen 7",
            )
        )
        assert "gen 7" in msg

    def test_crash_includes_pid(self) -> None:
        msg = format_alert_message(
            _make_alert(
                alert_type=AlertType.CRASH,
                severity=AlertSeverity.ERROR,
                message="Run B process (PID 12345) is not alive",
            )
        )
        assert "12345" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Both-formatters-same-data test (NOT-06)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatterConsistency:
    def test_both_formatters_same_data(self) -> None:
        snaps = [
            _make_snapshot(label="A", db=1, generation=5, fitness=0.762),
            _make_snapshot(label="B", db=2, generation=10, fitness=0.881),
            _make_snapshot(label="C", db=3, generation=15, fitness=0.550),
        ]

        md = format_status_table_markdown(snaps)
        tg = format_status_table_telegram(snaps)

        # Extract data values from markdown
        md_lines = md.strip().split("\n")[2:]  # skip header + separator
        md_labels = []
        md_gens = []
        md_fitness = []
        for line in md_lines:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            md_labels.append(cells[0])
            md_gens.append(cells[2])
            md_fitness.append(cells[3])

        # Extract data values from telegram
        inner = tg.replace("<pre>\n", "").replace("\n</pre>", "")
        tg_lines = inner.split("\n")[2:]  # skip header + separator
        tg_labels = []
        tg_gens = []
        tg_fitness = []
        for line in tg_lines:
            parts = line.split()
            tg_labels.append(parts[0])
            tg_gens.append(parts[2])
            tg_fitness.append(parts[3])

        assert md_labels == tg_labels
        assert md_gens == tg_gens
        assert md_fitness == tg_fitness
