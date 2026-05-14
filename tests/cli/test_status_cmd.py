"""Tests for the status CLI subcommand."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner
import fakeredis

from gigaevo.cli import main
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from tests.conftest import write_engine_snapshot_sync


def _metric_entry(step: int, value: float, ts: int = 123) -> str:
    return json.dumps({"s": step, "v": value, "t": ts, "k": "scalar"})


def _populate_run(
    server: fakeredis.FakeServer,
    db: int,
    prefix: str,
    generation: int,
    fitness: float,
    total: int,
    valid: int,
) -> None:
    """Populate a fakeredis DB with standard run data."""
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    write_engine_snapshot_sync(r, prefix, total_mutants=generation)
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(generation, fitness),
    )
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:programs_total_count",
        _metric_entry(1, total),
    )
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:programs_valid_count",
        _metric_entry(1, valid),
    )


def _make_invoker(server: fakeredis.FakeServer):
    """Return a function that invokes the CLI with a fakeredis factory injected."""
    runner = CliRunner()

    def invoke(args: list[str]):
        factory = lambda db: fakeredis.FakeRedis(  # noqa: E731
            server=server, db=db, decode_responses=True
        )
        # Inject redis_factory into ctx.obj by monkey-patching main's callback
        original_callback = main.callback

        def patched_callback(ctx, **kwargs):
            original_callback(ctx, **kwargs)
            ctx.obj["redis_factory"] = factory

        # Use obj= to set up initial context, then patch the callback
        return runner.invoke(
            main,
            args,
            catch_exceptions=False,
            obj={"redis_factory": factory},
        )

    return invoke


class TestStatusSingleRun:
    def test_table_output_contains_label(self):
        """Status shows the run label in table output."""
        server = fakeredis.FakeServer()
        _populate_run(
            server, 4, "test/prefix", generation=10, fitness=0.76, total=100, valid=85
        )

        invoke = _make_invoker(server)
        result = invoke(["-r", "test/prefix@4:A", "-f", "table", "status"])
        assert result.exit_code == 0, result.output
        assert "A" in result.output

    def test_json_output_has_expected_fields(self):
        """Status JSON output includes run data from monitoring lib."""
        server = fakeredis.FakeServer()
        _populate_run(
            server, 4, "test/prefix", generation=10, fitness=0.76, total=100, valid=85
        )

        invoke = _make_invoker(server)
        result = invoke(["-r", "test/prefix@4:A", "-f", "json", "status"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        row = data[0]
        assert row["Label"] == "A"
        assert row["Gen"] == 10

    def test_csv_output(self):
        """Status CSV output has header and data row."""
        server = fakeredis.FakeServer()
        _populate_run(
            server, 4, "test/prefix", generation=5, fitness=0.50, total=20, valid=18
        )

        invoke = _make_invoker(server)
        result = invoke(["-r", "test/prefix@4:X", "-f", "csv", "status"])
        assert result.exit_code == 0, result.output
        lines = result.output.strip().split("\n")
        assert len(lines) >= 2  # header + 1 data row
        assert "Label" in lines[0]


class TestStatusMultipleRuns:
    def test_multiple_runs_all_shown(self):
        """Each run gets its own row in the output."""
        server = fakeredis.FakeServer()
        _populate_run(server, 1, "p", generation=10, fitness=0.76, total=100, valid=85)
        _populate_run(server, 2, "p", generation=20, fitness=0.82, total=200, valid=190)

        invoke = _make_invoker(server)
        result = invoke(["-r", "p@1:A", "-r", "p@2:B", "-f", "json", "status"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 2
        labels = {row["Label"] for row in data}
        assert labels == {"A", "B"}


class TestStatusEmptyRedis:
    def test_empty_redis_shows_none_values(self):
        """Empty Redis returns rows with None/null values, not a crash."""
        server = fakeredis.FakeServer()
        # Do NOT populate -- DB is empty

        invoke = _make_invoker(server)
        result = invoke(["-r", "empty@0:E", "-f", "json", "status"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 1
        row = data[0]
        assert row["Label"] == "E"
        assert row["Gen"] is None


class TestStatusUsesMonitoringLib:
    def test_uses_experiment_monitor(self):
        """Status delegates to ExperimentMonitor.collect()."""
        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="M"),
            generation=42,
            metrics={"fitness": 0.99},
            total_programs=500,
            valid_programs=450,
        )

        with patch("gigaevo.cli.status.ExperimentMonitor") as mock_monitor_cls:
            mock_instance = mock_monitor_cls.return_value
            mock_instance.collect.return_value = [snapshot]

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-r", "p@4:M", "-f", "json", "status"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        mock_instance.collect.assert_called_once()
        data = json.loads(result.output)
        assert data[0]["Gen"] == 42


class TestFormatMetricValue:
    """Tests for _format_metric_value with metrics.yaml specs."""

    def test_sentinel_value_displays_na(self):
        """Sentinel value (-1.0) displays as 'N/A' when spec defines sentinel_value."""
        from gigaevo.cli.status import _format_metric_value

        specs = {
            "fitness": {
                "decimals": 5,
                "upper_bound": 1.0,
                "sentinel_value": -1.0,
            }
        }
        assert _format_metric_value(-1.0, "fitness", specs) == "N/A"

    def test_raw_display_even_when_upper_bound_is_1(self):
        """All metrics display as raw float — no percentage conversion."""
        from gigaevo.cli.status import _format_metric_value

        specs = {
            "fitness": {
                "decimals": 5,
                "upper_bound": 1.0,
                "sentinel_value": -1.0,
            }
        }
        assert _format_metric_value(0.85, "fitness", specs) == "0.85000"

    def test_raw_display_for_non_percentage_metric(self):
        """Metric with upper_bound != 1.0 displays as raw value with decimals."""
        from gigaevo.cli.status import _format_metric_value

        specs = {
            "actual_fitness": {
                "decimals": 5,
                "upper_bound": 0.0365,
                "sentinel_value": -1.0,
            }
        }
        assert _format_metric_value(0.02345, "actual_fitness", specs) == "0.02345"

    def test_none_displays_question_mark(self):
        """None metric value displays as '?'."""
        from gigaevo.cli.status import _format_metric_value

        assert _format_metric_value(None, "fitness", {}) == "?"

    def test_no_spec_uses_default_decimals(self):
        """Metric without spec uses 3 decimal places."""
        from gigaevo.cli.status import _format_metric_value

        assert _format_metric_value(0.123456, "unknown_metric", {}) == "0.123"

    def test_sentinel_for_actual_fitness(self):
        """Non-percentage metric sentinel value also displays as 'N/A'."""
        from gigaevo.cli.status import _format_metric_value

        specs = {
            "actual_fitness": {
                "decimals": 5,
                "upper_bound": 0.0365,
                "sentinel_value": -1.0,
            }
        }
        assert _format_metric_value(-1.0, "actual_fitness", specs) == "N/A"


class TestSnapshotToRowWithSpecs:
    """Tests for _snapshot_to_row metric formatting integration."""

    def test_snapshot_row_formats_raw_float(self):
        """_snapshot_to_row shows raw float with spec decimals — never percent."""
        from gigaevo.cli.status import _snapshot_to_row

        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="A"),
            generation=10,
            metrics={"fitness": 0.76},
        )
        specs = {"fitness": {"decimals": 5, "upper_bound": 1.0, "sentinel_value": -1.0}}
        row = _snapshot_to_row(snapshot, metric_specs=specs)
        assert row["Fitness"] == "0.76000"

    def test_snapshot_row_formats_sentinel_as_na(self):
        """_snapshot_to_row shows 'N/A' for sentinel values."""
        from gigaevo.cli.status import _snapshot_to_row

        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="A"),
            generation=10,
            metrics={"fitness": -1.0},
        )
        specs = {"fitness": {"decimals": 5, "upper_bound": 1.0, "sentinel_value": -1.0}}
        row = _snapshot_to_row(snapshot, metric_specs=specs)
        assert row["Fitness"] == "N/A"

    def test_snapshot_row_without_specs_uses_raw_formatted(self):
        """_snapshot_to_row without specs uses default 3-decimal formatting."""
        from gigaevo.cli.status import _snapshot_to_row

        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="A"),
            generation=10,
            metrics={"fitness": 0.76543},
        )
        row = _snapshot_to_row(snapshot)
        assert row["Fitness"] == "0.765"


class TestStatusNoRunFlag:
    def test_missing_run_flag_shows_error(self):
        """Status without --run or --experiment shows usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        assert result.exit_code != 0
