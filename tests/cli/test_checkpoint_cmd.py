"""Tests for the checkpoint CLI composite command."""

from __future__ import annotations

import json

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
) -> None:
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    write_engine_snapshot_sync(r, prefix, total_mutants=generation)
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness",
        _metric_entry(generation, fitness),
    )
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:programs_total_count",
        _metric_entry(1, 100),
    )
    r.rpush(
        f"{prefix}:metrics:history:program_metrics:programs_valid_count",
        _metric_entry(1, 90),
    )


def _make_obj(server: fakeredis.FakeServer) -> dict:
    return {
        "redis_factory": lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        ),
    }


class TestCheckpointRequiresExperiment:
    def test_no_experiment_shows_error(self):
        """Checkpoint without --experiment shows usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["checkpoint"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "experiment" in result.output.lower()


class TestCheckpointCollectsStatus:
    def test_json_output_with_snapshots(self):
        """Checkpoint collects snapshots and outputs status JSON."""
        server = fakeredis.FakeServer()
        _populate_run(server, 4, "test/prefix", generation=10, fitness=0.76)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@4:A", "-f", "json", "checkpoint", "--no-notify"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["Label"] == "A"
        assert data[0]["Gen"] == 10


class TestCheckpointNoNotify:
    def test_no_notify_skips_dispatch(self):
        """--no-notify returns after status display without importing dispatcher."""
        server = fakeredis.FakeServer()
        _populate_run(server, 4, "test/prefix", generation=10, fitness=0.76)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@4:A", "-f", "json", "checkpoint", "--no-notify"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        # Verify we got data back (dispatch was skipped, not errored)
        data = json.loads(result.output)
        assert len(data) == 1


class TestCheckpointMultipleRuns:
    def test_multiple_runs_all_collected(self):
        """Checkpoint collects data from all runs."""
        server = fakeredis.FakeServer()
        _populate_run(server, 1, "p", generation=10, fitness=0.76)
        _populate_run(server, 2, "p", generation=20, fitness=0.82)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@1:A", "-r", "p@2:B", "-f", "json", "checkpoint", "--no-notify"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 2
        labels = {row["Label"] for row in data}
        assert labels == {"A", "B"}


class TestCheckpointMetricFormatting:
    """Tests for checkpoint _snapshot_to_row metric formatting."""

    def test_sentinel_displays_na(self):
        """Checkpoint formats sentinel values as 'N/A'."""
        from gigaevo.cli.checkpoint import _snapshot_to_row

        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="A"),
            generation=10,
            metrics={"fitness": -1.0},
            total_programs=100,
            valid_programs=90,
        )
        specs = {"fitness": {"decimals": 5, "upper_bound": 1.0, "sentinel_value": -1.0}}
        row = _snapshot_to_row(snapshot, metric_specs=specs)
        assert row["Fitness"] == "N/A"

    def test_raw_display_even_when_upper_bound_is_1(self):
        """Checkpoint formats all metrics as raw float — never percent."""
        from gigaevo.cli.checkpoint import _snapshot_to_row

        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="A"),
            generation=10,
            metrics={"fitness": 0.85},
            total_programs=100,
            valid_programs=90,
        )
        specs = {"fitness": {"decimals": 5, "upper_bound": 1.0, "sentinel_value": -1.0}}
        row = _snapshot_to_row(snapshot, metric_specs=specs)
        assert row["Fitness"] == "0.85000"

    def test_formats_identically_to_status(self):
        """Checkpoint and status format metrics identically."""
        from gigaevo.cli.checkpoint import _snapshot_to_row as checkpoint_to_row
        from gigaevo.cli.status import _snapshot_to_row as status_to_row

        snapshot = RunSnapshot(
            run_spec=RunSpec(prefix="p", db=4, label="A"),
            generation=10,
            metrics={"fitness": 0.76, "actual_fitness": -1.0},
            total_programs=100,
            valid_programs=90,
        )
        specs = {
            "fitness": {"decimals": 5, "upper_bound": 1.0, "sentinel_value": -1.0},
            "actual_fitness": {
                "decimals": 5,
                "upper_bound": 0.0365,
                "sentinel_value": -1.0,
            },
        }
        status_row = status_to_row(snapshot, metric_specs=specs)
        checkpoint_row = checkpoint_to_row(snapshot, metric_specs=specs)
        assert status_row["Fitness"] == checkpoint_row["Fitness"]
        assert status_row["Actual Fitness"] == checkpoint_row["Actual Fitness"]
