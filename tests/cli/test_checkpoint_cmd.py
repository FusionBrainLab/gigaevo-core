"""Tests for the checkpoint CLI composite command."""

from __future__ import annotations

import json

from click.testing import CliRunner
import fakeredis

from gigaevo.cli import main


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
    r.hset(f"{prefix}:run_state", "engine:total_generations", str(generation))
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
