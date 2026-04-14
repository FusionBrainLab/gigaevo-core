"""Tests for the trajectory CLI subcommand."""

from __future__ import annotations

import json

from click.testing import CliRunner
import fakeredis

from gigaevo.cli import main


def _metric_entry(step: int, value: float, ts: int = 123) -> str:
    return json.dumps({"s": step, "v": value, "t": ts, "k": "scalar"})


def _populate_trajectory(
    server: fakeredis.FakeServer,
    db: int,
    prefix: str,
    generations: list[tuple[int, float, float]],
) -> None:
    """Populate fakeredis with gen-by-gen trajectory data.

    Each tuple is (gen, frontier_fitness, mean_fitness).
    """
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    r.hset(f"{prefix}:run_state", "engine:total_generations", str(len(generations)))
    for gen, frontier, mean in generations:
        r.rpush(
            f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness",
            _metric_entry(gen, frontier),
        )
        r.rpush(
            f"{prefix}:metrics:history:program_metrics:valid_gen_fitness_mean",
            _metric_entry(gen, mean),
        )


def _make_obj(server: fakeredis.FakeServer) -> dict:
    """Build ctx.obj with a fakeredis factory."""
    return {
        "redis_factory": lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        ),
    }


class TestTrajectoryBasic:
    def test_json_output_has_per_gen_rows(self):
        """Trajectory returns one row per generation in JSON."""
        server = fakeredis.FakeServer()
        gens = [(1, 0.42, 0.39), (2, 0.55, 0.44), (3, 0.60, 0.50)]
        _populate_trajectory(server, 4, "test/prefix", gens)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@4:O", "-f", "json", "trajectory"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 3
        assert data[0]["Gen"] == 1
        assert data[2]["Gen"] == 3

    def test_table_output_contains_gen_label(self):
        """Trajectory table output contains generation numbers."""
        server = fakeredis.FakeServer()
        gens = [(1, 0.50, 0.40)]
        _populate_trajectory(server, 4, "test/prefix", gens)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@4:O", "-f", "table", "trajectory"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output


class TestTrajectoryTail:
    def test_tail_limits_output(self):
        """--tail N shows only the last N generations."""
        server = fakeredis.FakeServer()
        gens = [(i, 0.40 + i * 0.01, 0.35 + i * 0.01) for i in range(1, 11)]
        _populate_trajectory(server, 4, "test/prefix", gens)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@4:O", "-f", "json", "trajectory", "--tail", "3"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 3
        assert data[0]["Gen"] == 8
        assert data[2]["Gen"] == 10


class TestTrajectoryEmptyRedis:
    def test_empty_redis_shows_no_data(self):
        """Empty Redis produces empty trajectory, no crash."""
        server = fakeredis.FakeServer()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "empty@0:E", "-f", "json", "trajectory"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data == []


class TestTrajectoryMultipleRuns:
    def test_multiple_runs_labeled(self):
        """With multiple --run flags, rows include the run label."""
        server = fakeredis.FakeServer()
        _populate_trajectory(server, 1, "p", [(1, 0.50, 0.40)])
        _populate_trajectory(server, 2, "p", [(1, 0.60, 0.45)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@1:A", "-r", "p@2:B", "-f", "json", "trajectory"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        labels = {row["Label"] for row in data}
        assert labels == {"A", "B"}


class TestTrajectoryMetricOption:
    def test_custom_metric(self):
        """--metric uses a different metric name for frontier/mean."""
        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=4, decode_responses=True)
        r.hset("test/prefix:run_state", "engine:total_generations", "2")
        r.rpush(
            "test/prefix:metrics:history:program_metrics:valid_frontier_accuracy",
            _metric_entry(1, 0.80),
        )
        r.rpush(
            "test/prefix:metrics:history:program_metrics:valid_frontier_accuracy",
            _metric_entry(2, 0.85),
        )
        r.rpush(
            "test/prefix:metrics:history:program_metrics:valid_gen_accuracy_mean",
            _metric_entry(1, 0.70),
        )
        r.rpush(
            "test/prefix:metrics:history:program_metrics:valid_gen_accuracy_mean",
            _metric_entry(2, 0.75),
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@4:O",
                "-f",
                "json",
                "trajectory",
                "--metric",
                "accuracy",
            ],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["Best"] == 0.80
        assert data[1]["Mean"] == 0.75


def _populate_metric_trajectory(
    server: fakeredis.FakeServer,
    db: int,
    prefix: str,
    metric: str,
    generations: list[tuple[int, float, float]],
) -> None:
    """Populate fakeredis with gen-by-gen trajectory data for a specific metric."""
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    for gen, frontier, mean in generations:
        r.rpush(
            f"{prefix}:metrics:history:program_metrics:valid_frontier_{metric}",
            _metric_entry(gen, frontier),
        )
        r.rpush(
            f"{prefix}:metrics:history:program_metrics:valid_gen_{metric}_mean",
            _metric_entry(gen, mean),
        )


class TestTrajectoryMultiMetric:
    def test_multiple_metric_flags_show_both(self):
        """--metric actual_fitness --metric quality shows both metrics in output."""
        server = fakeredis.FakeServer()
        _populate_metric_trajectory(
            server, 4, "test/prefix", "actual_fitness", [(1, 0.70, 0.60)]
        )
        _populate_metric_trajectory(
            server, 4, "test/prefix", "quality", [(1, 0.90, 0.85)]
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@4:O",
                "-f",
                "json",
                "trajectory",
                "--metric",
                "actual_fitness",
                "--metric",
                "quality",
            ],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        metrics_in_output = {row["Metric"] for row in data}
        assert metrics_in_output == {"actual_fitness", "quality"}
        assert len(data) == 2

    def test_auto_discovery_uses_run_config_metric_names(self):
        """When no --metric specified, trajectory auto-discovers from RunConfig.metric_names."""
        from unittest.mock import patch

        from gigaevo.cli.run_resolver import RunResolver
        from gigaevo.monitoring.experiment_monitor import RunConfig
        from gigaevo.monitoring.run_spec import RunSpec

        server = fakeredis.FakeServer()
        _populate_metric_trajectory(
            server, 4, "test/prefix", "fitness", [(1, 0.50, 0.40)]
        )
        _populate_metric_trajectory(
            server, 4, "test/prefix", "actual_fitness", [(1, 0.70, 0.60)]
        )

        configs = [
            RunConfig(
                run_spec=RunSpec(prefix="test/prefix", db=4, label="O"),
                metric_names=["fitness", "actual_fitness"],
            ),
        ]

        with patch.object(RunResolver, "resolve", return_value=configs):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-r", "test/prefix@4:O", "-f", "json", "trajectory"],
                obj=_make_obj(server),
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            metrics_in_output = {row["Metric"] for row in data}
            assert "fitness" in metrics_in_output
            assert "actual_fitness" in metrics_in_output
