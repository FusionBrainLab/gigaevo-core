"""Tests for the top CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import fakeredis

from gigaevo.cli import main
from gigaevo.cli.run_resolver import RunResolver


def _store_program(
    server: fakeredis.FakeServer,
    db: int,
    prefix: str,
    prog_id: str,
    generation: int,
    fitness: float,
    code: str = "def solve(): pass",
    state: str = "DONE",
) -> None:
    """Store a program JSON blob in fakeredis."""
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    prog = {
        "id": prog_id,
        "generation": generation,
        "metrics": {"fitness": fitness},
        "state": state,
        "code": code,
    }
    r.set(f"{prefix}:program:{prog_id}", json.dumps(prog))


def _make_obj(server: fakeredis.FakeServer) -> dict:
    return {
        "redis_factory": lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        ),
    }


class TestTopBasic:
    def test_json_output_ranked_by_fitness(self):
        """Top returns programs ranked highest fitness first."""
        server = fakeredis.FakeServer()
        _store_program(server, 4, "p", "aaa111111111", 1, 0.50)
        _store_program(server, 4, "p", "bbb222222222", 2, 0.80)
        _store_program(server, 4, "p", "ccc333333333", 3, 0.65)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "json", "top", "-n", "3"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 3
        assert data[0]["Rank"] == 1
        assert data[0]["Fitness"] == 0.80

    def test_top_n_limits_results(self):
        """--top-n limits the number of returned programs."""
        server = fakeredis.FakeServer()
        for i in range(10):
            _store_program(server, 4, "p", f"prog{i:012d}", i, 0.10 * i)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "json", "top", "-n", "3"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 3

    def test_table_output_contains_id(self):
        """Table output contains truncated program IDs."""
        server = fakeredis.FakeServer()
        _store_program(server, 4, "p", "abcdef123456", 1, 0.75)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "table", "top", "-n", "1"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "abcdef123456" in result.output


class TestTopCodeFlag:
    def test_show_code_prints_source(self):
        """--code flag prints program source code."""
        server = fakeredis.FakeServer()
        _store_program(
            server, 4, "p", "prog00000001", 1, 0.90, code="def solve(): return 42"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "table", "top", "-n", "1", "--code"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "return 42" in result.output


class TestTopSaveDir:
    def test_save_dir_creates_files(self, tmp_path: Path):
        """--save-dir writes program source to files."""
        server = fakeredis.FakeServer()
        _store_program(
            server, 4, "p", "prog00000001", 1, 0.90, code="def solve(): return 42"
        )

        runner = CliRunner()
        save_dir = str(tmp_path / "saved")
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "table", "top", "-n", "1", "--save-dir", save_dir],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        saved_files = list(Path(save_dir).glob("*.py"))
        assert len(saved_files) == 1
        assert "return 42" in saved_files[0].read_text()


class TestTopEmptyRedis:
    def test_empty_redis_no_crash(self):
        """Empty Redis returns empty table, no crash."""
        server = fakeredis.FakeServer()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "json", "top"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data == []


class TestTopMinimize:
    def test_minimize_reverses_sort(self):
        """--minimize sorts lowest fitness first."""
        server = fakeredis.FakeServer()
        _store_program(server, 4, "p", "aaa111111111", 1, 0.10)
        _store_program(server, 4, "p", "bbb222222222", 2, 0.90)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:A", "-f", "json", "top", "--minimize"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data[0]["Fitness"] == 0.10


class TestTopManifestDefaultMetric:
    def test_experiment_mode_uses_manifest_metric_name(self):
        """When --experiment is used and manifest has problem.metric_name, top ranks by it."""
        from unittest.mock import MagicMock, patch

        from gigaevo.monitoring.experiment_monitor import RunConfig
        from gigaevo.monitoring.run_spec import RunSpec

        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=4, decode_responses=True)
        prog = {
            "id": "aaa111111111",
            "generation": 1,
            "metrics": {"fitness": 0.50, "actual_fitness": 0.80},
            "state": "DONE",
            "code": "def solve(): pass",
        }
        r.set("p:program:aaa111111111", json.dumps(prog))

        mock_manifest = MagicMock()
        mock_manifest.contract.problem.metric_name = "actual_fitness"

        configs = [
            RunConfig(
                run_spec=RunSpec(prefix="p", db=4, label="A"),
                metric_names=["actual_fitness"],
            ),
        ]

        obj = _make_obj(server)
        obj["experiment"] = "test/exp"
        obj["runs"] = ()

        with (
            patch(
                "gigaevo.experiment.manifest.load_manifest",
                return_value=mock_manifest,
            ),
            patch.object(RunResolver, "resolve", return_value=configs),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "-f", "json", "top", "-n", "1"],
                obj=obj,
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert len(data) == 1
            assert (
                "Actual_Fitness" in data[0] or "actual_fitness" in str(data[0]).lower()
            )

    def test_explicit_metric_overrides_manifest(self):
        """Explicit --metric quality overrides manifest default."""
        from unittest.mock import MagicMock, patch

        from gigaevo.monitoring.experiment_monitor import RunConfig
        from gigaevo.monitoring.run_spec import RunSpec

        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=4, decode_responses=True)
        prog = {
            "id": "aaa111111111",
            "generation": 1,
            "metrics": {"fitness": 0.50, "quality": 0.95, "actual_fitness": 0.80},
            "state": "DONE",
            "code": "def solve(): pass",
        }
        r.set("p:program:aaa111111111", json.dumps(prog))

        mock_manifest = MagicMock()
        mock_manifest.contract.problem.metric_name = "actual_fitness"

        configs = [
            RunConfig(
                run_spec=RunSpec(prefix="p", db=4, label="A"),
            ),
        ]

        obj = _make_obj(server)
        obj["experiment"] = "test/exp"
        obj["runs"] = ()

        with (
            patch(
                "gigaevo.experiment.manifest.load_manifest",
                return_value=mock_manifest,
            ),
            patch.object(RunResolver, "resolve", return_value=configs),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-e",
                    "test/exp",
                    "-f",
                    "json",
                    "top",
                    "-n",
                    "1",
                    "--metric",
                    "quality",
                ],
                obj=obj,
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            data = json.loads(result.output)
            assert len(data) == 1
            assert "Quality" in data[0]
            assert data[0]["Quality"] == 0.95
