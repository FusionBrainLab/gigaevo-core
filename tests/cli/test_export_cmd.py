"""Tests for CLI export sub-group: csv and frontier commands."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd


def _make_evolution_df(n_rows: int = 20, label: str = "A") -> pd.DataFrame:
    """Create a small DataFrame mimicking fetch_evolution_dataframe output."""
    return pd.DataFrame(
        {
            "id": [f"prog_{label}_{i}" for i in range(n_rows)],
            "iteration": list(range(1, n_rows + 1)),
            "metric_fitness": [0.1 + 0.02 * i for i in range(n_rows)],
            "generation": [(i // 5) + 1 for i in range(n_rows)],
        }
    )


def _make_iteration_df(n_rows: int = 20) -> pd.DataFrame:
    """Create a small DataFrame mimicking prepare_iteration_dataframe output."""
    iterations = list(range(1, n_rows + 1))
    fitness = [0.1 + 0.02 * i for i in range(n_rows)]
    cummax = pd.Series(fitness).cummax().tolist()
    return pd.DataFrame(
        {
            "iteration": iterations,
            "metric_fitness": fitness,
            "running_mean_fitness": fitness,
            "running_std_fitness": [0.01] * n_rows,
            "running_mean_plus_std": [f + 0.01 for f in fitness],
            "running_mean_minus_std": [f - 0.01 for f in fitness],
            "frontier_fitness": cummax,
        }
    )


class TestExportGroupHelp:
    def test_export_help_exits_zero(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0

    def test_export_help_lists_subcommands(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["export", "--help"])
        assert "csv" in result.output
        assert "frontier" in result.output


class TestCsvExportCommand:
    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_csv_creates_file(self, mock_fetch, tmp_path):
        """csv command creates a CSV file at the given path."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(10)

        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "export",
                "csv",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert output_file.exists()

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_csv_has_expected_columns(self, mock_fetch, tmp_path):
        """csv output contains the expected columns from the source DataFrame."""
        from gigaevo.cli import main

        df = _make_evolution_df(5)
        mock_fetch.return_value = df

        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "csv", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        written = pd.read_csv(output_file)
        assert "id" in written.columns
        assert "metric_fitness" in written.columns
        assert len(written) == 5

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_csv_outputs_json_summary(self, mock_fetch, tmp_path):
        """csv command echoes a JSON summary."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(8)

        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "csv", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        summary = json.loads(result.output.strip())
        assert "output_file" in summary
        assert summary["rows"] == 8

    def test_csv_requires_output_file(self):
        """csv fails without -o flag."""
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "csv"],
        )
        assert result.exit_code != 0

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_csv_serializes_complex_columns(self, mock_fetch, tmp_path):
        """csv command serializes dict/list columns as JSON strings."""
        from gigaevo.cli import main

        df = pd.DataFrame(
            {
                "id": ["p1", "p2"],
                "iteration": [1, 2],
                "metric_fitness": [0.5, 0.6],
                "extra_dict": [{"key": "val"}, {"key": "val2"}],
            }
        )
        mock_fetch.return_value = df

        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "csv", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        written = pd.read_csv(output_file)
        parsed = json.loads(written["extra_dict"].iloc[0])
        assert parsed["key"] == "val"


class TestFrontierExportCommand:
    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_creates_file(self, mock_fetch, tmp_path):
        """frontier command creates a CSV file at the given path."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(15)

        output_file = tmp_path / "frontier.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "export",
                "frontier",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert output_file.exists()

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_has_gen_and_best_val_columns(self, mock_fetch, tmp_path):
        """frontier CSV has gen and best_val columns."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(10)

        output_file = tmp_path / "frontier.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "frontier", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        written = pd.read_csv(output_file)
        assert "gen" in written.columns
        assert "best_val" in written.columns

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_outputs_json_summary(self, mock_fetch, tmp_path):
        """frontier command echoes a JSON summary."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(10)

        output_file = tmp_path / "frontier.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "frontier", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        summary = json.loads(result.output.strip())
        assert "output_file" in summary
        assert "generations" in summary

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_custom_metric(self, mock_fetch, tmp_path):
        """frontier accepts --metric to change the fitness column."""
        from gigaevo.cli import main

        df = _make_evolution_df(10)
        df["metric_accuracy"] = [0.5 + 0.01 * i for i in range(10)]
        mock_fetch.return_value = df

        output_file = tmp_path / "frontier.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "export",
                "frontier",
                "-o",
                str(output_file),
                "--metric",
                "accuracy",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        written = pd.read_csv(output_file)
        assert "best_val" in written.columns

    def test_frontier_requires_output_file(self):
        """frontier fails without -o flag."""
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "export", "frontier"],
        )
        assert result.exit_code != 0
