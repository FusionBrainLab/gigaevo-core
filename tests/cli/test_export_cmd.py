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


class TestCsvMultiRun:
    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_multi_run_fans_out_paths(self, mock_fetch, tmp_path):
        """Multiple runs → one CSV per label with _<label> suffix in filename."""
        from gigaevo.cli import main

        mock_fetch.side_effect = [
            _make_evolution_df(5, label="A"),
            _make_evolution_df(7, label="B"),
        ]
        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "-r",
                "pfx@1:B",
                "export",
                "csv",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert not output_file.exists()
        assert (tmp_path / "out_A.csv").exists()
        assert (tmp_path / "out_B.csv").exists()

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_multi_run_summary_is_list(self, mock_fetch, tmp_path):
        """Multi-run summary is a JSON list of per-run summaries."""
        from gigaevo.cli import main

        mock_fetch.side_effect = [
            _make_evolution_df(5, label="A"),
            _make_evolution_df(7, label="B"),
        ]
        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "-r",
                "pfx@1:B",
                "export",
                "csv",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        summary = json.loads(result.output.strip())
        assert isinstance(summary, list)
        assert len(summary) == 2
        labels = {s["label"] for s in summary}
        assert labels == {"A", "B"}


class TestCsvPositionalLabel:
    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_positional_label_filters_to_one(self, mock_fetch, tmp_path):
        """Positional label filters resolved runs; single remaining → single output."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(5, label="A")
        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "-r",
                "pfx@1:B",
                "export",
                "csv",
                "A",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        assert mock_fetch.call_count == 1

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_unknown_label_errors_with_known(self, mock_fetch, tmp_path):
        """Unknown label exits with error listing known labels."""
        from gigaevo.cli import main

        output_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "-r",
                "pfx@1:B",
                "export",
                "csv",
                "BOGUS",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code != 0
        assert "BOGUS" in result.output
        assert "A" in result.output
        assert "B" in result.output
        assert mock_fetch.call_count == 0


class TestFrontierMultiRunAndLabel:
    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_multi_run_fans_out(self, mock_fetch, tmp_path):
        """frontier fans out paths for multiple runs."""
        from gigaevo.cli import main

        mock_fetch.side_effect = [
            _make_evolution_df(5, label="A"),
            _make_evolution_df(7, label="B"),
        ]
        output_file = tmp_path / "front.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "-r",
                "pfx@1:B",
                "export",
                "frontier",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert not output_file.exists()
        assert (tmp_path / "front_A.csv").exists()
        assert (tmp_path / "front_B.csv").exists()
        summary = json.loads(result.output.strip())
        assert isinstance(summary, list)
        assert len(summary) == 2

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_positional_label_filters(self, mock_fetch, tmp_path):
        """frontier filters by positional label."""
        from gigaevo.cli import main

        mock_fetch.return_value = _make_evolution_df(10, label="A")
        output_file = tmp_path / "front.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "-r",
                "pfx@1:B",
                "export",
                "frontier",
                "A",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        assert mock_fetch.call_count == 1

    @patch("gigaevo.cli.export._fetch_dataframe")
    def test_frontier_unknown_label_errors(self, mock_fetch, tmp_path):
        """frontier unknown label errors with known labels listed."""
        from gigaevo.cli import main

        output_file = tmp_path / "front.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "pfx@0:A",
                "export",
                "frontier",
                "BOGUS",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code != 0
        assert "BOGUS" in result.output
        assert mock_fetch.call_count == 0
