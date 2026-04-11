"""Tests for CLI plot sub-group: comparison and trajectory commands."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd


def _make_evolution_df(n_rows: int = 20, label: str = "A") -> pd.DataFrame:
    """Create a small DataFrame mimicking fetch_evolution_dataframe output."""
    return pd.DataFrame(
        {
            "id": [f"prog_{label}_{i}" for i in range(n_rows)],
            "metadata_iteration": list(range(1, n_rows + 1)),
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
            "metadata_iteration": iterations,
            "metric_fitness": fitness,
            "running_mean_fitness": fitness,
            "running_std_fitness": [0.01] * n_rows,
            "running_mean_plus_std": [f + 0.01 for f in fitness],
            "running_mean_minus_std": [f - 0.01 for f in fitness],
            "frontier_fitness": cummax,
        }
    )


class TestPlotGroupHelp:
    def test_plot_help_exits_zero(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plot", "--help"])
        assert result.exit_code == 0

    def test_plot_help_lists_subcommands(self):
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plot", "--help"])
        assert "comparison" in result.output
        assert "trajectory" in result.output


class TestMatplotlibLazyImport:
    def test_no_matplotlib_at_module_import(self):
        """Importing plot_group must NOT pull in matplotlib."""
        mpl_keys_before = {k for k in sys.modules if "matplotlib" in k}
        import gigaevo.cli.plot_group  # noqa: F401

        mpl_keys_after = {k for k in sys.modules if "matplotlib" in k}
        new_mpl = mpl_keys_after - mpl_keys_before
        assert new_mpl == set(), f"matplotlib imported at module level: {new_mpl}"


class TestComparisonCommand:
    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_comparison_creates_output_files(self, mock_fetch, tmp_path):
        """comparison command creates png/pdf/svg in output dir."""
        from gigaevo.cli import main

        mock_fetch.return_value = [
            ("A", _make_iteration_df(20)),
        ]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "comparison",
                "-o",
                str(tmp_path),
                "--smoothing",
                "none",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (tmp_path / "evolution_runs_comparison.png").exists()
        assert (tmp_path / "evolution_runs_comparison.pdf").exists()
        assert (tmp_path / "evolution_runs_comparison.svg").exists()

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_comparison_outputs_json_summary(self, mock_fetch, tmp_path):
        """comparison command echoes a JSON summary."""
        from gigaevo.cli import main

        mock_fetch.return_value = [
            ("A", _make_iteration_df(10)),
        ]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "comparison",
                "-o",
                str(tmp_path),
                "--smoothing",
                "none",
            ],
        )
        assert result.exit_code == 0
        summary = json.loads(result.output.strip())
        assert "output_dir" in summary
        assert "runs" in summary

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_comparison_multiple_runs(self, mock_fetch, tmp_path):
        """comparison works with multiple runs."""
        from gigaevo.cli import main

        mock_fetch.return_value = [
            ("A", _make_iteration_df(15)),
            ("B", _make_iteration_df(15)),
        ]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "-r",
                "test/prefix@1:B",
                "plot",
                "comparison",
                "-o",
                str(tmp_path),
                "--smoothing",
                "none",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (tmp_path / "evolution_runs_comparison.png").exists()

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_comparison_smoothing_options(self, mock_fetch, tmp_path):
        """comparison accepts smoothing choices."""
        from gigaevo.cli import main

        mock_fetch.return_value = [("X", _make_iteration_df(30))]

        for method in ("lowess", "ema", "savgol", "gaussian", "rolling", "none"):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-r",
                    "test/prefix@0:X",
                    "plot",
                    "comparison",
                    "-o",
                    str(tmp_path),
                    "--smoothing",
                    method,
                ],
            )
            assert result.exit_code == 0, f"smoothing={method} failed: {result.output}"

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_comparison_custom_metric(self, mock_fetch, tmp_path):
        """comparison accepts --metric flag."""
        from gigaevo.cli import main

        df = _make_iteration_df(15)
        mock_fetch.return_value = [("A", df)]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "comparison",
                "-o",
                str(tmp_path),
                "--metric",
                "fitness",
                "--smoothing",
                "none",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"

    def test_comparison_requires_output_dir(self):
        """comparison fails without -o flag."""
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "plot", "comparison"],
        )
        assert result.exit_code != 0


class TestTrajectoryCommand:
    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_trajectory_creates_output_file(self, mock_fetch, tmp_path):
        """trajectory command creates a png file in output dir."""
        from gigaevo.cli import main

        mock_fetch.return_value = [("A", _make_iteration_df(20))]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "trajectory",
                "-o",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (tmp_path / "trajectory.png").exists()

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_trajectory_pdf_flag(self, mock_fetch, tmp_path):
        """trajectory --pdf also creates a PDF file."""
        from gigaevo.cli import main

        mock_fetch.return_value = [("A", _make_iteration_df(20))]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "trajectory",
                "-o",
                str(tmp_path),
                "--pdf",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (tmp_path / "trajectory.png").exists()
        assert (tmp_path / "trajectory.pdf").exists()

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_trajectory_outputs_json_summary(self, mock_fetch, tmp_path):
        """trajectory command echoes a JSON summary."""
        from gigaevo.cli import main

        mock_fetch.return_value = [("A", _make_iteration_df(10))]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "trajectory",
                "-o",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        summary = json.loads(result.output.strip())
        assert "output_dir" in summary

    @patch("gigaevo.cli.plot_group._fetch_run_data")
    def test_trajectory_no_best_flag(self, mock_fetch, tmp_path):
        """trajectory --no-best suppresses best fitness line."""
        from gigaevo.cli import main

        mock_fetch.return_value = [("A", _make_iteration_df(20))]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "-r",
                "test/prefix@0:A",
                "plot",
                "trajectory",
                "-o",
                str(tmp_path),
                "--no-best",
            ],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"

    def test_trajectory_requires_output_dir(self):
        """trajectory fails without -o flag."""
        from gigaevo.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "test/prefix@0:A", "plot", "trajectory"],
        )
        assert result.exit_code != 0
