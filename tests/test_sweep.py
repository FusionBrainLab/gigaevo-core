"""Regression tests for :mod:`gigaevo.sweep`.

The sweep runner exists to give every run a fresh Python interpreter
so module-level state cannot leak between iterations of a parameter
sweep. The tests below pin the structural promises:

  * distinct overrides land in distinct ``outputs/{experiment_id}/``
    directories (proves subprocess isolation actually runs separate
    interpreters),
  * a failure in one run does not block the rest,
  * the aggregate exit code reflects whether every subprocess
    succeeded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-sweep")


def _write_experiment(target: Path, output_dir: Path) -> Path:
    """Materialise a minimal experiment file whose ``output_dir``
    points at the test's temporary directory."""
    body = dedent(
        f"""
        from pathlib import Path
        from gigaevo.config.schemas import (
            BehaviorSpaceConfig,
            ChatOpenAIConfig,
            DataPlaneSettings,
            DefaultPipelineBuilderConfig,
            EnsembleRouterConfig,
            ExperimentConfig,
            FitnessProportionalEliteSelectorConfig,
            IslandConfig,
            PipelineConfig,
            ProblemConfig,
            RedisConfig,
            SingleIslandConfig,
            SteadyStateEngineConfig,
            SumArchiveSelectorConfig,
            TopFitnessMigrantSelectorConfig,
        )


        def build() -> ExperimentConfig:
            redis = RedisConfig()
            return ExperimentConfig(
                name="sweep_test",
                seed=1,
                output_dir=Path({str(output_dir)!r}),
                redis=redis,
                dataplane=DataPlaneSettings(redis=redis, key_prefix="gigaevo:sweep_test"),
                problem=ProblemConfig(name="sweep_test", problem_dir=Path("/srv/x")),
                algorithm=SingleIslandConfig(
                    island=IslandConfig(
                        island_id="main",
                        behavior_space=BehaviorSpaceConfig(
                            keys=["fitness"],
                            bounds=[(0.0, 1.0)],
                            resolutions=[100],
                            binning_types=["linear"],
                        ),
                        archive_selector=SumArchiveSelectorConfig(
                            fitness_keys=["fitness"], fitness_key_higher_is_better=[True]
                        ),
                        elite_selector=FitnessProportionalEliteSelectorConfig(
                            fitness_key="fitness"
                        ),
                        migrant_selector=TopFitnessMigrantSelectorConfig(
                            fitness_key="fitness"
                        ),
                    )
                ),
                engine=SteadyStateEngineConfig(),
                pipeline=PipelineConfig(builder=DefaultPipelineBuilderConfig()),
                llm=EnsembleRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")]),
            )
        """
    )
    target.write_text(body)
    return target


def _write_sweep(target: Path, runs: list[list[str]]) -> Path:
    body = "def define_sweep() -> list[list[str]]:\n"
    body += f"    return {runs!r}\n"
    target.write_text(body)
    return target


def _invoke_sweep(experiment: Path, sweep: Path, *, parallel: int = 1) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "gigaevo.sweep",
        str(experiment),
        str(sweep),
        "--parallel",
        str(parallel),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)


class TestSweepIsolation:
    def test_distinct_overrides_produce_distinct_output_dirs(
        self, tmp_path: Path
    ) -> None:
        """Three runs with three different seeds materialise three
        directories under the experiment's ``output_dir``."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = _write_sweep(
            tmp_path / "sweep.py",
            [["--dry-run", "--seed", str(seed)] for seed in (1, 2, 3)],
        )

        result = _invoke_sweep(experiment, sweep)

        assert result.returncode == 0, result.stderr
        run_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
        assert len(run_dirs) == 3, (
            f"expected 3 distinct output dirs, got {[p.name for p in run_dirs]}"
        )

        seeds = sorted(
            json.loads((d / "config.json").read_text())["seed"] for d in run_dirs
        )
        assert seeds == [1, 2, 3]

    def test_parallel_run_matches_sequential(self, tmp_path: Path) -> None:
        """``--parallel 2`` produces the same set of output dirs as
        the sequential run (idempotent against the resolved
        ``experiment_id``)."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = _write_sweep(
            tmp_path / "sweep.py",
            [["--dry-run", "--seed", str(seed)] for seed in (7, 8, 9, 10)],
        )

        result = _invoke_sweep(experiment, sweep, parallel=2)

        assert result.returncode == 0, result.stderr
        run_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
        assert len(run_dirs) == 4
        seeds = sorted(
            json.loads((d / "config.json").read_text())["seed"] for d in run_dirs
        )
        assert seeds == [7, 8, 9, 10]


class TestSweepFailurePropagation:
    def test_one_bogus_override_does_not_block_others(self, tmp_path: Path) -> None:
        """Three runs total: the middle run carries an invalid
        override and fails; the other two still produce output dirs.
        The aggregate exit code is non-zero."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = _write_sweep(
            tmp_path / "sweep.py",
            [
                ["--dry-run", "--seed", "11"],
                ["--dry-run", "--seed", "not-an-int"],
                ["--dry-run", "--seed", "13"],
            ],
        )

        result = _invoke_sweep(experiment, sweep)

        assert result.returncode == 1, result.stderr
        run_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
        seeds = sorted(
            json.loads((d / "config.json").read_text())["seed"] for d in run_dirs
        )
        assert seeds == [11, 13]


class TestSweepShape:
    def test_missing_define_sweep_raises(self, tmp_path: Path) -> None:
        """A sweep module that forgets ``define_sweep`` fails fast
        with a non-zero exit and a stderr message naming the
        attribute."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = tmp_path / "sweep.py"
        sweep.write_text("# no define_sweep\n")

        result = _invoke_sweep(experiment, sweep)

        assert result.returncode != 0
        assert "define_sweep" in result.stderr

    def test_wrong_return_type_raises(self, tmp_path: Path) -> None:
        """``define_sweep`` returning the wrong shape fails fast
        before any subprocess is spawned."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = tmp_path / "sweep.py"
        sweep.write_text(
            "def define_sweep():\n    return ['flat', 'list']\n"
        )

        result = _invoke_sweep(experiment, sweep)

        assert result.returncode != 0
        assert "list[list[str]]" in result.stderr

    def test_empty_sweep_succeeds(self, tmp_path: Path) -> None:
        """An empty sweep is a no-op success: nothing to fail, nothing
        to do, exit code zero."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = _write_sweep(tmp_path / "sweep.py", [])

        result = _invoke_sweep(experiment, sweep)

        assert result.returncode == 0, result.stderr
        assert "0/0" in result.stdout
        assert not list(out_dir.iterdir())

    def test_missing_experiment_file_fails_fast(self, tmp_path: Path) -> None:
        """A non-existent experiment path is rejected before any
        subprocess is spawned. No output dirs are created."""
        sweep = _write_sweep(
            tmp_path / "sweep.py", [["--dry-run", "--seed", "1"]]
        )
        missing = tmp_path / "does_not_exist.py"

        result = _invoke_sweep(missing, sweep)

        assert result.returncode == 2
        assert "experiment file not found" in result.stderr

    def test_missing_sweep_file_fails_fast(self, tmp_path: Path) -> None:
        """A non-existent sweep path is rejected with a typed message."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        missing = tmp_path / "no_sweep_here.py"

        result = _invoke_sweep(experiment, missing)

        assert result.returncode == 2
        assert "sweep file not found" in result.stderr

    def test_non_py_sweep_file_fails_fast(self, tmp_path: Path) -> None:
        """The sweep loader rejects non-``.py`` paths before executing
        arbitrary user code."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        wrong_ext = tmp_path / "sweep.txt"
        wrong_ext.write_text(
            "def define_sweep():\n    return [['--seed', '1']]\n"
        )

        result = _invoke_sweep(experiment, wrong_ext)

        assert result.returncode == 2
        assert ".py extension" in result.stderr


class TestSweepFailureSurface:
    def test_all_failing_runs_report_non_zero(self, tmp_path: Path) -> None:
        """Every run is invalid; the aggregate exit code is 1 and the
        stderr summary spells out the failure count."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = _write_sweep(
            tmp_path / "sweep.py",
            [
                ["--dry-run", "--seed", "not-an-int"],
                ["--dry-run", "--seed", "also-not-int"],
            ],
        )

        result = _invoke_sweep(experiment, sweep)

        assert result.returncode == 1, result.stdout
        assert "2/2 runs failed" in result.stderr

    def test_parallel_exceeding_cpu_count_still_runs(
        self, tmp_path: Path
    ) -> None:
        """``--parallel`` larger than the available CPU count is not an
        error; ``ProcessPoolExecutor`` caps the active workers itself
        and the sweep still completes."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()
        experiment = _write_experiment(tmp_path / "exp.py", out_dir)
        sweep = _write_sweep(
            tmp_path / "sweep.py",
            [["--dry-run", "--seed", str(seed)] for seed in range(2)],
        )

        oversized = max(64, (os.cpu_count() or 1) * 8)
        result = _invoke_sweep(experiment, sweep, parallel=oversized)

        assert result.returncode == 0, result.stderr
        run_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
        assert len(run_dirs) == 2
