"""Integration tests for the typed CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


_EXPERIMENT_BODY = dedent(
    """
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
            name="cli_test",
            seed=99,
            output_dir=Path("{OUTPUT_DIR}"),
            redis=redis,
            dataplane=DataPlaneSettings(redis=redis, key_prefix="gigaevo:cli_test"),
            problem=ProblemConfig(name="cli_test", problem_dir=Path("/srv/x")),
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


def _make_experiment(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    body = _EXPERIMENT_BODY.replace("{OUTPUT_DIR}", str(out))
    exp = tmp_path / "experiment.py"
    exp.write_text(body)
    return exp


class TestCliDryRun:
    def test_dry_run_dumps_config_and_exits_zero(self, tmp_path: Path) -> None:
        from gigaevo.cli import main

        exp = _make_experiment(tmp_path)
        exit_code = main([str(exp), "--dry-run"])
        assert exit_code == 0

        out_root = tmp_path / "outputs"
        run_dirs = list(out_root.iterdir())
        assert len(run_dirs) == 1
        config_path = run_dirs[0] / "config.json"
        assert config_path.exists()

        dumped = json.loads(config_path.read_text())
        assert dumped["name"] == "cli_test"
        assert dumped["seed"] == 99

    def test_dry_run_directory_name_is_experiment_id(self, tmp_path: Path) -> None:
        from gigaevo.cli import main
        from gigaevo.config.experiment_loader import build_experiment

        exp = _make_experiment(tmp_path)
        cfg = build_experiment(exp)
        expected_id = cfg.experiment_id

        main([str(exp), "--dry-run"])

        run_dir = (tmp_path / "outputs" / expected_id)
        assert run_dir.exists(), (
            f"expected {run_dir}, got {list((tmp_path / 'outputs').iterdir())}"
        )


class TestCliErrors:
    def test_missing_experiment_file_propagates(self, tmp_path: Path) -> None:
        from gigaevo.cli import main

        with pytest.raises(FileNotFoundError):
            main([str(tmp_path / "no_such.py"), "--dry-run"])

    def test_invalid_experiment_raises(self, tmp_path: Path) -> None:
        from gigaevo.cli import main
        from gigaevo.config.experiment_loader import ExperimentModuleError

        bad = tmp_path / "bad.py"
        bad.write_text("x = 1\n")  # no build()

        with pytest.raises(ExperimentModuleError):
            main([str(bad), "--dry-run"])
