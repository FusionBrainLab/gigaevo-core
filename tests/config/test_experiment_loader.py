"""Unit tests for the experiment-file loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from gigaevo.config.experiment_loader import (
    ExperimentModuleError,
    build_experiment,
    load_experiment_module,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


_VALID_EXPERIMENT_BODY = dedent(
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
            name="loader_test",
            redis=redis,
            dataplane=DataPlaneSettings(redis=redis, key_prefix="gigaevo:loader_test"),
            problem=ProblemConfig(name="loader_test", problem_dir=Path("/srv/x")),
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


class TestLoadExperimentModule:
    def test_loads_module_with_build(self, tmp_path: Path) -> None:
        exp = tmp_path / "good.py"
        exp.write_text(_VALID_EXPERIMENT_BODY)
        module = load_experiment_module(exp)
        assert hasattr(module, "build")
        assert callable(module.build)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_experiment_module(tmp_path / "missing.py")

    def test_directory_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ExperimentModuleError, match="not a file"):
            load_experiment_module(tmp_path)

    def test_missing_build_raises(self, tmp_path: Path) -> None:
        exp = tmp_path / "no_build.py"
        exp.write_text("x = 1\n")
        with pytest.raises(ExperimentModuleError, match="build"):
            load_experiment_module(exp)

    def test_non_callable_build_raises(self, tmp_path: Path) -> None:
        exp = tmp_path / "build_not_callable.py"
        exp.write_text("build = 42\n")
        with pytest.raises(ExperimentModuleError, match="build"):
            load_experiment_module(exp)

    def test_build_with_required_args_rejected(self, tmp_path: Path) -> None:
        exp = tmp_path / "bad_signature.py"
        exp.write_text("def build(x): return x\n")
        with pytest.raises(ExperimentModuleError, match="required"):
            load_experiment_module(exp)

    def test_build_with_default_args_accepted(self, tmp_path: Path) -> None:
        exp = tmp_path / "default_args.py"
        exp.write_text("def build(x=1, y=2): return None\n")
        # signature validation passes — build_experiment would reject
        # the None return, but load_experiment_module is signature-only
        load_experiment_module(exp)

    def test_two_experiments_load_under_distinct_names(
        self, tmp_path: Path
    ) -> None:
        """sys.modules collision check: loading two experiments with
        the same stem from different directories must yield distinct
        modules so a sweep can iterate them in one process."""
        exp_a = tmp_path / "a" / "exp.py"
        exp_b = tmp_path / "b" / "exp.py"
        for p in (exp_a, exp_b):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("def build(): return None\n")
        mod_a = load_experiment_module(exp_a)
        mod_b = load_experiment_module(exp_b)
        # Both modules carry build callables — the in-memory identities
        # need to be distinct objects so reloading does not return a
        # stale cached module.
        assert mod_a is not mod_b


class TestBuildExperiment:
    def test_returns_experiment_config(self, tmp_path: Path) -> None:
        from gigaevo.config.schemas.experiment import ExperimentConfig

        exp = tmp_path / "ok.py"
        exp.write_text(_VALID_EXPERIMENT_BODY)
        cfg = build_experiment(exp)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "loader_test"

    def test_rejects_non_config_return(self, tmp_path: Path) -> None:
        exp = tmp_path / "wrong_return.py"
        exp.write_text("def build(): return {'name': 'x'}\n")
        with pytest.raises(ExperimentModuleError, match="ExperimentConfig"):
            build_experiment(exp)
