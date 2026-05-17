"""Unit tests for the typed-config -> runtime-object adapter."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from gigaevo.config.object_graph import (
    _required_behavior_keys,
    _resolve_bandit_keys,
    build_object_graph,
)
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
    MultiIslandConfig,
    FitnessArchiveRemoverConfig,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def problem_dir(tmp_path: Path) -> Path:
    """Build a minimal on-disk problem directory with the metrics.yaml
    and task_description.md the runtime ProblemContext loads."""
    pdir = tmp_path / "stub_problem"
    pdir.mkdir()
    (pdir / "task_description.txt").write_text("Stub task for unit tests.\n")
    (pdir / "metrics.yaml").write_text(
        dedent(
            """
            specs:
              fitness:
                description: primary fitness score
                higher_is_better: true
                is_primary: true
                lower_bound: 0.0
                upper_bound: 1.0
              is_valid:
                description: validity indicator
                higher_is_better: true
                lower_bound: 0.0
                upper_bound: 1.0
            """
        )
    )
    return pdir


def _island(island_id: str = "main", max_size: int | None = None) -> IslandConfig:
    return IslandConfig(
        island_id=island_id,
        max_size=max_size,
        behavior_space=BehaviorSpaceConfig(
            keys=["fitness"],
            bounds=[(0.0, 1.0)],
            resolutions=[100],
            binning_types=["linear"],
        ),
        archive_selector=SumArchiveSelectorConfig(
            fitness_keys=["fitness"], fitness_key_higher_is_better=[True]
        ),
        elite_selector=FitnessProportionalEliteSelectorConfig(fitness_key="fitness"),
        archive_remover=(
            FitnessArchiveRemoverConfig(fitness_key="fitness")
            if max_size is not None
            else None
        ),
        migrant_selector=TopFitnessMigrantSelectorConfig(fitness_key="fitness"),
    )


def _config(problem_dir: Path, **overrides) -> ExperimentConfig:  # type: ignore[no-untyped-def]
    redis = RedisConfig()
    defaults = {
        "name": "graph_test",
        "redis": redis,
        "dataplane": DataPlaneSettings(redis=redis, key_prefix="gigaevo:graph_test"),
        "problem": ProblemConfig(name="graph_test", problem_dir=problem_dir),
        "algorithm": SingleIslandConfig(island=_island()),
        "engine": SteadyStateEngineConfig(),
        "pipeline": PipelineConfig(builder=DefaultPipelineBuilderConfig()),
        "llm": EnsembleRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")]),
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestRequiredBehaviorKeys:
    def test_single_island_returns_island_keys(self, problem_dir: Path) -> None:
        cfg = _config(problem_dir)
        assert _required_behavior_keys(cfg) == ["fitness"]

    def test_multi_island_unions_keys(self, problem_dir: Path) -> None:
        island_a = _island("a")
        island_b = IslandConfig(
            island_id="b",
            behavior_space=BehaviorSpaceConfig(
                keys=["complexity"],
                bounds=[(0.0, 100.0)],
                resolutions=[10],
                binning_types=["linear"],
            ),
            archive_selector=SumArchiveSelectorConfig(
                fitness_keys=["complexity"], fitness_key_higher_is_better=[False]
            ),
            elite_selector=FitnessProportionalEliteSelectorConfig(
                fitness_key="complexity", fitness_key_higher_is_better=False
            ),
            migrant_selector=TopFitnessMigrantSelectorConfig(
                fitness_key="complexity", fitness_key_higher_is_better=False
            ),
        )
        cfg = _config(problem_dir, algorithm=MultiIslandConfig(islands=[island_a, island_b]))
        keys = _required_behavior_keys(cfg)
        assert set(keys) == {"fitness", "complexity"}
        assert keys == sorted(keys)


class TestResolveBanditKeys:
    def test_metrics_yaml_drives_resolution(self, problem_dir: Path) -> None:
        cfg = _config(problem_dir)
        key, higher = _resolve_bandit_keys(cfg)
        assert key == "fitness"
        assert higher is True

    def test_problem_override_takes_precedence(self, problem_dir: Path) -> None:
        cfg = _config(
            problem_dir,
            problem=ProblemConfig(
                name="graph_test",
                problem_dir=problem_dir,
                primary_metric="is_valid",
                higher_is_better=False,
            ),
        )
        key, higher = _resolve_bandit_keys(cfg)
        assert key == "is_valid"
        assert higher is False


class TestBuildObjectGraph:
    def test_returns_expected_keys(self, problem_dir: Path) -> None:
        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        expected = {
            "redis_storage",
            "problem_context",
            "llm",
            "strategy",
            "runtime_engine_config",
            "evolution_context",
            "pipeline_builder",
            "dag_blueprint",
            "runtime_runner_config",
            "primary_metric",
            "higher_is_better",
            "required_behavior_keys",
        }
        assert set(graph.keys()) == expected

    def test_runtime_runner_config_built_from_schema(
        self, problem_dir: Path
    ) -> None:
        from gigaevo.runner.dag_runner import (
            DagRunnerConfig as RuntimeDagRunnerConfig,
        )

        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        runner_cfg = graph["runtime_runner_config"]
        assert isinstance(runner_cfg, RuntimeDagRunnerConfig)
        # Default schema values must propagate.
        assert runner_cfg.poll_interval == 0.5
        assert runner_cfg.max_concurrent_dags == 8

    def test_redis_storage_uses_key_prefix_from_dataplane(
        self, problem_dir: Path
    ) -> None:
        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        storage = graph["redis_storage"]
        assert storage.config.key_prefix == "gigaevo:graph_test"
        assert storage.config.redis_url == cfg.redis.url

    def test_problem_context_loads_metrics(self, problem_dir: Path) -> None:
        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        problem_ctx = graph["problem_context"]
        assert problem_ctx.metrics_context.get_primary_key() == "fitness"

    def test_strategy_is_multi_island_runtime(self, problem_dir: Path) -> None:
        from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland

        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        assert isinstance(graph["strategy"], MapElitesMultiIsland)

    def test_runtime_engine_config_carries_behavior_keys(
        self, problem_dir: Path
    ) -> None:
        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        engine_cfg = graph["runtime_engine_config"]
        # The acceptor's required keys were threaded through
        from gigaevo.evolution.engine.acceptor import RequiredBehaviorKeysAcceptor

        sub = [
            a
            for a in engine_cfg.program_acceptor.acceptors
            if isinstance(a, RequiredBehaviorKeysAcceptor)
        ]
        assert len(sub) == 1
        assert sub[0].required_behavior_keys == {"fitness"}

    def test_pipeline_builder_constructs(self, problem_dir: Path) -> None:
        from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder

        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        assert isinstance(graph["pipeline_builder"], DefaultPipelineBuilder)

    def test_resolved_primary_metric_metadata(self, problem_dir: Path) -> None:
        cfg = _config(problem_dir)
        graph = build_object_graph(cfg)
        assert graph["primary_metric"] == "fitness"
        assert graph["higher_is_better"] is True
        assert graph["required_behavior_keys"] == ["fitness"]


class TestBuildObjectGraphBandit:
    """The bandit-router build path resolves fitness_key and
    higher_is_better from the problem's MetricsContext at graph-build
    time. This path is structurally distinct from the ensemble path —
    a separate test pins the resolution contract."""

    def test_bandit_router_receives_metrics_resolved_fitness_key(
        self, problem_dir: Path
    ) -> None:
        from gigaevo.config.schemas import BanditRouterConfig
        from gigaevo.llm.bandit import BanditModelRouter

        cfg = _config(
            problem_dir,
            llm=BanditRouterConfig(
                models=[
                    ChatOpenAIConfig(model="gpt-4o-mini"),
                    ChatOpenAIConfig(model="deepseek-chat"),
                ],
                exploration_constant=2.0,
                window_size=50,
            ),
        )
        graph = build_object_graph(cfg)
        router = graph["llm"]
        assert isinstance(router, BanditModelRouter)
        assert router.fitness_key == "fitness"
        assert router.higher_is_better is True
        assert len(router.models) == 2

    def test_bandit_honors_problem_override_for_higher_is_better(
        self, problem_dir: Path
    ) -> None:
        from gigaevo.config.schemas import BanditRouterConfig

        cfg = _config(
            problem_dir,
            problem=ProblemConfig(
                name="graph_test",
                problem_dir=problem_dir,
                primary_metric="is_valid",
                higher_is_better=False,
            ),
            llm=BanditRouterConfig(models=[ChatOpenAIConfig(model="x")]),
        )
        graph = build_object_graph(cfg)
        router = graph["llm"]
        assert router.fitness_key == "is_valid"
        assert router.higher_is_better is False
