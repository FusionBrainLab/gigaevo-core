"""Validation tests for the reference experiment.

This file is the template every Phase 2 experiment migration follows:
load the experiment module via the loader, assert build() returns a
fully-validated ExperimentConfig, and pin the observable shape of the
key fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = REPO_ROOT / "experiments" / "steady_state_hotpotqa.py"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestReferenceExperimentBuilds:
    def test_file_exists_at_canonical_path(self) -> None:
        assert EXPERIMENT_PATH.exists(), (
            f"reference experiment missing at {EXPERIMENT_PATH}"
        )

    def test_loader_validates_signature(self) -> None:
        from gigaevo.config.experiment_loader import load_experiment_module

        module = load_experiment_module(EXPERIMENT_PATH)
        assert callable(module.build)

    def test_build_returns_experiment_config(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.schemas import ExperimentConfig

        cfg = build_experiment(EXPERIMENT_PATH)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "steady_state_hotpotqa"

    def test_key_prefix_follows_convention(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment

        cfg = build_experiment(EXPERIMENT_PATH)
        assert cfg.dataplane.key_prefix == "gigaevo:steady_state_hotpotqa"

    def test_problem_dir_points_at_hotpotqa_static_f1(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment

        cfg = build_experiment(EXPERIMENT_PATH)
        assert cfg.problem.problem_dir == Path("problems/chains/hotpotqa/static_f1")


class TestReferenceExperimentShape:
    def test_uses_single_island_with_max_size_and_remover(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.schemas import SingleIslandConfig

        cfg = build_experiment(EXPERIMENT_PATH)
        assert isinstance(cfg.algorithm, SingleIslandConfig)
        island = cfg.algorithm.island
        assert island.max_size == 300
        assert island.archive_remover is not None

    def test_uses_steady_state_engine(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.schemas import SteadyStateEngineConfig

        cfg = build_experiment(EXPERIMENT_PATH)
        assert isinstance(cfg.engine, SteadyStateEngineConfig)
        assert cfg.engine.max_generations == 200
        assert cfg.engine.max_in_flight <= cfg.engine.max_mutations_per_generation

    def test_uses_context_pipeline_for_hotpotqa(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.schemas import ContextPipelineBuilderConfig

        cfg = build_experiment(EXPERIMENT_PATH)
        assert isinstance(cfg.pipeline.builder, ContextPipelineBuilderConfig)

    def test_uses_bandit_router_with_two_endpoints(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.schemas import BanditRouterConfig

        cfg = build_experiment(EXPERIMENT_PATH)
        assert isinstance(cfg.llm, BanditRouterConfig)
        assert len(cfg.llm.models) == 2
        assert any(
            m.base_url and "openrouter" in m.base_url for m in cfg.llm.models
        ), "expected one endpoint to route through OpenRouter"

    def test_fitness_is_only_behavior_key(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.schemas import SingleIslandConfig

        cfg = build_experiment(EXPERIMENT_PATH)
        assert isinstance(cfg.algorithm, SingleIslandConfig)
        assert cfg.algorithm.island.behavior_space.behavior_keys == ["fitness"]


class TestReferenceExperimentObjectGraph:
    """Integration test: the reference experiment must successfully
    construct the full runtime object graph via build_object_graph.
    Verifies the schema layer composes coherently against a real
    problem directory with a real metrics.yaml."""

    def test_build_object_graph_succeeds_on_reference_experiment(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.object_graph import build_object_graph

        cfg = build_experiment(EXPERIMENT_PATH)
        graph = build_object_graph(cfg)

        assert graph["primary_metric"] == "fitness"
        assert graph["higher_is_better"] is True
        assert graph["required_behavior_keys"] == ["fitness"]

    def test_constructed_bandit_router_carries_resolved_keys(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment
        from gigaevo.config.object_graph import build_object_graph
        from gigaevo.llm.bandit import BanditModelRouter

        cfg = build_experiment(EXPERIMENT_PATH)
        graph = build_object_graph(cfg)
        router = graph["llm"]
        assert isinstance(router, BanditModelRouter)
        assert router.fitness_key == "fitness"
        assert router.higher_is_better is True


class TestReferenceExperimentDeterminism:
    """The experiment_id is the canonical reproducibility token. Two
    loads of the same file must produce identical IDs."""

    def test_experiment_id_stable_across_loads(self) -> None:
        from gigaevo.config.experiment_loader import build_experiment

        first = build_experiment(EXPERIMENT_PATH).experiment_id
        second = build_experiment(EXPERIMENT_PATH).experiment_id
        assert first == second
        assert len(first) == 12

    def test_model_copy_with_seed_changes_experiment_id(self) -> None:
        """Sweep variants compose via model_copy(update={...}). The
        contract is that any field change yields a new experiment_id
        so output directories diverge."""
        from gigaevo.config.experiment_loader import build_experiment

        baseline = build_experiment(EXPERIMENT_PATH)
        variant = baseline.model_copy(update={"seed": 7})
        assert variant.experiment_id != baseline.experiment_id
        assert variant.seed == 7
