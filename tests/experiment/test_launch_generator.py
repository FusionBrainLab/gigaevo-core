"""Integration tests for launch_generator.generate().

Verifies that generate() produces valid bash from a real v2 manifest,
catching type mismatches between the Pydantic schema and the generator's
assumptions (e.g. config.extra.get() vs config.get()).
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from gigaevo.experiment.launch_generator import generate


@pytest.fixture()
def dummy_experiment(tmp_path: Path, monkeypatch):
    """Create a minimal v2 experiment.yaml and point PROJ at tmp_path."""
    exp_dir = tmp_path / "experiments" / "toy" / "gen-test"
    exp_dir.mkdir(parents=True)

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/gen-test
            task: toy
            branch: test-branch
          problem:
            has_test_set: false
            fitness_type: fractional
            metric_name: fitness
          config:
            extra:
              stage_timeout: 99
              dag_timeout: 199
              max_mutations_per_generation: 3
              max_elites_per_generation: 4
              num_parents: 2
              mutation_mode: rewrite
          runs:
          - label: A1
            db: 15
            prefix: test_prefix
            pipeline: standard
            problem_name: toy_kadane
            condition: test condition
            mutation_url: https://example.com/v1
            model_name: test-model
          servers:
          - example.com
          custom_env:
            FOO: bar
          max_generations: 5
          baseline:
            reference: null
          tools: []
        lifecycle:
          status: implemented
          launch:
            time: null
            commit: null
          smoke_test:
            completed: true
          treatment_verification:
            completed: true
        telemetry:
          checkpoints: []
          treatment_checks:
            completed: false
            results: []
        control_plane:
          watchdog: {}
          notifications:
            pr:
              enabled: false
            telegram:
              enabled: false
    """)
    (exp_dir / "experiment.yaml").write_text(yaml_content)

    monkeypatch.setattr("gigaevo.experiment.launch_generator.PROJ_PATH", str(tmp_path))
    monkeypatch.setattr("gigaevo.experiment.manifest.PROJ", tmp_path)
    return "toy/gen-test"


class TestGenerateProducesValidBash:
    def test_generates_without_error(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert result.startswith("#!/usr/bin/env bash")

    def test_contains_config_extra_values(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert "stage_timeout=99" in result
        assert "dag_timeout=199" in result
        assert "max_mutations_per_generation=3" in result
        assert "max_elites_per_generation=4" in result
        assert "num_parents=2" in result

    def test_contains_mutation_mode(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert "mutation_mode=rewrite" in result

    def test_contains_run_params(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert "problem.name=toy_kadane" in result
        assert "pipeline=standard" in result
        assert "redis.db=15" in result
        assert "model_name=test-model" in result
        assert "max_generations=5" in result

    def test_contains_custom_env(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert 'export FOO="bar"' in result

    def test_contains_experiment_header(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert "Experiment: toy/gen-test" in result
        assert "Branch: test-branch" in result

    def test_contains_no_proxy(self, dummy_experiment):
        result = generate(dummy_experiment)
        assert "example.com" in result
        assert "NO_PROXY" in result


@pytest.fixture()
def experiment_with_novel_extras(tmp_path: Path, monkeypatch):
    """Manifest with config.extra keys NOT in the known-handled set.

    Reproduces the silent-treatment bug from heilbron/k5-budget-v2 launch:
    n_opponents/source_prompt_k were declared under config.extra but
    silently dropped by _build_run_cmd, leading to K=1 instead of K=3.
    """
    exp_dir = tmp_path / "experiments" / "toy" / "novel-extras"
    exp_dir.mkdir(parents=True)

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/novel-extras
            task: toy
            branch: test-branch
          problem:
            has_test_set: false
            fitness_type: fractional
            metric_name: fitness
          config:
            extra:
              stage_timeout: 99
              n_opponents: 3
              source_prompt_k: 3
              pipeline_builder.archive_reeval: true
              opponent_provider.cache_ttl: 2.0
          runs:
          - label: A1
            db: 15
            prefix: test_prefix
            pipeline: standard
            problem_name: toy_kadane
            condition: test condition
            mutation_url: https://example.com/v1
            model_name: test-model
          servers:
          - example.com
          custom_env: {}
          max_generations: 5
          baseline:
            reference: null
          tools: []
        lifecycle:
          status: implemented
          launch:
            time: null
            commit: null
          smoke_test:
            completed: true
          treatment_verification:
            completed: true
        telemetry:
          checkpoints: []
          treatment_checks:
            completed: false
            results: []
        control_plane:
          watchdog: {}
          notifications:
            pr:
              enabled: false
            telegram:
              enabled: false
    """)
    (exp_dir / "experiment.yaml").write_text(yaml_content)

    monkeypatch.setattr("gigaevo.experiment.launch_generator.PROJ_PATH", str(tmp_path))
    monkeypatch.setattr("gigaevo.experiment.manifest.PROJ", tmp_path)
    return "toy/novel-extras"


class TestNovelExtrasPassThrough:
    """Regression: every key under config.extra must reach Hydra as an override
    (except documented utility keys that control shell generation itself).
    """

    def test_n_opponents_passed_as_hydra_override(self, experiment_with_novel_extras):
        result = generate(experiment_with_novel_extras)
        assert "n_opponents=3" in result

    def test_source_prompt_k_passed_as_hydra_override(
        self, experiment_with_novel_extras
    ):
        result = generate(experiment_with_novel_extras)
        assert "source_prompt_k=3" in result

    def test_dotted_override_passed(self, experiment_with_novel_extras):
        result = generate(experiment_with_novel_extras)
        assert "pipeline_builder.archive_reeval=true" in result.lower()

    def test_float_override_passed(self, experiment_with_novel_extras):
        result = generate(experiment_with_novel_extras)
        assert "opponent_provider.cache_ttl=2.0" in result

    def test_each_extra_key_emitted_exactly_twice(self, experiment_with_novel_extras):
        """Every extra key appears once per run command block (cfg-verify +
        nohup launch) = 2 times total for a single-run manifest. No
        duplication from a second emission path."""
        result = generate(experiment_with_novel_extras)
        assert result.count("stage_timeout=99") == 2
        assert result.count("n_opponents=3") == 2
        assert result.count("source_prompt_k=3") == 2
