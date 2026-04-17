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
              max_generations: 5
          runs:
          - label: A1
            db: 15
            prefix: test_prefix
            pipeline: standard
            problem_name: toy_kadane
            condition: test condition
            mutation_url: https://example.com/v1
            model_name: test-model
            extra_overrides:
              - problem.name=toy_kadane
              - pipeline=standard
              - redis.db=15
              - model_name=test-model
              - llm_base_url=https://example.com/v1
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


# ---------------------------------------------------------------------------
# Phase 2: task_group Hydra experiment override
# ---------------------------------------------------------------------------


def _build_cmd(run, manifest, cfg_only: bool = False) -> list[str]:
    """Expose the private _build_run_cmd for direct unit tests."""
    from gigaevo.experiment.launch_generator import _build_run_cmd

    return _build_run_cmd(run, manifest, cfg_only=cfg_only)


@pytest.fixture()
def experiment_with_task_group(tmp_path: Path, monkeypatch):
    """Manifest with contract.config.task_group = 'heilbron'."""
    exp_dir = tmp_path / "experiments" / "toy" / "with-task-group"
    exp_dir.mkdir(parents=True)

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/with-task-group
            task: toy
            branch: test-branch
          problem:
            has_test_set: false
            fitness_type: fractional
            metric_name: fitness
          config:
            task_group: heilbron
            extra:
              n_opponents: 3
          runs:
          - label: A1
            db: 15
            prefix: test_prefix
            pipeline: adversarial_asymmetric
            problem_name: heilbron/k5
            condition: test condition
            mutation_url: https://example.com/v1
            model_name: test-model
            extra_overrides:
              - pipeline=adversarial_asymmetric
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
    return "toy/with-task-group"


@pytest.fixture()
def experiment_without_task_group(tmp_path: Path, monkeypatch):
    """Manifest where contract.config.task_group is absent (defaults to None)."""
    exp_dir = tmp_path / "experiments" / "toy" / "no-task-group"
    exp_dir.mkdir(parents=True)

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/no-task-group
            task: toy
            branch: test-branch
          problem:
            has_test_set: false
            fitness_type: fractional
            metric_name: fitness
          config:
            extra:
              n_opponents: 1
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
    return "toy/no-task-group"


class TestTaskGroupEmission:
    """Phase 2: launch_generator emits ``experiment=<task_group>`` as the
    first Hydra override when ``contract.config.task_group`` is set.

    Reasoning: Hydra composes ``experiment=<name>`` natively by swapping
    the ``experiment:`` slot in ``config/config.yaml``'s defaults list.
    It must be the FIRST override so subsequent CLI overrides (pipeline,
    scalar extras) win over anything the task-group file composes.
    """

    def test_task_group_emitted_first(self, experiment_with_task_group):
        from gigaevo.experiment.manifest import load_manifest

        m = load_manifest(experiment_with_task_group)
        params = _build_cmd(m.contract.runs[0], m, cfg_only=False)
        assert params[0] == "experiment=heilbron", (
            f"experiment=heilbron must be the FIRST override, got: {params[0]!r}. "
            f"Full params: {params}"
        )

    def test_task_group_before_pipeline(self, experiment_with_task_group):
        from gigaevo.experiment.manifest import load_manifest

        m = load_manifest(experiment_with_task_group)
        params = _build_cmd(m.contract.runs[0], m, cfg_only=False)
        exp_idx = params.index("experiment=heilbron")
        pipe_idx = next(i for i, p in enumerate(params) if p.startswith("pipeline="))
        assert exp_idx < pipe_idx, (
            "experiment=<task_group> must appear before pipeline= so the "
            f"CLI pipeline override wins. Got exp at {exp_idx}, pipeline at {pipe_idx}"
        )

    def test_task_group_before_extras(self, experiment_with_task_group):
        from gigaevo.experiment.manifest import load_manifest

        m = load_manifest(experiment_with_task_group)
        params = _build_cmd(m.contract.runs[0], m, cfg_only=False)
        exp_idx = params.index("experiment=heilbron")
        extra_idx = next(
            i for i, p in enumerate(params) if p.startswith("n_opponents=")
        )
        assert exp_idx < extra_idx, (
            "experiment=<task_group> must come before config.extra entries so "
            "CLI extras win over the task-group file."
        )

    def test_task_group_absent_no_emission(self, experiment_without_task_group):
        from gigaevo.experiment.manifest import load_manifest

        m = load_manifest(experiment_without_task_group)
        params = _build_cmd(m.contract.runs[0], m, cfg_only=False)
        assert not any(p.startswith("experiment=") for p in params), (
            f"No task_group set, but emitted experiment= override: {params}"
        )

    def test_task_group_appears_in_generated_script(self, experiment_with_task_group):
        result = generate(experiment_with_task_group)
        assert "experiment=heilbron" in result, (
            "Generated launch.sh must contain experiment=heilbron for Hydra "
            "to compose the task-group file."
        )

    def test_task_group_and_pipeline_coexist(self, experiment_with_task_group):
        """Both task_group and pipeline appear — pipeline wins via ordering."""
        from gigaevo.experiment.manifest import load_manifest

        m = load_manifest(experiment_with_task_group)
        params = _build_cmd(m.contract.runs[0], m, cfg_only=False)
        assert "experiment=heilbron" in params
        assert "pipeline=adversarial_asymmetric" in params
