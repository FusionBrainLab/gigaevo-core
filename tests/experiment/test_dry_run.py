"""Tests for gigaevo.experiment.dry_run — Phase 3 of the integrity pipeline.

See .claude/plans/humble-weaving-shamir.md Phase 3.

dry_run(experiment) invokes `python run.py ... --cfg job` per run, parses
the resolved OmegaConf yaml, persists it to experiments/<exp>/cfg_run_<label>.yaml,
and returns a DryRunResult with resolved configs, fingerprints, and CLI args.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest


@pytest.fixture()
def dummy_manifest(tmp_path: Path, monkeypatch):
    """Create a minimal manifest with two runs + redirect PROJ to tmp_path."""
    exp_dir = tmp_path / "experiments" / "toy" / "dry-run-test"
    exp_dir.mkdir(parents=True)

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/dry-run-test
            task: toy
            branch: test-branch
          problem:
            has_test_set: false
            fitness_type: fractional
            metric_name: fitness
          config:
            task_group: heilbron
            shared_overrides:
              n_opponents: 3
              num_parents: 1
          runs:
          - label: A1
            db: 15
            prefix: test_prefix
            pipeline: standard
            problem_name: toy_kadane
            condition: control
            mutation_url: https://example.com/v1
            model_name: test-model
          - label: A2
            db: 14
            prefix: test_prefix_2
            pipeline: standard
            problem_name: toy_kadane
            condition: treatment
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

    # Create a fake config/ tree so fingerprinting has something to hash.
    cfg = tmp_path / "config"
    (cfg / "constants").mkdir(parents=True)
    (cfg / "pipeline").mkdir(parents=True)
    (cfg / "experiment").mkdir(parents=True)
    (cfg / "config.yaml").write_text("# root\n")
    (cfg / "constants" / "pipeline.yaml").write_text("dag_concurrency: 16\n")
    (cfg / "pipeline" / "standard.yaml").write_text("# standard\n")
    (cfg / "experiment" / "heilbron.yaml").write_text("num_parents: 1\n")
    (cfg / "experiment" / "base.yaml").write_text("# base\n")

    monkeypatch.setattr("gigaevo.experiment.manifest.PROJ", tmp_path)
    monkeypatch.setattr("gigaevo.experiment.launch_generator.PROJ_PATH", str(tmp_path))
    return "toy/dry-run-test", tmp_path


@pytest.fixture()
def stub_invoke(monkeypatch):
    """Replace dry_run._invoke_run_py_cfg_job with a canned OmegaConf yaml.

    Returns a mutable list that each call appends to, so tests can assert
    on how many times the seam was called and with what args.
    """
    calls: list[dict] = []

    def _fake(args: list[str], cwd: Path, timeout: float = 120) -> str:
        calls.append({"args": list(args), "cwd": str(cwd), "timeout": timeout})
        # A well-formed Hydra --cfg job output (top-level yaml, no banners).
        # Reflect the CLI args so tests can assert resolution honored them.
        lines = ["problem:", "  name: toy_kadane", "n_opponents: 3", "num_parents: 1"]
        for a in args:
            if a.startswith("pipeline="):
                lines.append(f"pipeline: {a.split('=', 1)[1]}")
        return "\n".join(lines) + "\n"

    import gigaevo.experiment.dry_run as mod  # noqa: PLC0415 — import-on-fixture

    monkeypatch.setattr(mod, "_invoke_run_py_cfg_job", _fake)
    return calls


class TestDryRunResult:
    def test_result_contains_resolved_for_each_run(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        res = dry_run(exp)
        assert set(res.resolved.keys()) == {"A1", "A2"}
        assert res.resolved["A1"]["n_opponents"] == 3
        assert res.resolved["A1"]["num_parents"] == 1

    def test_result_records_cli_args_per_run(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        res = dry_run(exp)
        assert "A1" in res.cli_args
        # Task-group first, CLI overrides follow
        assert res.cli_args["A1"][0] == "experiment=heilbron"

    def test_cfg_job_flag_passed(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        dry_run(exp)
        # Each call must include "--cfg job"
        for call in stub_invoke:
            assert "--cfg job" in call["args"]

    def test_persists_cfg_run_yaml_per_run(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, tmp_path = dummy_manifest
        dry_run(exp)
        out_dir = tmp_path / "experiments" / exp
        assert (out_dir / "cfg_run_A1.yaml").exists()
        assert (out_dir / "cfg_run_A2.yaml").exists()
        # File is parseable yaml with expected keys
        import yaml

        doc = yaml.safe_load((out_dir / "cfg_run_A1.yaml").read_text())
        assert doc["n_opponents"] == 3

    def test_fingerprint_non_empty(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        res = dry_run(exp)
        assert len(res.fingerprint) > 0, "fingerprint should hash at least config.yaml"
        # hashes are 64-char hex (sha256)
        for path, digest in res.fingerprint.items():
            assert len(digest) == 64, f"{path}: {digest!r} is not sha256 hex"
            assert all(c in "0123456789abcdef" for c in digest)

    def test_fingerprint_includes_config_root(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        res = dry_run(exp)
        assert any(
            p.endswith("config.yaml") and "/" not in p.removeprefix("config/")
            for p in res.fingerprint
        ), f"config/config.yaml missing from {list(res.fingerprint)}"

    def test_fingerprint_includes_task_group_file(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        res = dry_run(exp)
        assert any(p.endswith("experiment/heilbron.yaml") for p in res.fingerprint), (
            f"config/experiment/heilbron.yaml missing from {list(res.fingerprint)}"
        )

    def test_fingerprint_stable_across_calls(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, _ = dummy_manifest
        res1 = dry_run(exp)
        res2 = dry_run(exp)
        assert res1.fingerprint == res2.fingerprint

    def test_fingerprint_changes_if_file_edited(self, dummy_manifest, stub_invoke):
        from gigaevo.experiment.dry_run import dry_run

        exp, tmp_path = dummy_manifest
        res1 = dry_run(exp)
        # Edit the task-group file
        (tmp_path / "config" / "experiment" / "heilbron.yaml").write_text(
            "num_parents: 2\n"  # drift
        )
        res2 = dry_run(exp)
        heilbron_key = next(
            p for p in res1.fingerprint if p.endswith("experiment/heilbron.yaml")
        )
        assert res1.fingerprint[heilbron_key] != res2.fingerprint[heilbron_key]


class TestInvokeFailureSurface:
    def test_subprocess_nonzero_raises(self, dummy_manifest, monkeypatch):
        import gigaevo.experiment.dry_run as mod
        from gigaevo.experiment.dry_run import dry_run

        def _bad(args, cwd, timeout=120):
            raise RuntimeError("Hydra compose failed: Key 'pipeline' ...")

        monkeypatch.setattr(mod, "_invoke_run_py_cfg_job", _bad)
        exp, _ = dummy_manifest
        with pytest.raises(RuntimeError, match="Hydra compose failed"):
            dry_run(exp)
