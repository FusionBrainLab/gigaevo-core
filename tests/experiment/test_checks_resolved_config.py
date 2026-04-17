"""Tests for Phase 4 — preflight pin check + fingerprint check.

Verifies ``_check_resolved_config_matches_pinned`` diffs the resolved
Hydra config (produced by ``dry_run``) against ``contract.config.pinned``
(plus per-run ``pinned`` overlays), and ``_check_config_fingerprint_stable``
rejects a re-launch when any fingerprinted config file has drifted since
the original launch recorded the digest.

Both checks stub out ``dry_run`` so tests don't spin up Hydra.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from gigaevo.experiment.dry_run import DryRunResult


@pytest.fixture()
def manifest_with_pinned(tmp_path: Path, monkeypatch):
    """Manifest with contract-level and per-run pins; stubs dry_run output."""
    exp_dir = tmp_path / "experiments" / "toy" / "pin-check"
    exp_dir.mkdir(parents=True)

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/pin-check
            task: toy
            branch: test-branch
          problem:
            has_test_set: false
            fitness_type: fractional
            metric_name: fitness
          config:
            extra:
              stage_timeout: 600
            pinned:
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
            pinned:
              n_opponents: 5
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

    monkeypatch.setattr("gigaevo.experiment.manifest.PROJ", tmp_path)
    return "toy/pin-check", tmp_path


def _make_stub(resolved_per_run: dict[str, dict], fingerprint: dict | None = None):
    """Build a fake dry_run() that returns canned resolved configs."""
    fp = fingerprint if fingerprint is not None else {"config/config.yaml": "abc"}

    def _fake(experiment, **kwargs):
        return DryRunResult(
            resolved=resolved_per_run,
            fingerprint=fp,
            cli_args={label: [] for label in resolved_per_run},
        )

    return _fake


class TestResolvedConfigMatchesPinned:
    def test_pin_matches_pass(self, manifest_with_pinned, monkeypatch):
        """n_opponents=3 (contract pin), num_parents=1 (contract pin), A2 override n_opponents=5 → PASS."""
        from gigaevo.experiment import checks as checks_mod

        exp, _ = manifest_with_pinned
        stub = _make_stub(
            {
                "A1": {"n_opponents": 3, "num_parents": 1},
                "A2": {"n_opponents": 5, "num_parents": 1},
            }
        )
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_resolved_config_matches_pinned(results, m, exp)
        assert len(results) == 1
        assert results[0].passed, results[0].message

    def test_pin_mismatch_fails_critical(self, manifest_with_pinned, monkeypatch):
        """n_opponents pinned=3 but resolved=1 → CRITICAL."""
        from gigaevo.experiment import checks as checks_mod

        exp, _ = manifest_with_pinned
        stub = _make_stub(
            {
                "A1": {"n_opponents": 1, "num_parents": 1},  # drift
                "A2": {"n_opponents": 5, "num_parents": 1},
            }
        )
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_resolved_config_matches_pinned(results, m, exp)
        assert len(results) == 1
        r = results[0]
        assert not r.passed
        assert r.is_blocking
        assert "n_opponents" in r.message
        assert "A1" in r.message  # must identify the failing run

    def test_per_run_pin_overrides_contract(self, manifest_with_pinned, monkeypatch):
        """A2 has run-level pinned {n_opponents: 5}; resolved=3 satisfies contract pin but violates per-run pin."""
        from gigaevo.experiment import checks as checks_mod

        exp, _ = manifest_with_pinned
        stub = _make_stub(
            {
                "A1": {"n_opponents": 3, "num_parents": 1},
                "A2": {
                    "n_opponents": 3,
                    "num_parents": 1,
                },  # violates A2.pinned.n_opponents=5
            }
        )
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_resolved_config_matches_pinned(results, m, exp)
        r = results[0]
        assert not r.passed
        assert "A2" in r.message
        assert "n_opponents" in r.message

    def test_pin_absent_from_resolved_fails(self, manifest_with_pinned, monkeypatch):
        """If a pinned key is not present in resolved config at all → FAIL (typo/missing)."""
        from gigaevo.experiment import checks as checks_mod

        exp, _ = manifest_with_pinned
        stub = _make_stub(
            {
                "A1": {"num_parents": 1},  # n_opponents missing entirely
                "A2": {"num_parents": 1, "n_opponents": 5},
            }
        )
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_resolved_config_matches_pinned(results, m, exp)
        r = results[0]
        assert not r.passed
        assert "n_opponents" in r.message

    def test_dry_run_failure_surfaces_as_critical(
        self, manifest_with_pinned, monkeypatch
    ):
        """Hydra compose error in dry_run → check fails CRITICAL, doesn't crash."""
        from gigaevo.experiment import checks as checks_mod

        def _bad(experiment, **kwargs):
            raise RuntimeError("Hydra compose failed: Key 'pipeline' ...")

        monkeypatch.setattr(checks_mod, "dry_run", _bad)

        results: list = []
        exp, _ = manifest_with_pinned
        m = checks_mod.load_manifest(exp)
        checks_mod._check_resolved_config_matches_pinned(results, m, exp)
        r = results[0]
        assert not r.passed
        assert r.is_blocking
        assert "Hydra" in r.message or "compose" in r.message.lower()

    def test_empty_pinned_passes_advisory(self, tmp_path, monkeypatch):
        """When contract.config.pinned is empty AND no per-run pins → check passes (no assertions to make)."""
        exp_dir = tmp_path / "experiments" / "toy" / "no-pin"
        exp_dir.mkdir(parents=True)
        yaml_content = textwrap.dedent("""\
            schema_version: 2
            contract:
              identity:
                name: toy/no-pin
                task: toy
                branch: test-branch
              problem:
                has_test_set: false
                fitness_type: fractional
                metric_name: fitness
              config:
                extra:
                  stage_timeout: 600
              runs:
              - label: A1
                db: 15
                prefix: test_prefix
                pipeline: standard
                problem_name: toy_kadane
                condition: control
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
        monkeypatch.setattr("gigaevo.experiment.manifest.PROJ", tmp_path)

        from gigaevo.experiment import checks as checks_mod

        stub = _make_stub({"A1": {"n_opponents": 1}})
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest("toy/no-pin")
        checks_mod._check_resolved_config_matches_pinned(results, m, "toy/no-pin")
        r = results[0]
        assert r.passed, r.message


class TestConfigFingerprintStable:
    def test_fresh_launch_no_fingerprint_recorded_passes(
        self, manifest_with_pinned, monkeypatch
    ):
        """Empty lifecycle.launch.config_fingerprint → first-launch, skip check."""
        from gigaevo.experiment import checks as checks_mod

        exp, _ = manifest_with_pinned
        stub = _make_stub(
            {"A1": {}, "A2": {}}, fingerprint={"config/config.yaml": "abc"}
        )
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_config_fingerprint_stable(results, m, exp)
        r = results[0]
        assert r.passed, r.message
        assert "fresh" in r.message.lower() or "no fingerprint" in r.message.lower()

    def test_relaunch_matching_fingerprint_passes(
        self, manifest_with_pinned, monkeypatch
    ):
        """Recorded fingerprint matches current → PASS."""
        from gigaevo.experiment import checks as checks_mod
        from gigaevo.experiment.manifest import update_manifest

        exp, _ = manifest_with_pinned
        fp = {"config/config.yaml": "abc", "config/experiment/heilbron.yaml": "def"}

        def set_fp(raw):
            raw.setdefault("lifecycle", {}).setdefault("launch", {})[
                "config_fingerprint"
            ] = fp

        update_manifest(exp, set_fp)

        stub = _make_stub({"A1": {}, "A2": {}}, fingerprint=fp)
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_config_fingerprint_stable(results, m, exp)
        r = results[0]
        assert r.passed, r.message

    def test_relaunch_drifted_fingerprint_fails_critical(
        self, manifest_with_pinned, monkeypatch
    ):
        """Recorded fingerprint differs from current → CRITICAL."""
        from gigaevo.experiment import checks as checks_mod
        from gigaevo.experiment.manifest import update_manifest

        exp, _ = manifest_with_pinned
        recorded = {
            "config/config.yaml": "abc",
            "config/experiment/heilbron.yaml": "def",
        }

        def set_fp(raw):
            raw.setdefault("lifecycle", {}).setdefault("launch", {})[
                "config_fingerprint"
            ] = recorded

        update_manifest(exp, set_fp)

        # Current scan sees a DIFFERENT sha for heilbron.yaml
        current = {
            "config/config.yaml": "abc",
            "config/experiment/heilbron.yaml": "CHANGED",
        }
        stub = _make_stub({"A1": {}, "A2": {}}, fingerprint=current)
        monkeypatch.setattr(checks_mod, "dry_run", stub)

        results: list = []
        m = checks_mod.load_manifest(exp)
        checks_mod._check_config_fingerprint_stable(results, m, exp)
        r = results[0]
        assert not r.passed
        assert r.is_blocking
        assert "heilbron" in r.message or "drift" in r.message.lower()
