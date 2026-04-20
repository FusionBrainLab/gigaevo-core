"""Tests for Phase 5 — LAUNCH_PREVIEW.md writer.

Verifies ``write_launch_preview`` renders a markdown report with:
  - Per-run resolved-config table showing provenance (extra_overrides /
    shared_overrides / task-group / hydra-default) and pin match status
  - Hydra default fingerprint table
  - Header with task_group and pass/fail summary

The writer does not invoke ``dry_run`` — it consumes a ``DryRunResult``.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from gigaevo.experiment.dry_run import DryRunResult


@pytest.fixture()
def manifest_for_preview(tmp_path: Path, monkeypatch):
    exp_dir = tmp_path / "experiments" / "toy" / "preview"
    exp_dir.mkdir(parents=True)

    # Minimal task-group file so the writer can report it as a source.
    cfg = tmp_path / "config" / "experiment"
    cfg.mkdir(parents=True)
    (cfg / "heilbron.yaml").write_text("num_parents: 1\nstage_timeout: 2400\n")

    yaml_content = textwrap.dedent("""\
        schema_version: 2
        contract:
          identity:
            name: toy/preview
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
              source_prompt_k: 3
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
            extra_overrides:
              - pipeline_builder.archive_reeval=true
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
    return "toy/preview", tmp_path


@pytest.fixture()
def result_for_preview():
    """A DryRunResult with one run, mixing several provenance sources."""
    return DryRunResult(
        resolved={
            "A1": {
                "num_parents": 1,  # task-group
                "n_opponents": 3,  # shared_overrides
                "source_prompt_k": 3,  # shared_overrides
                "stage_timeout": 2400,  # task-group
                "pipeline_builder": {"archive_reeval": True},  # extra_overrides
            }
        },
        fingerprint={
            "config/config.yaml": "a" * 64,
            "config/experiment/heilbron.yaml": "b" * 64,
        },
        cli_args={
            "A1": [
                "experiment=heilbron",
                "n_opponents=3",
                "source_prompt_k=3",
                "pipeline_builder.archive_reeval=true",
            ]
        },
    )


class TestWriteLaunchPreview:
    def test_creates_file_at_expected_path(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, tmp_path = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        assert out.exists()
        assert out == tmp_path / "experiments" / exp / "LAUNCH_PREVIEW.md"

    def test_header_names_task_group(self, manifest_for_preview, result_for_preview):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        text = out.read_text()
        assert "heilbron" in text
        assert "Launch Preview" in text

    def test_reports_pass_when_all_pins_satisfied(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        text = out.read_text()
        # Both pins (n_opponents=3, num_parents=1) are satisfied
        assert "PASS" in text
        assert "2 pin" in text or "2/2" in text

    def test_reports_fail_with_drift(self, manifest_for_preview):
        from gigaevo.experiment.launch_preview import write_launch_preview

        # num_parents pinned=1 but resolved=2 → drift
        bad_result = DryRunResult(
            resolved={"A1": {"num_parents": 2, "n_opponents": 3, "source_prompt_k": 3}},
            fingerprint={"config/config.yaml": "a" * 64},
            cli_args={"A1": ["n_opponents=3", "source_prompt_k=3"]},
        )
        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, bad_result)
        text = out.read_text()
        assert "FAIL" in text or "✗" in text or "MISMATCH" in text.upper()

    def test_per_run_table_shows_pin_rows(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        text = out.read_text()
        assert "A1" in text
        assert "num_parents" in text
        assert "n_opponents" in text

    def test_provenance_reflects_extra_overrides(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        text = out.read_text()
        # archive_reeval came from extra_overrides — must appear somewhere
        assert "archive_reeval" in text
        assert "extra_overrides" in text

    def test_provenance_reflects_shared_overrides(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        text = out.read_text()
        # n_opponents came from shared_overrides; table column header must exist
        assert "shared_overrides" in text

    def test_fingerprint_table_includes_all_files(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out = write_launch_preview(exp, result_for_preview)
        text = out.read_text()
        assert "config/config.yaml" in text
        assert "config/experiment/heilbron.yaml" in text
        # sha prefix (first 8 chars) should be visible so reviewers can diff
        assert "aaaaaaaa" in text or "a" * 8 in text

    def test_is_idempotent_and_overwrites(
        self, manifest_for_preview, result_for_preview
    ):
        from gigaevo.experiment.launch_preview import write_launch_preview

        exp, _ = manifest_for_preview
        out1 = write_launch_preview(exp, result_for_preview)
        text1 = out1.read_text()
        out2 = write_launch_preview(exp, result_for_preview)
        text2 = out2.read_text()

        # Strip the generated-timestamp line to compare stable body
        def _strip_ts(t: str) -> str:
            return "\n".join(
                ln for ln in t.splitlines() if not ln.lower().startswith("**generated")
            )

        assert _strip_ts(text1) == _strip_ts(text2)
