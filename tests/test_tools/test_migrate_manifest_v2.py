"""Tests for the one-shot v1→v2 manifest migration CLI (step 6).

``tools/experiment/migrate_manifest_v2.py`` converts flat v1 yamls to the
nested v2 shape introduced in step 4 and validated through ``OmegaConf``
in step 5. The CLI wraps :func:`gigaevo.experiment.manifest._migrate_v1_to_v2`
— a pure dict transform already tested in
``tests/experiment/test_manifest_v2_groups.py`` — and adds two things that
the in-memory migration does not provide:

  1. File-level operations: discovery of all experiment yamls, dry-run
     diff preview, atomic apply.
  2. Heuristic parsing of ``experiment.stopping_rule`` (prose) into a
     structured ``StoppingRule`` model with a ``NEEDS_REVIEW`` marker
     where the parser cannot extract conditions.

These tests lock down those two behaviors. The pure-transform contract is
covered by the existing test suite; here we only verify the CLI wrapping.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml

from gigaevo.experiment.manifest import ExperimentManifest

# The module we're about to create.
from tools.experiment.migrate_manifest_v2 import (
    find_v1_yamls,
    migrate_file,
    parse_stopping_rule,
)

REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Fixture: a representative v1 manifest dict
# ---------------------------------------------------------------------------


def _v1_fixture() -> dict:
    """A v1 dict with enough shape to exercise the migrator.

    Includes: experiment identity, problem, runs, watchdog with
    plot_commands + alert_thresholds, launch with cron IDs, checkpoints,
    mid_run_test_eval, checkpoint_analysis, treatment_checks.
    """
    return {
        "schema_version": 1,
        "experiment": {
            "name": "hover/demo",
            "task": "hover",
            "branch": "exp/hover/demo",
            "status": "running",
            "max_generations": 50,
            "stopping_rule": ("max_generations=50 OR futility_at_gen25(metric < 0.03)"),
            "prereg_commit": "abc123",
            "pr_number": 42,
            "tracking_issue": None,
        },
        "problem": {
            "has_test_set": True,
            "fitness_type": "continuous",
            "metric_name": "actual_fitness",
        },
        "runs": [
            {
                "label": "R1",
                "db": 5,
                "prefix": "r1",
                "pipeline": "standard",
                "problem_name": "hover",
                "condition": "control",
                "chain_url": "http://example.com",
                "mutation_url": "http://example.com",
                "model_name": "gpt-4",
                "pid": 1234,
            }
        ],
        "servers": ["10.0.0.1"],
        "custom_env": {"OPENAI_API_KEY": "sk-demo"},
        "config": {"num_parents": 1, "pipeline": "standard"},
        "watchdog": {
            "plugin": "solo",
            "sentinel_value": -1.0,
            "plot_metrics": ["actual_fitness"],
            "plot_commands": [
                {
                    "command": "comparison",
                    "args": {"metric": "actual_fitness", "smoothing": "ema"},
                    "output_name": "comparison.png",
                    "caption": "Run comparison",
                }
            ],
            "alert_thresholds": {
                "invalidity_rate": 0.75,
                "stagnation_window": 10,
                "generation_gap_threshold": 5,
            },
            "poll_interval_s": 3600,
            "plot_retries": 3,
            "plot_retry_delay_s": 30,
            "rolling_comment_threshold_hours": 24,
            "checkpoint_milestones": [0.1, 0.2, 0.5, 1.0],
            "no_proxy_hosts": [],
        },
        "launch": {
            "time": "2026-04-14T15:08:16Z",
            "commit": "4022d407",
            "confirmed_at": "2026-04-14T15:08:20Z",
            "watchdog_pid": 1133798,
            "anomaly_detector_cron_id": "17f15a1c",
            "checkpoint_cron_id": "1960d819",
        },
        "checkpoints": [
            {
                "gen": 12,
                "timestamp": "2026-04-14T23:26:49Z",
                "run_metrics": [
                    {"label": "R1", "gen": 14, "best_actual_fitness": 0.027}
                ],
                "notes": "",
            }
        ],
        "mid_run_test_eval": {"completed": False},
        "checkpoint_analysis": {
            "mid_run": {"completed": True, "completed_at": "2026-04-15T03:27:20Z"}
        },
        # smoke_test + treatment_verification are required on every live v1
        # yaml that reaches status=running; include them here so the
        # migrated output validates against ExperimentManifest.
        "smoke_test": {
            "completed": True,
            "completed_at": "2026-04-14T12:00:00Z",
            "generations": 3,
        },
        "treatment_verification": {
            "completed": True,
            "completed_at": "2026-04-14T12:05:00Z",
            "note": "verified via smoke test",
        },
        "baseline": {
            "reference": "hover/baseline",
            "mean": 0.03449,
            "metric": "actual_fitness",
        },
    }


# ---------------------------------------------------------------------------
# parse_stopping_rule: prose → structured StoppingRule
# ---------------------------------------------------------------------------


class TestParseStoppingRule:
    """Heuristic parser turns prose stopping_rule into StoppingRule shape.

    The migration cannot invent scientific intent — when the prose is
    ambiguous, the output carries a ``NEEDS_REVIEW`` marker in the
    description so the researcher can fix it before committing.
    """

    def test_empty_prose_yields_empty_rule(self):
        result = parse_stopping_rule("")
        assert result == {
            "description": "",
            "conditions": [],
            "enforce_at": "checkpoint",
        }

    def test_max_generations_only_extracts_no_conditions(self):
        result = parse_stopping_rule("max_generations=50")
        assert result["description"] == "max_generations=50"
        assert result["conditions"] == []

    def test_fitness_plateau_pattern_extracted(self):
        """``futility_at_gen<N>(... < <threshold>)`` → fitness_plateau."""
        result = parse_stopping_rule(
            "max_generations=50 OR futility_at_gen25(metric < 0.03)"
        )
        assert result["description"].startswith("max_generations=50")
        conditions = result["conditions"]
        assert len(conditions) == 1
        c = conditions[0]
        assert c["kind"] == "fitness_plateau"
        assert c["threshold"] == 0.03
        assert c["window"] == 25

    def test_unparseable_prose_gets_needs_review_marker(self):
        prose = "eventually it feels done"
        result = parse_stopping_rule(prose)
        assert "NEEDS_REVIEW" in result["description"]
        assert prose in result["description"]  # original prose preserved
        assert result["conditions"] == []


# ---------------------------------------------------------------------------
# migrate_file: file-level dry-run + apply
# ---------------------------------------------------------------------------


class TestMigrateFileDryRun:
    """--dry-run must not touch the file; must report what would change."""

    def test_dry_run_does_not_write(self, tmp_path: Path):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(_v1_fixture()))
        before = path.read_text()

        result = migrate_file(path, apply=False)

        assert path.read_text() == before
        assert result["changed"] is True
        assert result["new_content"] is not None
        # New content must be valid v2 (loads + has nested groups)
        out = yaml.safe_load(result["new_content"])
        assert out["schema_version"] == 2
        assert "contract" in out
        assert "lifecycle" in out
        assert "telemetry" in out
        assert "control_plane" in out

    def test_dry_run_on_already_v2_file_is_noop(self, tmp_path: Path):
        v2 = {
            "schema_version": 2,
            "contract": {
                "identity": {"name": "x/y", "task": "x"},
                "problem": {},
                "runs": [],
                "servers": [],
                "custom_env": {},
                "max_generations": 25,
                "stopping_rule": {"description": ""},
                "tools": [],
            },
            "lifecycle": {"status": "preregistered"},
            "telemetry": {},
            "control_plane": {},
        }
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(v2))

        result = migrate_file(path, apply=False)

        assert result["changed"] is False


class TestMigrateFileApply:
    """--apply writes the v2 shape atomically; output validates."""

    def test_apply_writes_v2_file(self, tmp_path: Path):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(_v1_fixture()))

        result = migrate_file(path, apply=True)

        assert result["changed"] is True
        loaded = yaml.safe_load(path.read_text())
        assert loaded["schema_version"] == 2
        assert loaded["contract"]["identity"]["name"] == "hover/demo"
        assert loaded["contract"]["identity"]["task"] == "hover"
        assert loaded["contract"]["max_generations"] == 50
        # Structured stopping rule derived from prose
        sr = loaded["contract"]["stopping_rule"]
        assert "max_generations=50" in sr["description"]
        assert any(c["kind"] == "fitness_plateau" for c in sr["conditions"])

    def test_apply_preserves_plot_commands_bit_for_bit(self, tmp_path: Path):
        path = tmp_path / "experiment.yaml"
        src = _v1_fixture()
        path.write_text(yaml.safe_dump(src))

        migrate_file(path, apply=True)

        loaded = yaml.safe_load(path.read_text())
        watchdog = loaded["control_plane"]["watchdog"]
        assert watchdog["plot_commands"] == src["watchdog"]["plot_commands"]
        assert watchdog["alert_thresholds"] == src["watchdog"]["alert_thresholds"]
        assert (
            watchdog["checkpoint_milestones"]
            == src["watchdog"]["checkpoint_milestones"]
        )

    def test_apply_moves_cron_ids_to_control_plane(self, tmp_path: Path):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(_v1_fixture()))

        migrate_file(path, apply=True)

        loaded = yaml.safe_load(path.read_text())
        cp = loaded["control_plane"]
        assert cp["watchdog_pid"] == 1133798
        assert cp["anomaly_detector_cron_id"] == "17f15a1c"
        assert cp["checkpoint_cron_id"] == "1960d819"
        # And launch section no longer has them
        launch = loaded["lifecycle"]["launch"]
        assert "watchdog_pid" not in launch
        assert "anomaly_detector_cron_id" not in launch
        assert "checkpoint_cron_id" not in launch

    def test_apply_output_validates_against_experiment_manifest(self, tmp_path: Path):
        """Migrated yaml must load cleanly through ExperimentManifest."""
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(_v1_fixture()))

        migrate_file(path, apply=True)

        # ExperimentManifest currently accepts v2 via extra="allow" on the
        # top-level model; once callers migrate (step 9) it will use the
        # typed sub-sections directly. Either way, the validator must not
        # reject the v2 dict.
        loaded = yaml.safe_load(path.read_text())
        ExperimentManifest.from_dict(loaded)


# ---------------------------------------------------------------------------
# find_v1_yamls: discovery
# ---------------------------------------------------------------------------


class TestFindV1Yamls:
    def test_finds_all_live_yamls(self, tmp_path: Path):
        (tmp_path / "experiments" / "hover" / "a").mkdir(parents=True)
        (tmp_path / "experiments" / "hover" / "a" / "experiment.yaml").write_text(
            "schema_version: 1\n"
        )
        (tmp_path / "experiments" / "hotpotqa" / "b").mkdir(parents=True)
        (tmp_path / "experiments" / "hotpotqa" / "b" / "experiment.yaml").write_text(
            "schema_version: 1\n"
        )

        found = find_v1_yamls(tmp_path / "experiments")

        assert len(found) == 2
        names = {p.parent.name for p in found}
        assert names == {"a", "b"}

    def test_skips_template_dir_by_default(self, tmp_path: Path):
        (tmp_path / "experiments" / "_template").mkdir(parents=True)
        (tmp_path / "experiments" / "_template" / "experiment.yaml").write_text(
            "schema_version: 1\n"
        )
        (tmp_path / "experiments" / "hover" / "a").mkdir(parents=True)
        (tmp_path / "experiments" / "hover" / "a" / "experiment.yaml").write_text(
            "schema_version: 1\n"
        )

        found = find_v1_yamls(tmp_path / "experiments")

        assert len(found) == 1
        assert found[0].parent.name == "a"


# ---------------------------------------------------------------------------
# CLI smoke test: actually invoke tools/experiment/migrate_manifest_v2.py
# ---------------------------------------------------------------------------


class TestCliSmoke:
    def test_single_dry_run_prints_diff_and_exits_zero(self, tmp_path: Path):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(_v1_fixture()))

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "experiment" / "migrate_manifest_v2.py"),
                "--single",
                str(path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # File unchanged
        assert yaml.safe_load(path.read_text())["schema_version"] == 1

    def test_single_apply_rewrites_file(self, tmp_path: Path):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(_v1_fixture()))

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "experiment" / "migrate_manifest_v2.py"),
                "--single",
                str(path),
                "--apply",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert yaml.safe_load(path.read_text())["schema_version"] == 2
