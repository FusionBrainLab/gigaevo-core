"""Tests for schema v2 sub-model groups (step 4).

The four top-level sub-model groups on ExperimentManifest:

    ExperimentManifest
      ├── contract:      ContractSection
      ├── lifecycle:     LifecycleState
      ├── telemetry:     TelemetryLog
      └── control_plane: ControlPlane

Also asserts that ``plot_commands`` round-trip correctly through the nested
v2 sections (watchdog fence).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gigaevo.experiment.manifest import (
    SUPPORTED_SCHEMA_VERSIONS,
    ContractSection,
    ControlPlane,
    ExperimentIdentity,
    ExperimentManifest,
    LifecycleState,
    TelemetryLog,
)

REPO_ROOT = Path(__file__).parent.parent.parent


def _all_v1_yamls() -> list[Path]:
    return [
        p
        for p in (REPO_ROOT / "experiments").glob("*/*/experiment.yaml")
        if "_template" not in str(p)
    ]


# ---------------------------------------------------------------------------
# Schema version — v2 only (step 8 removed v1 support)
# ---------------------------------------------------------------------------


class TestSchemaVersionRange:
    def test_accepts_v2_only(self):
        assert SUPPORTED_SCHEMA_VERSIONS == {2}


# ---------------------------------------------------------------------------
# New sub-model group types
# ---------------------------------------------------------------------------


class TestExperimentIdentity:
    def test_minimum_fields(self):
        i = ExperimentIdentity(name="hover/foo", task="hover")
        assert i.name == "hover/foo"
        assert i.task == "hover"
        assert i.branch == ""
        assert i.prereg_commit is None
        assert i.pr_number is None
        assert i.tracking_issue is None

    def test_full(self):
        i = ExperimentIdentity(
            name="hover/foo",
            task="hover",
            branch="exp/hover/foo",
            prereg_commit="abc123",
            pr_number=42,
            tracking_issue=17,
        )
        assert i.branch == "exp/hover/foo"
        assert i.pr_number == 42


class TestContractSectionShape:
    """ContractSection bundles the pre-registered, researcher-authored facts."""

    def test_defaults(self):
        c = ContractSection(identity=ExperimentIdentity(name="x/y", task="x"))
        assert c.runs == []
        assert c.servers == []
        assert c.custom_env == {}
        assert c.tools == []
        assert c.max_generations == 25


class TestLifecycleStateShape:
    def test_defaults(self):
        s = LifecycleState(status="preregistered")
        assert s.status == "preregistered"
        assert s.launch.time is None
        assert s.smoke_test.completed is False
        assert s.treatment_verification.completed is False


class TestTelemetryLogShape:
    def test_defaults(self):
        t = TelemetryLog()
        assert t.checkpoints == []
        assert t.mid_run_test_eval.completed is False
        assert t.checkpoint_analysis.mid_run.completed is False
        assert t.treatment_checks.completed is False


class TestControlPlaneShape:
    def test_defaults(self):
        cp = ControlPlane()
        # Watchdog defaults mirror the existing WatchdogSection defaults.
        assert cp.watchdog.poll_interval_s == 3600
        assert cp.watchdog.plot_commands == []
        # Notifications default both channels enabled.
        assert cp.notifications.pr.enabled is True
        assert cp.notifications.telegram.enabled is True
        assert cp.watchdog_pid is None
        assert cp.anomaly_detector_cron_id is None
        assert cp.checkpoint_cron_id is None


# ---------------------------------------------------------------------------
# ExperimentManifest sub-group views (canonical nested access)
# ---------------------------------------------------------------------------


def _heilbron_v2_yaml() -> dict:
    path = (
        REPO_ROOT
        / "experiments"
        / "heilbron"
        / "asymmetric-iterations-v2"
        / "experiment.yaml"
    )
    return yaml.safe_load(path.read_text())


class TestManifestContractView:
    def test_identity_populated(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.contract.identity.name
        assert m.contract.identity.task

    def test_contract_runs_present(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert len(m.contract.runs) > 0
        # Every run has a label
        assert all(r.label for r in m.contract.runs)

    def test_contract_servers_present(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.contract.servers

    def test_contract_max_generations_is_int(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert isinstance(m.contract.max_generations, int)
        assert m.contract.max_generations > 0


class TestManifestLifecycleView:
    def test_status_set(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.lifecycle.status
        assert isinstance(m.lifecycle.status, str)

    def test_launch_time_populated(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        # heilbron v2 is running/complete, so launch should be set
        assert m.lifecycle.launch.time
        assert m.lifecycle.launch.commit

    def test_smoke_test_completed(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.lifecycle.smoke_test.completed is True

    def test_treatment_verification_boolean(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert isinstance(m.lifecycle.treatment_verification.completed, bool)


class TestManifestTelemetryView:
    def test_checkpoints_typed_as_checkpoint_entries(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        first = m.telemetry.checkpoints[0]
        # It's the typed CheckpointEntry model, not a dict.
        assert first.gen > 0
        assert first.run_metrics  # heilbron checkpoints have run_metrics
        # best_actual_fitness preserved via RunMetric.extra_allow
        rm = first.run_metrics[0]
        assert rm.model_dump()["best_actual_fitness"] > 0

    def test_checkpoint_analysis_mid_run_completed(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        # heilbron-v2 has checkpoint_analysis.mid_run.completed: true
        assert m.telemetry.checkpoint_analysis.mid_run.completed is True


class TestManifestControlPlaneView:
    def test_watchdog_has_plugin(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        # heilbron v2 uses a watchdog plugin
        assert m.control_plane.watchdog.plugin

    def test_watchdog_pid_populated(self):
        """``watchdog_pid`` is sourced from nested ``control_plane.watchdog_pid``."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.watchdog_pid == 1133798

    def test_cron_ids_populated(self):
        """Cron IDs live under ``control_plane.*`` in v2."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.anomaly_detector_cron_id == "17f15a1c"
        assert m.control_plane.checkpoint_cron_id == "1960d819"

    def test_notifications_defaults_both_enabled(self):
        """v1 yamls never set notifications — defaults must be on."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.notifications.pr.enabled is True
        assert m.control_plane.notifications.telegram.enabled is True


# ---------------------------------------------------------------------------
# Watchdog fence — every live yaml's control_plane.watchdog round-trips
# bit-for-bit through the nested section and validates cleanly.
# ---------------------------------------------------------------------------


class TestWatchdogFencePlotCommands:
    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_plot_commands_preserved(self, yaml_path: Path):
        """After flatten migration, all YAMLs are nested-only. Verify plot_commands
        are preserved in the nested control_plane section and model-validate."""
        raw = yaml.safe_load(yaml_path.read_text())

        # Post-flatten, nested is the only source of truth.
        nested_pc = ((raw.get("control_plane") or {}).get("watchdog") or {}).get(
            "plot_commands", []
        )

        # Verify the schema accepts it (model_validate catches any corruption).
        manifest = ExperimentManifest.from_dict(raw)
        loaded_pc = manifest.control_plane.watchdog.plot_commands

        # Both should match: nested raw = validated model.
        assert len(loaded_pc) == len(nested_pc), (
            f"plot_commands count diverged for {yaml_path}: "
            f"nested_raw={len(nested_pc)}, loaded={len(loaded_pc)}"
        )
        for i, cmd in enumerate(loaded_pc):
            assert cmd.command == nested_pc[i]["command"]
            assert cmd.output_name == nested_pc[i].get("output_name", "")

    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_alert_thresholds_preserved(self, yaml_path: Path):
        """After flatten, verify alert_thresholds exist in nested control_plane."""
        raw = yaml.safe_load(yaml_path.read_text())
        # Verify it loads without error.
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.control_plane.watchdog.alert_thresholds is not None

    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_checkpoint_milestones_preserved(self, yaml_path: Path):
        """After flatten, verify checkpoint_milestones exist in nested control_plane."""
        raw = yaml.safe_load(yaml_path.read_text())
        # Verify it loads without error.
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.control_plane.watchdog.checkpoint_milestones is not None


# ---------------------------------------------------------------------------
# Regression: every migrated yaml validates against ExperimentManifest
# ---------------------------------------------------------------------------


class TestEveryYamlValidates:
    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_every_real_yaml_validates(self, yaml_path: Path):
        raw = yaml.safe_load(yaml_path.read_text())
        m = ExperimentManifest.from_dict(raw)
        assert m.schema_version == 2
