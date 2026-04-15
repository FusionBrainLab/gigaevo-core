"""Tests for schema v2 sub-model groups + v1→v2 migration (step 4).

Step 4 introduces the four top-level sub-model groups on ExperimentManifest:

    ExperimentManifest
      ├── contract:      ContractSection
      ├── lifecycle:     LifecycleState
      ├── telemetry:     TelemetryLog
      └── control_plane: ControlPlane

For v1 yamls, these are computed views over the existing flat fields so
existing callers continue to work unchanged. Step 5 will switch the loader
to OmegaConf and make v2 (nested) the canonical on-disk shape.

Also locks down ``_migrate_v1_to_v2`` — a pure dict transformation used by
the step 6 migration CLI — and asserts that ``plot_commands`` round-trip
bit-for-bit through migration (watchdog fence).
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
    _migrate_v1_to_v2,
)

REPO_ROOT = Path(__file__).parent.parent.parent


def _all_v1_yamls() -> list[Path]:
    return [
        p
        for p in (REPO_ROOT / "experiments").glob("*/*/experiment.yaml")
        if "_template" not in str(p)
    ]


# ---------------------------------------------------------------------------
# Schema version — now accepts both 1 and 2
# ---------------------------------------------------------------------------


class TestSchemaVersionRange:
    def test_accepts_v1_and_v2(self):
        assert SUPPORTED_SCHEMA_VERSIONS == {1, 2}


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
        # StoppingRule default (description="" means prose not set yet)
        assert c.stopping_rule.description == ""
        assert c.stopping_rule.conditions == []


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
# ExperimentManifest gains group views (computed from flat for v1 inputs)
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
    def test_identity_mirrors_experiment_section(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.contract.identity.name == m.experiment.name
        assert m.contract.identity.task == m.experiment.task
        assert m.contract.identity.branch == m.experiment.branch
        assert m.contract.identity.pr_number == m.experiment.pr_number

    def test_contract_runs_match_flat_runs(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert len(m.contract.runs) == len(m.runs)
        assert [r.label for r in m.contract.runs] == [r.label for r in m.runs]

    def test_contract_servers_match_flat(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.contract.servers == m.servers

    def test_contract_max_generations_prefers_experiment_section(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.contract.max_generations == m.experiment.max_generations

    def test_contract_stopping_rule_description_from_flat_prose(self):
        """v1 prose stopping_rule lifts into StoppingRule.description."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.experiment.stopping_rule  # prose exists on the flat yaml
        assert m.contract.stopping_rule.description == m.experiment.stopping_rule
        # conditions stay empty until an explicit structured rule is authored
        assert m.contract.stopping_rule.conditions == []


class TestManifestLifecycleView:
    def test_status_mirrors_flat(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.lifecycle.status == m.experiment.status

    def test_launch_time_mirrors_flat(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.lifecycle.launch.time == m.launch.time
        assert m.lifecycle.launch.commit == m.launch.commit

    def test_smoke_test_mirrors_flat(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.lifecycle.smoke_test.completed == m.smoke_test.completed

    def test_treatment_verification_mirrors_flat(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert (
            m.lifecycle.treatment_verification.completed
            == m.treatment_verification.completed
        )


class TestManifestTelemetryView:
    def test_checkpoints_typed_as_checkpoint_entries(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert len(m.telemetry.checkpoints) == len(m.checkpoints)
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
    def test_watchdog_mirrors_flat(self):
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.watchdog == m.watchdog

    def test_watchdog_pid_lifts_from_launch(self):
        """``watchdog_pid`` is sourced from either the v1 flat ``launch.*``
        field or the v2 nested ``control_plane.watchdog_pid`` — whichever
        is present in the manifest."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.watchdog_pid == 1133798

    def test_cron_ids_lift_from_launch_extras(self):
        """Cron IDs live under ``control_plane.*`` in v2; v1 kept them under
        ``launch.*``. The migration and property resolver accept either."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.anomaly_detector_cron_id == "17f15a1c"
        assert m.control_plane.checkpoint_cron_id == "1960d819"

    def test_notifications_defaults_both_enabled(self):
        """v1 yamls never set notifications — defaults must be on."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        assert m.control_plane.notifications.pr.enabled is True
        assert m.control_plane.notifications.telegram.enabled is True


# ---------------------------------------------------------------------------
# _migrate_v1_to_v2 — pure dict transform used by step 6 migration CLI
# ---------------------------------------------------------------------------


class TestMigrateV1ToV2Core:
    """Top-level structural transformation."""

    def test_bumps_schema_version(self):
        v1 = _heilbron_v2_yaml()
        v2 = _migrate_v1_to_v2(v1)
        assert v2["schema_version"] == 2

    def test_produces_four_sections(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert set(["contract", "lifecycle", "telemetry", "control_plane"]).issubset(
            set(v2.keys())
        )

    def test_input_not_mutated(self):
        v1 = _heilbron_v2_yaml()
        snapshot = yaml.safe_dump(v1)
        _migrate_v1_to_v2(v1)
        # Pure function — input should survive unchanged.
        assert yaml.safe_dump(v1) == snapshot

    def test_output_validates_with_experiment_manifest(self):
        """The v2 shape must load through the model without error."""
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        m = ExperimentManifest.from_dict(v2)
        assert m.schema_version == 2


class TestMigrateV1ToV2Identity:
    def test_identity_fields_move_under_contract(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        idv = v2["contract"]["identity"]
        assert idv["name"] == "heilbron/asymmetric-iterations-v2"
        assert idv["task"] == "heilbron"
        assert idv["branch"] == "exp/heilbron/asymmetric-iterations-v2"
        assert idv["pr_number"] == 206
        assert idv["prereg_commit"] == "3f4dc1e6"


class TestMigrateV1ToV2ContractContents:
    def test_runs_preserved(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert len(v2["contract"]["runs"]) == 8
        assert v2["contract"]["runs"][0]["label"] == "A1_G"
        # roles preserved
        assert v2["contract"]["runs"][0]["role"] == "constructor"

    def test_servers_preserved(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["contract"]["servers"] == ["INTERNAL_IP"]

    def test_custom_env_preserved(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["contract"]["custom_env"]["OPENAI_API_KEY"] == "sk-gigaevo"

    def test_config_becomes_configspec_extra(self):
        """Problem-specific knobs land in contract.config.extra."""
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        extra = v2["contract"]["config"]["extra"]
        assert extra["num_parents"] == 1
        assert extra["max_elites_per_generation"] == 8

    def test_max_generations_lives_under_contract(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["contract"]["max_generations"] == 50

    def test_baseline_preserved(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["contract"]["baseline"]["mean"] == 0.03449

    def test_stopping_rule_prose_becomes_description(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        sr = v2["contract"]["stopping_rule"]
        assert "max_generations=50" in sr["description"]
        assert isinstance(sr.get("conditions", []), list)


class TestMigrateV1ToV2Lifecycle:
    def test_status_lifts_into_lifecycle(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["lifecycle"]["status"] == "running"

    def test_launch_fields_land_under_lifecycle(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        launch = v2["lifecycle"]["launch"]
        assert launch["time"] == "2026-04-14T15:08:16Z"
        assert launch["commit"].startswith("4022d407")

    def test_smoke_test_lifts(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["lifecycle"]["smoke_test"]["completed"] is True

    def test_treatment_verification_lifts(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["lifecycle"]["treatment_verification"]["completed"] is True


class TestMigrateV1ToV2Telemetry:
    def test_checkpoints_lift(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert len(v2["telemetry"]["checkpoints"]) == 3
        # Shape preserved — run_metrics and nested best_actual_fitness intact
        first = v2["telemetry"]["checkpoints"][0]
        assert first["run_metrics"][0]["best_actual_fitness"] > 0

    def test_checkpoint_analysis_lifts(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["telemetry"]["checkpoint_analysis"]["mid_run"]["completed"] is True

    def test_mid_run_test_eval_absent_in_heilbron_but_valid_default(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        mre = v2["telemetry"].get("mid_run_test_eval", {})
        # either absent or present with completed=false — both acceptable
        assert mre.get("completed", False) is False

    def test_treatment_checks_absent_in_heilbron_ok(self):
        """heilbron-v2 doesn't carry v1 treatment_checks — migration emits empty."""
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        tc = v2["telemetry"].get("treatment_checks", {})
        assert tc.get("completed", False) is False
        assert tc.get("results", []) == []


class TestMigrateV1ToV2ControlPlane:
    def test_watchdog_moves_under_control_plane(self):
        v1 = _heilbron_v2_yaml()
        v2 = _migrate_v1_to_v2(v1)
        assert v2["control_plane"]["watchdog"] == v1["watchdog"]

    def test_watchdog_pid_lifts_from_launch(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["control_plane"]["watchdog_pid"] == 1133798

    def test_cron_ids_lift_from_launch(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        assert v2["control_plane"]["anomaly_detector_cron_id"] == "17f15a1c"
        assert v2["control_plane"]["checkpoint_cron_id"] == "1960d819"

    def test_launch_no_longer_carries_watchdog_pid(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        launch = v2["lifecycle"]["launch"]
        # These keys moved to control_plane — launch should not duplicate them.
        assert "watchdog_pid" not in launch
        assert "anomaly_detector_cron_id" not in launch
        assert "checkpoint_cron_id" not in launch

    def test_notifications_defaulted_when_absent(self):
        v2 = _migrate_v1_to_v2(_heilbron_v2_yaml())
        notif = v2["control_plane"]["notifications"]
        assert notif["pr"]["enabled"] is True
        assert notif["telegram"]["enabled"] is True


# ---------------------------------------------------------------------------
# Watchdog fence — plot_commands round-trip bit-for-bit (hard requirement)
# ---------------------------------------------------------------------------


class TestWatchdogFencePlotCommands:
    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_plot_commands_preserved_through_migration(self, yaml_path: Path):
        v1 = yaml.safe_load(yaml_path.read_text())
        original_pc = (v1.get("watchdog") or {}).get("plot_commands", [])
        v2 = _migrate_v1_to_v2(v1)
        migrated_pc = (v2["control_plane"].get("watchdog") or {}).get(
            "plot_commands", []
        )
        assert migrated_pc == original_pc, (
            f"plot_commands diverged in migration for {yaml_path}"
        )

    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_alert_thresholds_preserved(self, yaml_path: Path):
        v1 = yaml.safe_load(yaml_path.read_text())
        original_at = (v1.get("watchdog") or {}).get("alert_thresholds")
        v2 = _migrate_v1_to_v2(v1)
        migrated_at = (v2["control_plane"].get("watchdog") or {}).get(
            "alert_thresholds"
        )
        assert migrated_at == original_at

    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_checkpoint_milestones_preserved(self, yaml_path: Path):
        v1 = yaml.safe_load(yaml_path.read_text())
        original = (v1.get("watchdog") or {}).get("checkpoint_milestones")
        v2 = _migrate_v1_to_v2(v1)
        migrated = (v2["control_plane"].get("watchdog") or {}).get(
            "checkpoint_milestones"
        )
        assert migrated == original


# ---------------------------------------------------------------------------
# Regression: migration output validates against ExperimentManifest for every
# real yaml
# ---------------------------------------------------------------------------


class TestMigrationOutputValidates:
    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_every_real_yaml_migrates_and_validates(self, yaml_path: Path):
        v1 = yaml.safe_load(yaml_path.read_text())
        v2 = _migrate_v1_to_v2(v1)
        # The migrated dict must load through the current model cleanly.
        m = ExperimentManifest.from_dict(v2)
        assert m.schema_version == 2
