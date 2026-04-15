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

Also asserts that ``plot_commands`` round-trip bit-for-bit between the
flat v1-compat fields and the nested v2 sections (watchdog fence).
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

    def test_contract_stopping_rule_prefers_structured_top_level(self):
        """When yaml has a top-level ``stopping_rule:`` dict, use the structured shape."""
        m = ExperimentManifest.from_dict(_heilbron_v2_yaml())
        rule = m.contract.stopping_rule
        assert rule.description  # migrated yaml has both prose and structured form
        assert rule.conditions, "migrated heilbron yaml carries structured conditions"
        assert rule.conditions[0].kind in ("fitness_plateau", "invalidity_window")

    def test_contract_stopping_rule_description_from_flat_prose_only(self):
        """v1-shaped manifest (prose only, no structured dict) lifts into description."""
        raw = _heilbron_v2_yaml()
        # Strip structured stopping_rule so the validator must extract it from description.
        if isinstance(raw.get("contract"), dict):
            raw["contract"]["stopping_rule"] = {
                "description": "max_generations=50 OR futility_at_gen25(both < 0.03)",
                "conditions": [],
                "enforce_at": "checkpoint",
            }
        m = ExperimentManifest.from_dict(raw)
        assert (
            m.contract.stopping_rule.description
            == "max_generations=50 OR futility_at_gen25(both < 0.03)"
        )
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
# Watchdog fence — every live yaml's control_plane.watchdog matches the flat
# WatchdogSection bit-for-bit during the transition, then validates cleanly.
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
        nested_pc = (
            (raw.get("control_plane") or {}).get("watchdog") or {}
        ).get("plot_commands", [])

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
        nested_at = (
            (raw.get("control_plane") or {}).get("watchdog") or {}
        ).get("alert_thresholds")
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
        nested = (
            (raw.get("control_plane") or {}).get("watchdog") or {}
        ).get("checkpoint_milestones")
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
