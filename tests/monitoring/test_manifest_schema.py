"""Tests for the Pydantic v2 ExperimentManifest schema.

Tests cover:
- Valid manifest loading at every status level
- Validation error messages with field paths and actionable suggestions
- Status-gated required fields
- JSON Schema export
- YAML loading (string and file)
- Round-trip serialization
- Integration with real experiment.yaml files
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gigaevo.monitoring.manifest_schema import (
    VALID_STATUSES,
    AlertThresholds,
    ExperimentManifest,
    PlotCommand,
    WatchdogSection,
    export_json_schema,
)

# ---------------------------------------------------------------------------
# Repo root for integration tests
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _minimal_preregistered() -> dict:
    """Minimal valid manifest at status=preregistered."""
    return {
        "schema_version": 1,
        "experiment": {
            "name": "hover/test",
            "task": "hover",
            "status": "preregistered",
            "branch": "exp/hover/test",
            "max_generations": 50,
        },
        "problem": {"has_test_set": True, "fitness_type": "discrete"},
    }


def _minimal_implemented() -> dict:
    """Minimal valid manifest at status=implemented."""
    raw = _minimal_preregistered()
    raw["experiment"]["status"] = "implemented"
    raw["runs"] = [
        {
            "label": "R1",
            "db": 1,
            "prefix": "chains/hover/test",
            "pipeline": "standard",
            "problem_name": "chains/hover/test",
            "condition": "treatment",
            "mutation_url": "http://localhost:4000/v1",
            "model_name": "test-model",
        }
    ]
    raw["servers"] = ["10.0.0.1"]
    raw["config"] = {"stage_timeout": 120}
    raw["smoke_test"] = {"completed": True}
    return raw


def _minimal_running() -> dict:
    """Minimal valid manifest at status=running."""
    raw = _minimal_implemented()
    raw["experiment"]["status"] = "running"
    raw["launch"] = {
        "time": "2026-01-01T00:00:00Z",
        "commit": "abc123",
    }
    raw["runs"][0]["pid"] = 12345
    return raw


def _minimal_complete() -> dict:
    """Minimal valid manifest at status=complete."""
    raw = _minimal_running()
    raw["experiment"]["status"] = "complete"
    return raw


# ---------------------------------------------------------------------------
# 1. Valid manifest loading tests
# ---------------------------------------------------------------------------


class TestValidManifestLoading:
    def test_load_preregistered_manifest(self) -> None:
        raw = _minimal_preregistered()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.experiment.name == "hover/test"
        assert manifest.experiment.status == "preregistered"
        assert manifest.experiment.max_generations == 50

    def test_load_implemented_manifest(self) -> None:
        raw = _minimal_implemented()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.experiment.status == "implemented"
        assert len(manifest.runs) == 1
        assert manifest.runs[0].label == "R1"
        assert manifest.servers == ["10.0.0.1"]
        assert manifest.smoke_test.completed is True

    def test_load_running_manifest(self) -> None:
        raw = _minimal_running()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.experiment.status == "running"
        assert manifest.launch.time == "2026-01-01T00:00:00Z"
        assert manifest.launch.commit == "abc123"
        assert manifest.runs[0].pid == 12345

    def test_load_complete_manifest(self) -> None:
        raw = _minimal_complete()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.experiment.status == "complete"

    def test_load_invalid_status_manifest(self) -> None:
        """Status 'invalid' is a valid status (used for abandoned experiments)."""
        raw = _minimal_preregistered()
        raw["experiment"]["status"] = "invalid"
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.experiment.status == "invalid"

    def test_defaults_for_optional_sections(self) -> None:
        """Omitted sections get sensible defaults."""
        raw = _minimal_preregistered()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.runs == []
        assert manifest.servers == []
        assert manifest.config == {}
        assert manifest.launch.time is None
        assert manifest.smoke_test.completed is False
        assert manifest.baseline.reference is None


# ---------------------------------------------------------------------------
# 2. Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_missing_experiment_name(self) -> None:
        raw = _minimal_preregistered()
        del raw["experiment"]["name"]
        with pytest.raises(Exception):
            ExperimentManifest.from_dict(raw)

    def test_missing_experiment_task(self) -> None:
        raw = _minimal_preregistered()
        del raw["experiment"]["task"]
        with pytest.raises(Exception):
            ExperimentManifest.from_dict(raw)

    def test_invalid_status(self) -> None:
        raw = _minimal_preregistered()
        raw["experiment"]["status"] = "bogus"
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        error_str = str(exc_info.value)
        assert "bogus" in error_str

    def test_unsupported_schema_version(self) -> None:
        raw = _minimal_preregistered()
        raw["schema_version"] = 99
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        error_str = str(exc_info.value)
        assert "99" in error_str

    def test_implemented_without_runs(self) -> None:
        raw = _minimal_implemented()
        raw["runs"] = []
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        assert "runs" in str(exc_info.value).lower()

    def test_implemented_without_servers(self) -> None:
        raw = _minimal_implemented()
        raw["servers"] = []
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        assert "servers" in str(exc_info.value).lower()

    def test_implemented_without_smoke_test(self) -> None:
        raw = _minimal_implemented()
        raw["smoke_test"] = {"completed": False}
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        assert "smoke_test" in str(exc_info.value).lower()

    def test_running_without_launch_time(self) -> None:
        raw = _minimal_running()
        raw["launch"]["time"] = None
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        assert "launch.time" in str(exc_info.value)

    def test_running_without_pids(self) -> None:
        raw = _minimal_running()
        raw["runs"][0]["pid"] = None
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        error_str = str(exc_info.value)
        assert "R1" in error_str or "pid" in error_str.lower()

    def test_non_numeric_db(self) -> None:
        raw = _minimal_implemented()
        raw["runs"][0]["db"] = "abc"
        with pytest.raises(Exception):
            ExperimentManifest.from_dict(raw)

    def test_negative_max_generations(self) -> None:
        raw = _minimal_preregistered()
        raw["experiment"]["max_generations"] = -1
        with pytest.raises(Exception):
            ExperimentManifest.from_dict(raw)


# ---------------------------------------------------------------------------
# 3. Actionable error message tests
# ---------------------------------------------------------------------------


class TestActionableErrors:
    def test_missing_name_mentions_field_path(self) -> None:
        raw = _minimal_preregistered()
        del raw["experiment"]["name"]
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        error_str = str(exc_info.value)
        assert "name" in error_str.lower()

    def test_invalid_status_lists_valid_values(self) -> None:
        raw = _minimal_preregistered()
        raw["experiment"]["status"] = "bogus"
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        error_str = str(exc_info.value)
        # Should mention at least some valid statuses
        assert "preregistered" in error_str or "Valid" in error_str

    def test_implemented_without_config_mentions_status(self) -> None:
        raw = _minimal_implemented()
        raw["config"] = {}
        with pytest.raises(Exception) as exc_info:
            ExperimentManifest.from_dict(raw)
        error_str = str(exc_info.value)
        assert "implemented" in error_str.lower()


# ---------------------------------------------------------------------------
# 4. JSON Schema export tests
# ---------------------------------------------------------------------------


class TestJsonSchema:
    def test_model_json_schema_returns_dict(self) -> None:
        schema = ExperimentManifest.model_json_schema()
        assert isinstance(schema, dict)

    def test_schema_has_type_object(self) -> None:
        schema = ExperimentManifest.model_json_schema()
        assert schema.get("type") == "object"

    def test_schema_required_fields(self) -> None:
        schema = ExperimentManifest.model_json_schema()
        required = schema.get("required", [])
        assert "schema_version" in required
        assert "experiment" in required

    def test_schema_has_properties(self) -> None:
        schema = ExperimentManifest.model_json_schema()
        assert "properties" in schema
        assert "schema_version" in schema["properties"]
        assert "experiment" in schema["properties"]

    def test_export_json_schema_writes_file(self, tmp_path: Path) -> None:
        output = tmp_path / "schema.json"
        export_json_schema(output)
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert isinstance(loaded, dict)
        assert "properties" in loaded

    def test_export_json_schema_roundtrip(self, tmp_path: Path) -> None:
        """Exported file matches model_json_schema() output."""
        output = tmp_path / "schema.json"
        export_json_schema(output)
        loaded = json.loads(output.read_text())
        expected = ExperimentManifest.model_json_schema()
        assert loaded == expected


# ---------------------------------------------------------------------------
# 5. YAML loading tests
# ---------------------------------------------------------------------------


class TestYamlLoading:
    def test_from_yaml_string(self) -> None:
        raw = _minimal_preregistered()
        yaml_str = yaml.safe_dump(raw)
        manifest = ExperimentManifest.from_yaml(yaml_str)
        assert manifest.experiment.name == "hover/test"

    def test_from_yaml_file(self, tmp_path: Path) -> None:
        raw = _minimal_preregistered()
        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text(yaml.safe_dump(raw))
        manifest = ExperimentManifest.from_yaml_file(yaml_path)
        assert manifest.experiment.name == "hover/test"

    def test_invalid_yaml_raises_value_error(self) -> None:
        bad_yaml = ":\n  :\n  - [invalid"
        with pytest.raises(ValueError, match="YAML"):
            ExperimentManifest.from_yaml(bad_yaml)

    def test_yaml_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            ExperimentManifest.from_yaml_file(missing)

    def test_yaml_file_error_includes_path(self, tmp_path: Path) -> None:
        """Error from invalid file content mentions the file path."""
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(
            "schema_version: 99\nexperiment:\n  name: x\n  task: y\n  status: preregistered\n"
        )
        with pytest.raises(ValueError) as exc_info:
            ExperimentManifest.from_yaml_file(bad_path)
        assert str(bad_path) in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_dict_and_reload(self) -> None:
        raw = _minimal_running()
        manifest = ExperimentManifest.from_dict(raw)
        exported = manifest.to_dict()
        reloaded = ExperimentManifest.from_dict(exported)
        assert reloaded.experiment.name == manifest.experiment.name
        assert reloaded.experiment.status == manifest.experiment.status
        assert len(reloaded.runs) == len(manifest.runs)

    def test_yaml_roundtrip(self) -> None:
        raw = _minimal_running()
        manifest = ExperimentManifest.from_dict(raw)
        yaml_str = yaml.safe_dump(manifest.to_dict())
        reloaded = ExperimentManifest.from_yaml(yaml_str)
        assert reloaded.experiment.name == manifest.experiment.name

    def test_extra_fields_ignored(self) -> None:
        """Unknown fields in YAML are silently ignored (not 'forbid')."""
        raw = _minimal_preregistered()
        raw["unknown_section"] = {"foo": "bar"}
        raw["experiment"]["unknown_field"] = "baz"
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.experiment.name == "hover/test"


# ---------------------------------------------------------------------------
# 7. Manifest-optional tests (RunSpec works independently)
# ---------------------------------------------------------------------------


class TestManifestOptional:
    def test_run_spec_parse_works_independently(self) -> None:
        """RunSpec.parse() from the monitoring package works without manifest."""
        from gigaevo.monitoring.run_spec import RunSpec

        spec = RunSpec.parse("chains/hover/test@4:R1")
        assert spec.prefix == "chains/hover/test"
        assert spec.db == 4
        assert spec.label == "R1"

    def test_collect_snapshot_works_without_manifest(self) -> None:
        """collect_snapshot depends on RunSpec, not ExperimentManifest."""
        # Just verify the import works -- actual Redis calls are in other tests
        from gigaevo.monitoring.redis_queries import collect_snapshot

        assert callable(collect_snapshot)


# ---------------------------------------------------------------------------
# 8. Integration: real experiment.yaml files from the repo
# ---------------------------------------------------------------------------


def _discover_experiment_yamls() -> list[Path]:
    """Find all experiment.yaml files in the repo (excluding _template)."""
    return [
        p
        for p in REPO_ROOT.glob("experiments/*/*/experiment.yaml")
        if "_template" not in str(p)
    ]


class TestRealManifests:
    def test_load_real_heilbron_manifest(self) -> None:
        """The heilbron/adversarial-dynamic-updates manifest loads without errors."""
        path = (
            REPO_ROOT
            / "experiments"
            / "heilbron"
            / "adversarial-dynamic-updates"
            / "experiment.yaml"
        )
        if not path.exists():
            pytest.skip("heilbron manifest not found")
        manifest = ExperimentManifest.from_yaml_file(path)
        assert "heilbron" in manifest.experiment.name
        assert manifest.experiment.status in VALID_STATUSES
        assert len(manifest.runs) > 0

    def test_load_all_existing_manifests(self) -> None:
        """All experiment.yaml files in the repo load through the Pydantic schema."""
        yamls = _discover_experiment_yamls()
        if not yamls:
            pytest.skip("No experiment.yaml files found in repo")

        failures: list[tuple[Path, str]] = []
        for yaml_path in yamls:
            try:
                ExperimentManifest.from_yaml_file(yaml_path)
            except Exception as exc:
                failures.append((yaml_path, str(exc)))

        if failures:
            msg_parts = [f"  {p}: {e}" for p, e in failures]
            pytest.fail(
                f"{len(failures)}/{len(yamls)} experiment.yaml files failed:\n"
                + "\n".join(msg_parts)
            )

    def test_json_schema_validates_real_manifest(self, tmp_path: Path) -> None:
        """JSON Schema is consistent with Pydantic model on a real manifest."""
        yamls = _discover_experiment_yamls()
        if not yamls:
            pytest.skip("No experiment.yaml files found")

        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        schema = ExperimentManifest.model_json_schema()
        path = yamls[0]
        with open(path) as f:
            raw = yaml.safe_load(f)
        jsonschema.validate(instance=raw, schema=schema)


# ---------------------------------------------------------------------------
# 9. WatchdogSection tests
# ---------------------------------------------------------------------------


class TestWatchdogSection:
    def test_watchdog_section_defaults(self) -> None:
        section = WatchdogSection()
        assert section.plugin is None
        assert section.plot_commands == []
        assert section.plot_metrics == []
        assert section.alert_thresholds.invalidity_rate == 0.75
        assert section.poll_interval_s == 3600
        assert section.plot_retries == 3
        assert section.plot_retry_delay_s == 30
        assert section.rolling_comment_threshold_hours == 24
        assert section.checkpoint_milestones == [0.1, 0.2, 0.5, 1.0]
        assert section.no_proxy_hosts == []

    def test_watchdog_section_with_plot_commands(self) -> None:
        section = WatchdogSection(
            plot_commands=[
                PlotCommand(command="arms-race", args={"metric": "fitness"}),
                PlotCommand(command="comparison", output_name="cmp", caption="Compare"),
            ]
        )
        assert len(section.plot_commands) == 2
        assert section.plot_commands[0].command == "arms-race"
        assert section.plot_commands[0].args == {"metric": "fitness"}
        assert section.plot_commands[1].output_name == "cmp"

    def test_watchdog_section_custom_alert_thresholds(self) -> None:
        section = WatchdogSection(
            alert_thresholds=AlertThresholds(
                invalidity_rate=0.5,
                stagnation_window=20,
                generation_gap_threshold=3,
            )
        )
        assert section.alert_thresholds.invalidity_rate == 0.5
        assert section.alert_thresholds.stagnation_window == 20
        assert section.alert_thresholds.generation_gap_threshold == 3

    def test_watchdog_section_in_manifest(self) -> None:
        raw = _minimal_preregistered()
        raw["watchdog"] = {
            "plugin": "heilbron",
            "plot_metrics": ["fitness", "prompt_length"],
            "checkpoint_milestones": [0.25, 0.5, 1.0],
        }
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plugin == "heilbron"
        assert manifest.watchdog.plot_metrics == ["fitness", "prompt_length"]
        assert manifest.watchdog.checkpoint_milestones == [0.25, 0.5, 1.0]

    def test_manifest_default_watchdog_section(self) -> None:
        raw = _minimal_preregistered()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plugin is None
        assert manifest.watchdog.plot_commands == []

    def test_extra_fields_ignored_in_watchdog(self) -> None:
        raw = _minimal_preregistered()
        raw["watchdog"] = {"plugin": "solo", "unknown_field": "ignored"}
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plugin == "solo"


class TestPlotCommand:
    def test_plot_command_defaults(self) -> None:
        cmd = PlotCommand(command="comparison")
        assert cmd.command == "comparison"
        assert cmd.args == {}
        assert cmd.output_name == ""
        assert cmd.caption == ""

    def test_plot_command_full(self) -> None:
        cmd = PlotCommand(
            command="arms-race",
            args={"metric": "actual_fitness", "smoothing": "ema", "window": 10},
            output_name="arms_race",
            caption="Arms-race dynamics",
        )
        assert cmd.command == "arms-race"
        assert cmd.args["smoothing"] == "ema"
        assert cmd.output_name == "arms_race"
        assert cmd.caption == "Arms-race dynamics"


class TestAlertThresholdsSchema:
    def test_alert_thresholds_defaults(self) -> None:
        thresholds = AlertThresholds()
        assert thresholds.invalidity_rate == 0.75
        assert thresholds.stagnation_window == 10
        assert thresholds.generation_gap_threshold == 5

    def test_alert_thresholds_custom(self) -> None:
        thresholds = AlertThresholds(
            invalidity_rate=0.9,
            stagnation_window=5,
            generation_gap_threshold=10,
        )
        assert thresholds.invalidity_rate == 0.9
        assert thresholds.stagnation_window == 5


class TestLegacyWatchdogMigration:
    def test_legacy_watchdog_plugin_migrated(self) -> None:
        raw = _minimal_preregistered()
        raw["watchdog_plugin"] = "heilbron"
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plugin == "heilbron"

    def test_legacy_watchdog_plugin_options_migrated(self) -> None:
        raw = _minimal_preregistered()
        raw["watchdog_plugin_options"] = {"plot_metrics": ["fitness", "accuracy"]}
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plot_metrics == ["fitness", "accuracy"]

    def test_explicit_watchdog_section_not_overridden_by_legacy(self) -> None:
        raw = _minimal_preregistered()
        raw["watchdog"] = {"plugin": "adversarial", "plot_metrics": ["loss"]}
        raw["watchdog_plugin"] = "solo"
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plugin == "adversarial"
        assert manifest.watchdog.plot_metrics == ["loss"]

    def test_no_legacy_fields_no_migration(self) -> None:
        raw = _minimal_preregistered()
        manifest = ExperimentManifest.from_dict(raw)
        assert manifest.watchdog.plugin is None
        assert manifest.watchdog.plot_metrics == []
