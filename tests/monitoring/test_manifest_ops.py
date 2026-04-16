"""Tests for gigaevo.experiment.manifest — Pydantic manifest operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from gigaevo.experiment.manifest import (
    RECOVERY_TRANSITIONS,
    VALID_TRANSITIONS,
    ExperimentManifest,
    experiment_dir,
    generate_pr_description,
    load_manifest,
    manifest_path,
    recover_status,
    set_status,
    write_manifest_atomic,
)


def _minimal_manifest_dict(
    *,
    name: str = "hover/test-exp",
    task: str = "hover",
    status: str = "preregistered",
) -> dict:
    return {
        "schema_version": 2,
        "contract": {
            "identity": {
                "name": name,
                "task": task,
                "branch": "exp/hover/test-exp",
            },
            "max_generations": 25,
            "problem": {
                "has_test_set": True,
                "fitness_type": "discrete",
                "metric_name": "accuracy",
            },
            "runs": [],
            "servers": [],
            "config": {},
        },
        "lifecycle": {
            "status": status,
        },
    }


def _implementable_manifest_dict(
    *, name: str = "hover/test-exp", task: str = "hover", status: str = "preregistered"
) -> dict:
    d = _minimal_manifest_dict(name=name, task=task, status=status)
    d["contract"]["runs"] = [
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
        }
    ]
    d["contract"]["servers"] = ["server1"]
    d["contract"]["config"] = {"key": "value"}
    d["lifecycle"]["smoke_test"] = {"completed": True}
    return d


def _running_manifest_dict(
    *, name: str = "hover/test-exp", task: str = "hover"
) -> dict:
    d = _implementable_manifest_dict(name=name, task=task, status="running")
    d["contract"]["runs"][0]["pid"] = 12345
    d["lifecycle"]["launch"] = {"time": "2026-04-13T00:00:00", "commit": "abc123"}
    return d


class TestExperimentDir:
    def test_returns_path_under_experiments(self):
        result = experiment_dir("hover/foo")
        assert result.name == "foo"
        assert result.parent.name == "hover"
        assert "experiments" in str(result)

    def test_manifest_path_appends_yaml(self):
        result = manifest_path("hover/foo")
        assert result.name == "experiment.yaml"
        assert result.parent.name == "foo"


class TestLoadManifest:
    def test_load_missing_raises(self, tmp_path):
        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_manifest("nonexistent/exp")

    def test_load_valid_yaml(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"
        data = _minimal_manifest_dict()
        yaml_path.write_text(yaml.safe_dump(data))

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            result = load_manifest("hover/test-exp")

        assert isinstance(result, ExperimentManifest)
        assert result.contract.identity.name == "hover/test-exp"
        assert result.lifecycle.status == "preregistered"

    def test_strict_rejects_unknown_top_level_key(self, tmp_path):
        """strict=True must catch typos that the model would otherwise drop."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        data = _minimal_manifest_dict()
        data["lifeycycle"] = {"status": "preregistered"}  # typo, sibling
        (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(data))

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            # default (non-strict) silently ignores it
            load_manifest("hover/test-exp")
            # strict mode raises with a helpful message
            with pytest.raises(Exception, match="lifeycycle"):
                load_manifest("hover/test-exp", strict=True)

    def test_strict_accepts_canonical_v2_keys(self, tmp_path):
        """strict mode must NOT reject any of the documented v2 sections."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        data = _minimal_manifest_dict()
        data["telemetry"] = {"checkpoints": []}
        data["control_plane"] = {}
        (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(data))

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            result = load_manifest("hover/test-exp", strict=True)

        assert result.contract.identity.name == "hover/test-exp"


class TestSetStatus:
    def test_valid_transition(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"
        data = _implementable_manifest_dict(status="preregistered")
        yaml_path.write_text(yaml.safe_dump(data))

        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=mock_redis),
        ):
            result = set_status("hover/test-exp", "implemented")

        assert result.lifecycle.status == "implemented"

    def test_invalid_transition_raises(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"
        data = _minimal_manifest_dict(status="preregistered")
        yaml_path.write_text(yaml.safe_dump(data))

        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=mock_redis),
        ):
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status("hover/test-exp", "running")

    def test_recovery_transition(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"
        data = _running_manifest_dict()
        yaml_path.write_text(yaml.safe_dump(data))

        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=mock_redis),
        ):
            result = recover_status("hover/test-exp", "implemented")

        assert result.lifecycle.status == "implemented"


class TestWriteManifestAtomic:
    def test_writes_yaml_file(self, tmp_path):
        target = tmp_path / "test.yaml"
        data = {"key": "value", "nested": {"a": 1}}
        write_manifest_atomic(target, data)

        assert target.exists()
        loaded = yaml.safe_load(target.read_text())
        assert loaded["key"] == "value"
        assert loaded["nested"]["a"] == 1

    def test_tmp_file_cleaned_up(self, tmp_path):
        target = tmp_path / "test.yaml"
        write_manifest_atomic(target, {"key": "value"})
        assert not target.with_suffix(".yaml.tmp").exists()


class TestGeneratePrDescription:
    def test_returns_markdown_with_experiment_name(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"
        data = _minimal_manifest_dict()
        yaml_path.write_text(yaml.safe_dump(data))

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            result = generate_pr_description("hover/test-exp")

        assert "hover/test-exp" in result
        assert "## Runs" in result
        assert "## Checkpoints" in result


class TestTransitionConstants:
    def test_valid_transitions_has_all_statuses(self):
        assert "preregistered" in VALID_TRANSITIONS
        assert "implemented" in VALID_TRANSITIONS
        assert "running" in VALID_TRANSITIONS
        assert "complete" in VALID_TRANSITIONS
        assert "invalid" in VALID_TRANSITIONS

    def test_recovery_transitions_allows_running_to_implemented(self):
        assert "implemented" in RECOVERY_TRANSITIONS["running"]
