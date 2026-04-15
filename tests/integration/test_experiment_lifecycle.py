"""Integration tests for experiment state machine lifecycle.

Tests the full `preregistered → implemented → running → complete` transition
sequence using real YAML files on disk + `fakeredis` for Redis state. This catches
regressions in the refactored manifest/lock code where intermediate state-transition
steps may break.
"""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import pytest
import yaml

from gigaevo.experiment.manifest import (
    set_status,
    update_manifest,
)


def _minimal_manifest_dict(
    *,
    name: str = "hover/test-exp",
    task: str = "hover",
    status: str = "preregistered",
) -> dict:
    """Minimal valid manifest for initial preregistered state."""
    return {
        "schema_version": 1,
        "experiment": {
            "name": name,
            "task": task,
            "status": status,
            "branch": "exp/hover/test-exp",
            "max_generations": 25,
        },
        "problem": {
            "has_test_set": True,
            "fitness_type": "discrete",
            "metric_name": "accuracy",
        },
        "runs": [],
        "servers": [],
        "config": {},
    }


@pytest.fixture
def fake_redis():
    """Provide a fakeredis instance for each test."""
    return fakeredis.FakeRedis()


class TestHappyPath:
    def test_preregistered_to_complete_full_sequence(self, tmp_path, fake_redis):
        """Test the full sequence: preregistered → implemented → running → complete."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        # Seed with minimal preregistered manifest
        data = _minimal_manifest_dict()
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest._get_redis", return_value=fake_redis),
        ):
            # Step 1: preregistered → implemented
            # First populate required fields
            def make_implementable(raw):
                raw["runs"] = [
                    {
                        "label": "R1",
                        "db": 5,
                        "prefix": "r1",
                        "pipeline": "standard",
                        "problem_name": "hover",
                        "condition": "control",
                        "model_name": "gpt-4",
                    }
                ]
                raw["servers"] = ["server1"]
                raw["config"] = {"key": "value"}
                raw["smoke_test"] = {"completed": True}

            manifest = update_manifest("hover/test-exp", make_implementable)
            assert manifest.experiment.status == "preregistered"

            # Now transition to implemented
            manifest = set_status("hover/test-exp", "implemented")
            assert manifest.experiment.status == "implemented"

            # Verify on disk
            reloaded = yaml.safe_load(yaml_path.read_text())
            assert reloaded["experiment"]["status"] == "implemented"

            # Step 2: implemented → running
            # Add launch info and PIDs
            def add_launch_info(raw):
                raw["launch"] = {
                    "time": "2026-04-14T10:00:00Z",
                    "commit": "abc123def",
                }
                raw["runs"][0]["pid"] = 12345

            manifest = update_manifest("hover/test-exp", add_launch_info)
            assert manifest.experiment.status == "implemented"

            manifest = set_status("hover/test-exp", "running")
            assert manifest.experiment.status == "running"

            reloaded = yaml.safe_load(yaml_path.read_text())
            assert reloaded["experiment"]["status"] == "running"
            assert reloaded["runs"][0]["pid"] == 12345

            # Step 3: running → complete
            manifest = set_status("hover/test-exp", "complete")
            assert manifest.experiment.status == "complete"

            reloaded = yaml.safe_load(yaml_path.read_text())
            assert reloaded["experiment"]["status"] == "complete"


class TestInvalidTransitions:
    def test_complete_is_terminal(self, tmp_path, fake_redis):
        """Complete status is terminal — no further transitions allowed."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        data["status"] = "complete"
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest._get_redis", return_value=fake_redis),
        ):
            # Try to transition from complete to anything → should fail
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status("hover/test-exp", "running")

    def test_preregistered_cannot_go_to_running_directly(self, tmp_path, fake_redis):
        """Preregistered must transition to implemented first, then running."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest._get_redis", return_value=fake_redis),
        ):
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status("hover/test-exp", "running")


class TestRecovery:
    def test_running_to_implemented_allowed_with_allow_recovery_true(self, tmp_path, fake_redis):
        """Running → implemented transition is allowed when allow_recovery=True."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        data["status"] = "running"
        data["runs"] = [
            {
                "label": "R1",
                "db": 5,
                "prefix": "r1",
                "pipeline": "standard",
                "problem_name": "hover",
                "condition": "control",
                "model_name": "gpt-4",
                "pid": 12345,
            }
        ]
        data["servers"] = ["server1"]
        data["config"] = {"key": "value"}
        data["smoke_test"] = {"completed": True}
        data["launch"] = {"time": "2026-04-14T10:00:00Z", "commit": "abc123"}
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest._get_redis", return_value=fake_redis),
        ):
            # With allow_recovery=True, the transition is allowed
            manifest = set_status(
                "hover/test-exp", "implemented", allow_recovery=True
            )
            assert manifest.experiment.status == "implemented"

            # PIDs should still be present
            assert manifest.runs[0].pid == 12345


class TestAtomicWrites:
    def test_lock_is_released_after_set_status(self, tmp_path, fake_redis):
        """After set_status completes, the Redis lock should be released."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        data["runs"] = [
            {
                "label": "R1",
                "db": 5,
                "prefix": "r1",
                "pipeline": "standard",
                "problem_name": "hover",
                "condition": "control",
                "model_name": "gpt-4",
            }
        ]
        data["servers"] = ["server1"]
        data["config"] = {"key": "value"}
        data["smoke_test"] = {"completed": True}
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest._get_redis", return_value=fake_redis),
        ):
            set_status("hover/test-exp", "implemented")

            # Lock key should not exist (or have been deleted)
            lock_key = "experiments:hover/test-exp:yaml_lock"
            assert fake_redis.get(lock_key) is None

    def test_tmp_file_cleaned_up_after_write(self, tmp_path, fake_redis):
        """After write, no .tmp file should remain."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        data["runs"] = [
            {
                "label": "R1",
                "db": 5,
                "prefix": "r1",
                "pipeline": "standard",
                "problem_name": "hover",
                "condition": "control",
                "model_name": "gpt-4",
            }
        ]
        data["servers"] = ["server1"]
        data["config"] = {"key": "value"}
        data["smoke_test"] = {"completed": True}
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest._get_redis", return_value=fake_redis),
        ):
            set_status("hover/test-exp", "implemented")

            # No .tmp file should remain
            tmp_file = yaml_path.with_suffix(".yaml.tmp")
            assert not tmp_file.exists()
