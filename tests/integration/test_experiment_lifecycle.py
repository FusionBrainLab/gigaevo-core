"""Integration tests for experiment state machine lifecycle.

Tests the full `preregistered → implemented → running → complete` transition
sequence using real YAML files on disk + `fakeredis` for Redis state. This catches
regressions in the refactored manifest/lock code where intermediate state-transition
steps may break.

All fixtures use the v2 canonical nested shape (contract/lifecycle/telemetry/control_plane).
"""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import pytest
import yaml

from gigaevo.experiment.manifest import (
    recover_status,
    set_status,
    update_manifest,
)


def _minimal_manifest_dict(
    *,
    name: str = "hover/test-exp",
    task: str = "hover",
    status: str = "preregistered",
) -> dict:
    """Minimal valid v2 manifest (nested shape) for initial preregistered state."""
    return {
        "schema_version": 2,
        "contract": {
            "identity": {
                "name": name,
                "task": task,
                "branch": "exp/hover/test-exp",
            },
            "problem": {
                "has_test_set": True,
                "fitness_type": "discrete",
                "metric_name": "accuracy",
            },
            "max_generations": 25,
            "runs": [],
            "servers": [],
            "config": {"extra": {}},
        },
        "lifecycle": {"status": status},
    }


def _run_entry(*, pid: int | None = None) -> dict:
    run = {
        "label": "R1",
        "db": 5,
        "prefix": "r1",
        "pipeline": "standard",
        "problem_name": "hover",
        "condition": "control",
        "model_name": "gpt-4",
    }
    if pid is not None:
        run["pid"] = pid
    return run


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

        data = _minimal_manifest_dict()
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=fake_redis),
        ):
            # Step 1: populate required fields, then preregistered → implemented.
            def make_implementable(raw):
                raw["contract"]["runs"] = [_run_entry()]
                raw["contract"]["servers"] = ["server1"]
                raw["contract"]["config"] = {"extra": {"key": "value"}}
                raw.setdefault("lifecycle", {})["smoke_test"] = {"completed": True}

            manifest = update_manifest("hover/test-exp", make_implementable)
            assert manifest.lifecycle.status == "preregistered"

            manifest = set_status("hover/test-exp", "implemented")
            assert manifest.lifecycle.status == "implemented"

            reloaded = yaml.safe_load(yaml_path.read_text())
            assert reloaded["lifecycle"]["status"] == "implemented"

            # Step 2: implemented → running. Launch info under lifecycle.launch;
            # PIDs under contract.runs[].pid.
            def add_launch_info(raw):
                raw.setdefault("lifecycle", {})["launch"] = {
                    "time": "2026-04-14T10:00:00Z",
                    "commit": "abc123def",
                }
                raw["contract"]["runs"][0]["pid"] = 12345

            manifest = update_manifest("hover/test-exp", add_launch_info)
            assert manifest.lifecycle.status == "implemented"

            manifest = set_status("hover/test-exp", "running")
            assert manifest.lifecycle.status == "running"

            reloaded = yaml.safe_load(yaml_path.read_text())
            assert reloaded["lifecycle"]["status"] == "running"
            assert reloaded["contract"]["runs"][0]["pid"] == 12345

            # Step 3: running → complete.
            manifest = set_status("hover/test-exp", "complete")
            assert manifest.lifecycle.status == "complete"

            reloaded = yaml.safe_load(yaml_path.read_text())
            assert reloaded["lifecycle"]["status"] == "complete"


class TestInvalidTransitions:
    def test_complete_is_terminal(self, tmp_path, fake_redis):
        """Complete status is terminal — no further transitions allowed."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict(status="complete")
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=fake_redis),
        ):
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status("hover/test-exp", "running")

    def test_preregistered_cannot_go_to_running_directly(self, tmp_path, fake_redis):
        """Preregistered must transition to implemented first, then running."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict(status="preregistered")
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=fake_redis),
        ):
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status("hover/test-exp", "running")


class TestRecovery:
    def test_running_to_implemented_allowed_via_recover_status(
        self, tmp_path, fake_redis
    ):
        """Running → implemented recovery transition via recover_status."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict(status="running")
        data["contract"]["runs"] = [_run_entry(pid=12345)]
        data["contract"]["servers"] = ["server1"]
        data["contract"]["config"] = {"extra": {"key": "value"}}
        data["lifecycle"]["smoke_test"] = {"completed": True}
        data["lifecycle"]["launch"] = {
            "time": "2026-04-14T10:00:00Z",
            "commit": "abc123",
        }
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=fake_redis),
        ):
            manifest = recover_status("hover/test-exp", "implemented")
            assert manifest.lifecycle.status == "implemented"
            assert manifest.contract.runs[0].pid == 12345


class TestAtomicWrites:
    def test_lock_is_released_after_set_status(self, tmp_path, fake_redis):
        """After set_status completes, the Redis lock should be released."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        data["contract"]["runs"] = [_run_entry()]
        data["contract"]["servers"] = ["server1"]
        data["contract"]["config"] = {"extra": {"key": "value"}}
        data["lifecycle"]["smoke_test"] = {"completed": True}
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=fake_redis),
        ):
            set_status("hover/test-exp", "implemented")

            lock_key = "experiments:hover/test-exp:yaml_lock"
            assert fake_redis.get(lock_key) is None

    def test_tmp_file_cleaned_up_after_write(self, tmp_path, fake_redis):
        """After write, no .tmp file should remain."""
        exp_dir = tmp_path / "experiments" / "hover" / "test-exp"
        exp_dir.mkdir(parents=True)
        yaml_path = exp_dir / "experiment.yaml"

        data = _minimal_manifest_dict()
        data["contract"]["runs"] = [_run_entry()]
        data["contract"]["servers"] = ["server1"]
        data["contract"]["config"] = {"extra": {"key": "value"}}
        data["lifecycle"]["smoke_test"] = {"completed": True}
        yaml_path.write_text(yaml.safe_dump(data))

        with (
            patch("gigaevo.experiment.manifest.PROJ", tmp_path),
            patch("gigaevo.experiment.manifest.get_redis", return_value=fake_redis),
        ):
            set_status("hover/test-exp", "implemented")

            tmp_file = yaml_path.with_suffix(".yaml.tmp")
            assert not tmp_file.exists()
