"""Tests for tools.experiment.manifest — schema validation, state machine, locking."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tools.experiment.manifest import (
    VALID_STATUSES,
    VALID_TRANSITIONS,
    _validate,
    _write_manifest_atomic,
    claim_dbs,
    generate_pr_description,
    load_manifest,
    refresh_db_claims,
    release_db_claims,
    set_status,
    update_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_raw(status: str = "preregistered", **overrides) -> dict:
    """Return a minimal valid experiment.yaml dict."""
    raw = {
        "schema_version": 1,
        "experiment": {
            "name": "test/smoke",
            "task": "test",
            "branch": "exp/smoke",
            "status": status,
            "max_generations": 10,
        },
        "problem": {
            "has_test_set": False,
            "fitness_type": "continuous",
            "metric_name": "score",
        },
        "runs": [],
        "servers": [],
        "config": {},
        "smoke_test": {"completed": False},
    }
    raw.update(overrides)
    return raw


def _implemented_raw(**overrides) -> dict:
    """Return a valid implemented experiment.yaml dict."""
    raw = _minimal_raw(
        status="implemented",
        runs=[
            {
                "label": "A1",
                "db": 99,
                "prefix": "chains/test/smoke",
                "pipeline": "standard",
                "problem_name": "chains/test/smoke",
                "condition": "control",
                "chain_url": "http://10.0.0.1:8001/v1",
                "mutation_url": "http://10.0.0.1:8777/v1",
                "model_name": "test-model",
            }
        ],
        servers=["10.0.0.1"],
        config={"stage_timeout": 300},
        smoke_test={"completed": True, "db": 98, "generations": 3},
    )
    raw.update(overrides)
    return raw


def _running_raw(**overrides) -> dict:
    """Return a valid running experiment.yaml dict."""
    raw = _implemented_raw(status="running")
    raw["experiment"]["status"] = "running"
    raw["runs"][0]["pid"] = 12345
    raw["launch"] = {
        "time": "2026-01-01T00:00:00Z",
        "commit": "abc123",
        "watchdog_pid": 12346,
        "confirmed_at": "2026-01-01T00:01:00Z",
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def tmp_experiment(tmp_path: Path) -> tuple[str, Path]:
    """Create a temporary experiment directory with a valid manifest."""
    exp_dir = tmp_path / "experiments" / "test" / "smoke"
    exp_dir.mkdir(parents=True)
    yaml_path = exp_dir / "experiment.yaml"

    raw = _minimal_raw()
    with open(yaml_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)

    return "test/smoke", tmp_path


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_minimal_preregistered(self):
        raw = _minimal_raw()
        m = _validate(raw, "test/smoke")
        assert m.status == "preregistered"
        assert m.name == "test/smoke"
        assert m.task == "test"

    def test_invalid_status_rejected(self):
        raw = _minimal_raw(status="bogus")
        raw["experiment"]["status"] = "bogus"
        with pytest.raises(ValueError, match="Invalid status"):
            _validate(raw, "test/smoke")

    def test_unsupported_schema_version(self):
        raw = _minimal_raw()
        raw["schema_version"] = 99
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            _validate(raw, "test/smoke")

    def test_implemented_requires_runs(self):
        raw = _minimal_raw(status="implemented")
        raw["experiment"]["status"] = "implemented"
        raw["smoke_test"] = {"completed": True}
        # No runs
        with pytest.raises(ValueError, match="runs.*must be non-empty"):
            _validate(raw, "test/smoke")

    def test_implemented_requires_servers(self):
        raw = _implemented_raw()
        raw["servers"] = []
        with pytest.raises(ValueError, match="servers.*must be non-empty"):
            _validate(raw, "test/smoke")

    def test_implemented_requires_smoke_test(self):
        raw = _implemented_raw()
        raw["smoke_test"] = {"completed": False}
        with pytest.raises(ValueError, match="smoke_test.completed"):
            _validate(raw, "test/smoke")

    def test_running_requires_pids(self):
        raw = _running_raw()
        raw["runs"][0]["pid"] = None
        with pytest.raises(ValueError, match="pid is required"):
            _validate(raw, "test/smoke")

    def test_running_requires_launch_time(self):
        raw = _running_raw()
        raw["launch"]["time"] = None
        with pytest.raises(ValueError, match="launch.time is required"):
            _validate(raw, "test/smoke")

    def test_valid_implemented(self):
        raw = _implemented_raw()
        m = _validate(raw, "test/smoke")
        assert m.status == "implemented"
        assert len(m.runs) == 1
        assert m.runs[0].label == "A1"

    def test_valid_running(self):
        raw = _running_raw()
        m = _validate(raw, "test/smoke")
        assert m.status == "running"
        assert m.runs[0].pid == 12345
        assert m.launch.time == "2026-01-01T00:00:00Z"

    def test_problem_has_test_set_false(self):
        raw = _minimal_raw()
        raw["problem"]["has_test_set"] = False
        m = _validate(raw, "test/smoke")
        assert not m.problem.has_test_set

    def test_run_spec_fields(self):
        raw = _implemented_raw()
        m = _validate(raw, "test/smoke")
        run = m.runs[0]
        assert run.db == 99
        assert run.pipeline == "standard"
        assert run.chain_url == "http://10.0.0.1:8001/v1"

    def test_custom_env_parsed(self):
        raw = _implemented_raw()
        raw["custom_env"] = {"MY_VAR": "hello"}
        m = _validate(raw, "test/smoke")
        assert m.custom_env == {"MY_VAR": "hello"}


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestTransitions:
    def test_all_valid_transitions(self):
        """Every status in VALID_TRANSITIONS maps to valid statuses."""
        for source, targets in VALID_TRANSITIONS.items():
            assert source in VALID_STATUSES
            for t in targets:
                assert t in VALID_STATUSES

    def test_preregistered_to_implemented(self):
        assert "implemented" in VALID_TRANSITIONS["preregistered"]

    def test_implemented_to_running(self):
        assert "running" in VALID_TRANSITIONS["implemented"]

    def test_running_to_complete(self):
        assert "complete" in VALID_TRANSITIONS["running"]

    def test_running_to_invalid(self):
        assert "invalid" in VALID_TRANSITIONS["running"]

    def test_complete_is_terminal(self):
        assert VALID_TRANSITIONS["complete"] == set()

    def test_invalid_to_preregistered(self):
        assert "preregistered" in VALID_TRANSITIONS["invalid"]

    def test_no_backward_in_normal_mode(self):
        assert "preregistered" not in VALID_TRANSITIONS["implemented"]
        assert "implemented" not in VALID_TRANSITIONS["running"]


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_write_then_rename(self, tmp_path: Path):
        target = tmp_path / "test.yaml"
        data = {"key": "value", "nested": {"a": 1}}
        _write_manifest_atomic(target, data)

        assert target.exists()
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_tmp_cleaned_up(self, tmp_path: Path):
        target = tmp_path / "test.yaml"
        _write_manifest_atomic(target, {"x": 1})
        tmp = target.with_suffix(".yaml.tmp")
        assert not tmp.exists()

    def test_overwrites_existing(self, tmp_path: Path):
        target = tmp_path / "test.yaml"
        _write_manifest_atomic(target, {"version": 1})
        _write_manifest_atomic(target, {"version": 2})
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert loaded["version"] == 2


# ---------------------------------------------------------------------------
# Load from disk (with mocked PROJ)
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_load_valid(self, tmp_experiment):
        exp_name, tmp_root = tmp_experiment
        with patch("tools.experiment.manifest.PROJ", tmp_root):
            m = load_manifest(exp_name)
        assert m.name == "test/smoke"
        assert m.status == "preregistered"

    def test_load_missing_file(self, tmp_path):
        with patch("tools.experiment.manifest.PROJ", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_manifest("nonexistent/exp")

    def test_load_invalid_yaml(self, tmp_experiment):
        exp_name, tmp_root = tmp_experiment
        path = tmp_root / "experiments" / "test" / "smoke" / "experiment.yaml"
        path.write_text("{{invalid yaml: [")
        with patch("tools.experiment.manifest.PROJ", tmp_root):
            with pytest.raises(yaml.YAMLError):
                load_manifest(exp_name)


# ---------------------------------------------------------------------------
# set_status and update_manifest (with mocked Redis)
# ---------------------------------------------------------------------------


class TestSetStatus:
    def _setup_experiment(self, tmp_path: Path, raw: dict) -> tuple[str, Path]:
        exp_dir = tmp_path / "experiments" / "test" / "smoke"
        exp_dir.mkdir(parents=True)
        path = exp_dir / "experiment.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        return "test/smoke", tmp_path

    def _mock_redis(self):
        mock_r = MagicMock()
        mock_r.set.return_value = True  # lock acquired
        mock_r.get.return_value = None
        return mock_r

    def test_valid_transition(self, tmp_path: Path):
        raw = _implemented_raw()
        exp_name, root = self._setup_experiment(tmp_path, raw)
        mock_r = self._mock_redis()

        with (
            patch("tools.experiment.manifest.PROJ", root),
            patch("tools.experiment.manifest._get_redis", return_value=mock_r),
        ):
            # Need to add running requirements
            raw["runs"][0]["pid"] = 99999
            raw["launch"] = {"time": "2026-01-01", "commit": "abc"}
            path = root / "experiments" / "test" / "smoke" / "experiment.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(raw, f, sort_keys=False)

            m = set_status(exp_name, "running")
            assert m.status == "running"

    def test_invalid_transition_rejected(self, tmp_path: Path):
        raw = _minimal_raw(status="preregistered")
        exp_name, root = self._setup_experiment(tmp_path, raw)
        mock_r = self._mock_redis()

        with (
            patch("tools.experiment.manifest.PROJ", root),
            patch("tools.experiment.manifest._get_redis", return_value=mock_r),
        ):
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status(exp_name, "running")

    def test_recovery_transition(self, tmp_path: Path):
        raw = _running_raw()
        exp_name, root = self._setup_experiment(tmp_path, raw)
        mock_r = self._mock_redis()

        with (
            patch("tools.experiment.manifest.PROJ", root),
            patch("tools.experiment.manifest._get_redis", return_value=mock_r),
        ):
            # Normal transition: running -> implemented is NOT allowed
            with pytest.raises(ValueError, match="Invalid transition"):
                set_status(exp_name, "implemented")

            # Recovery transition: allowed with flag
            # But we need to re-write the file since validation changes status
            with open(root / "experiments/test/smoke/experiment.yaml", "w") as f:
                yaml.safe_dump(raw, f, sort_keys=False)

            m = set_status(exp_name, "implemented", allow_recovery=True)
            assert m.status == "implemented"


class TestUpdateManifest:
    def test_update_in_place(self, tmp_path: Path):
        raw = _minimal_raw()
        exp_dir = tmp_path / "experiments" / "test" / "smoke"
        exp_dir.mkdir(parents=True)
        path = exp_dir / "experiment.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

        mock_r = MagicMock()
        mock_r.set.return_value = True

        def add_tracking_issue(raw):
            raw["experiment"]["tracking_issue"] = 42

        with (
            patch("tools.experiment.manifest.PROJ", tmp_path),
            patch("tools.experiment.manifest._get_redis", return_value=mock_r),
        ):
            m = update_manifest("test/smoke", add_tracking_issue)
            assert m.tracking_issue == 42

        # Verify persisted
        with open(path) as f:
            saved = yaml.safe_load(f)
        assert saved["experiment"]["tracking_issue"] == 42


# ---------------------------------------------------------------------------
# DB claims (mocked Redis)
# ---------------------------------------------------------------------------


class TestDBClaims:
    def test_claim_success(self):
        mock_r = MagicMock()
        mock_r.set.return_value = True
        with patch("tools.experiment.manifest._get_redis", return_value=mock_r):
            failed = claim_dbs("test/exp", [9, 10])
        assert failed == []
        assert mock_r.set.call_count == 2

    def test_claim_collision(self):
        mock_r = MagicMock()
        # First call succeeds, second fails
        mock_r.set.side_effect = [True, False]
        mock_r.get.return_value = b"other/experiment"
        with patch("tools.experiment.manifest._get_redis", return_value=mock_r):
            failed = claim_dbs("test/exp", [9, 10])
        assert len(failed) == 1
        assert failed[0] == (10, "other/experiment")

    def test_claim_idempotent_same_owner(self):
        mock_r = MagicMock()
        mock_r.set.return_value = False  # already claimed
        mock_r.get.return_value = b"test/exp"  # by us
        with patch("tools.experiment.manifest._get_redis", return_value=mock_r):
            failed = claim_dbs("test/exp", [9])
        assert failed == []  # not a failure if we own it

    def test_refresh_uses_xx(self):
        mock_r = MagicMock()
        with patch("tools.experiment.manifest._get_redis", return_value=mock_r):
            refresh_db_claims("test/exp", [9, 10])
        for call in mock_r.set.call_args_list:
            assert call.kwargs.get("xx") is True

    def test_release(self):
        mock_r = MagicMock()
        with patch("tools.experiment.manifest._get_redis", return_value=mock_r):
            release_db_claims([9, 10])
        assert mock_r.delete.call_count == 2


# ---------------------------------------------------------------------------
# PR description generation
# ---------------------------------------------------------------------------


class TestGeneratePRDescription:
    def test_preregistered(self, tmp_path: Path):
        raw = _minimal_raw()
        exp_dir = tmp_path / "experiments" / "test" / "smoke"
        exp_dir.mkdir(parents=True)
        path = exp_dir / "experiment.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

        with patch("tools.experiment.manifest.PROJ", tmp_path):
            desc = generate_pr_description("test/smoke")

        assert "Pre-registered" in desc
        assert "test/smoke" in desc

    def test_running_with_checkpoints(self, tmp_path: Path):
        raw = _running_raw()
        raw["checkpoints"] = [{"gen": 5, "timestamp": "2026-01-02", "notes": "ok"}]
        exp_dir = tmp_path / "experiments" / "test" / "smoke"
        exp_dir.mkdir(parents=True)
        path = exp_dir / "experiment.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

        with patch("tools.experiment.manifest.PROJ", tmp_path):
            desc = generate_pr_description("test/smoke")

        assert "Running (gen 5/10)" in desc
        assert "A1" in desc  # run label in table
        assert "12345" in desc  # PID in table

    def test_includes_baseline(self, tmp_path: Path):
        raw = _minimal_raw()
        raw["baseline"] = {"reference": "test/base", "mean": 0.5, "metric": "score"}
        exp_dir = tmp_path / "experiments" / "test" / "smoke"
        exp_dir.mkdir(parents=True)
        path = exp_dir / "experiment.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

        with patch("tools.experiment.manifest.PROJ", tmp_path):
            desc = generate_pr_description("test/smoke")

        assert "test/base" in desc
        assert "0.5" in desc
