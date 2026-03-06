"""Tests for gigaevo/problems/context.py -- ProblemContext."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gigaevo.problems.context import ProblemContext
from gigaevo.problems.layout import ProblemLayout as PL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal valid metrics.yaml that passes all strict validation checks.
_VALID_METRICS: dict = {
    "specs": {
        "score": {
            "description": "Primary score",
            "higher_is_better": True,
            "is_primary": True,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
        },
        "is_valid": {
            "description": "Validity flag",
            "higher_is_better": True,
            "is_primary": False,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "significant_change": 1.0,
            "sentinel_value": 0.0,
        },
    }
}


def _make_problem_dir(tmp_path: Path, *, add_context: bool = False) -> Path:
    """Create a minimal valid problem directory under *tmp_path*.

    Returns the problem directory path.
    """
    problem_dir = tmp_path / "my_problem"
    problem_dir.mkdir()

    # task_description.txt
    (problem_dir / PL.TASK_DESCRIPTION).write_text("  Solve the problem.  \n")

    # validate.py
    (problem_dir / PL.VALIDATOR).write_text("def validate(): pass\n")

    # metrics.yaml
    (problem_dir / PL.METRICS_FILE).write_text(yaml.dump(_VALID_METRICS))

    # initial_programs/
    init_dir = problem_dir / PL.INITIAL_PROGRAMS_DIR
    init_dir.mkdir()
    (init_dir / "baseline.py").write_text("def solve(): return 0\n")

    if add_context:
        (problem_dir / PL.CONTEXT_FILE).write_text("context = {}\n")

    return problem_dir


# ---------------------------------------------------------------------------
# TestLoadText
# ---------------------------------------------------------------------------


class TestLoadText:
    def test_returns_stripped_content(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)
        text = pc.load_text(PL.TASK_DESCRIPTION)
        assert text == "Solve the problem."

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="Problem file not found"):
            pc.load_text("nonexistent.txt")

    def test_task_description_property(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)
        assert pc.task_description == "Solve the problem."


# ---------------------------------------------------------------------------
# TestIsContextual
# ---------------------------------------------------------------------------


class TestIsContextual:
    def test_true_when_context_file_exists(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path, add_context=True)
        pc = ProblemContext(problem_dir)
        assert pc.is_contextual is True

    def test_false_when_context_file_missing(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path, add_context=False)
        pc = ProblemContext(problem_dir)
        assert pc.is_contextual is False


# ---------------------------------------------------------------------------
# TestLoadMetricsContext
# ---------------------------------------------------------------------------


class TestLoadMetricsContext:
    def test_valid_metrics_yaml(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)
        ctx = pc.metrics_context
        assert ctx.get_primary_key() == "score"
        assert ctx.get_bounds("score") == (0.0, 100.0)
        assert ctx.get_bounds("is_valid") == (0.0, 1.0)
        assert set(ctx.specs.keys()) == {"score", "is_valid"}
        assert ctx.specs["score"].higher_is_better is True
        assert ctx.specs["is_valid"].is_primary is False

    def test_missing_metrics_file_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).unlink()
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="Missing metrics.yaml"):
            _ = pc.metrics_context

    def test_non_dict_metrics_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text("- just\n- a\n- list\n")
        pc = ProblemContext(problem_dir)
        with pytest.raises(ValueError, match="metrics.yaml must be a mapping"):
            _ = pc.metrics_context

    def test_missing_primary_bounds_raises(self, tmp_path: Path) -> None:
        """Primary metric without lower/upper bounds should raise."""
        metrics = {
            "specs": {
                "score": {
                    "description": "Primary score",
                    "higher_is_better": True,
                    "is_primary": True,
                    # No bounds
                },
                "is_valid": {
                    "description": "Validity flag",
                    "higher_is_better": True,
                    "is_primary": False,
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                    "significant_change": 1.0,
                    "sentinel_value": 0.0,
                },
            }
        }
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text(yaml.dump(metrics))
        pc = ProblemContext(problem_dir)
        with pytest.raises(ValueError, match="must define lower_bound and upper_bound"):
            _ = pc.metrics_context

    def test_missing_is_valid_spec_raises(self, tmp_path: Path) -> None:
        """Metrics YAML without an is_valid spec should raise."""
        metrics = {
            "specs": {
                "score": {
                    "description": "Primary score",
                    "higher_is_better": True,
                    "is_primary": True,
                    "lower_bound": 0.0,
                    "upper_bound": 100.0,
                },
            }
        }
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text(yaml.dump(metrics))
        pc = ProblemContext(problem_dir)
        with pytest.raises(ValueError, match="Missing required.*is_valid"):
            _ = pc.metrics_context

    def test_invalid_is_valid_bounds_raises(self, tmp_path: Path) -> None:
        """is_valid metric with wrong bounds should raise."""
        metrics = {
            "specs": {
                "score": {
                    "description": "Primary score",
                    "higher_is_better": True,
                    "is_primary": True,
                    "lower_bound": 0.0,
                    "upper_bound": 100.0,
                },
                "is_valid": {
                    "description": "Validity flag",
                    "higher_is_better": True,
                    "is_primary": False,
                    "lower_bound": 0.0,
                    "upper_bound": 5.0,  # Wrong: must be 1.0
                    "significant_change": 1.0,
                    "sentinel_value": -1.0,
                },
            }
        }
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text(yaml.dump(metrics))
        pc = ProblemContext(problem_dir)
        with pytest.raises(
            ValueError, match="must have lower_bound=0.0 and upper_bound=1.0"
        ):
            _ = pc.metrics_context


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_all_present_passes(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)
        # Should not raise
        pc.validate()

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.VALIDATOR).unlink()
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="Missing required files"):
            pc.validate()

    def test_missing_initial_programs_dir_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        # Remove the initial_programs directory entirely
        import shutil

        shutil.rmtree(problem_dir / PL.INITIAL_PROGRAMS_DIR)
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="Missing required"):
            pc.validate()

    def test_empty_initial_programs_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        # Remove all .py files from initial_programs
        for py_file in (problem_dir / PL.INITIAL_PROGRAMS_DIR).glob("*.py"):
            py_file.unlink()
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="No Python files found"):
            pc.validate()


# ---------------------------------------------------------------------------
# TestMetricsContextCaching
# ---------------------------------------------------------------------------


class TestMetricsContextCaching:
    def test_second_access_uses_cache(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)

        ctx1 = pc.metrics_context
        ctx2 = pc.metrics_context
        assert ctx1 is ctx2

    def test_cache_survives_file_deletion(self, tmp_path: Path) -> None:
        """Cached metrics_context is still returned after the file is deleted."""
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(problem_dir)
        ctx1 = pc.metrics_context
        (problem_dir / PL.METRICS_FILE).unlink()
        ctx2 = pc.metrics_context
        assert ctx1 is ctx2


# ---------------------------------------------------------------------------
# Additional edge-case tests from audit
# ---------------------------------------------------------------------------


class TestFlatMetricsFormat:
    def test_flat_metrics_yaml_without_specs_key(self, tmp_path: Path) -> None:
        """Flat metrics.yaml (no 'specs' wrapper) should still parse."""
        flat_metrics = {
            "score": {
                "description": "Primary score",
                "higher_is_better": True,
                "is_primary": True,
                "lower_bound": 0.0,
                "upper_bound": 100.0,
            },
            "is_valid": {
                "description": "Validity flag",
                "higher_is_better": True,
                "is_primary": False,
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "significant_change": 1.0,
                "sentinel_value": 0.0,
            },
        }
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text(yaml.dump(flat_metrics))
        pc = ProblemContext(problem_dir)
        ctx = pc.metrics_context
        assert ctx.get_primary_key() == "score"


class TestEmptyAndMalformedYaml:
    def test_empty_metrics_yaml_raises(self, tmp_path: Path) -> None:
        """Empty file -> yaml.safe_load returns None -> ValueError."""
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text("")
        pc = ProblemContext(problem_dir)
        with pytest.raises(ValueError, match="metrics.yaml must be a mapping"):
            _ = pc.metrics_context

    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """Syntactically invalid YAML raises ValueError wrapping the parse error."""
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text("key: [\n  unclosed\n")
        pc = ProblemContext(problem_dir)
        with pytest.raises(ValueError, match="Failed to parse metrics.yaml"):
            _ = pc.metrics_context


class TestIsValidLowerBound:
    def test_invalid_is_valid_lower_bound_raises(self, tmp_path: Path) -> None:
        """is_valid with wrong lower_bound (not 0.0) should raise."""
        metrics = {
            "specs": {
                "score": {
                    "description": "Primary score",
                    "higher_is_better": True,
                    "is_primary": True,
                    "lower_bound": 0.0,
                    "upper_bound": 100.0,
                },
                "is_valid": {
                    "description": "Validity flag",
                    "higher_is_better": True,
                    "is_primary": False,
                    "lower_bound": -1.0,
                    "upper_bound": 1.0,
                    "significant_change": 1.0,
                    "sentinel_value": -2.0,
                },
            }
        }
        problem_dir = _make_problem_dir(tmp_path)
        (problem_dir / PL.METRICS_FILE).write_text(yaml.dump(metrics))
        pc = ProblemContext(problem_dir)
        with pytest.raises(
            ValueError, match="must have lower_bound=0.0 and upper_bound=1.0"
        ):
            _ = pc.metrics_context


class TestValidateAddContext:
    def test_validate_add_context_passes(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path, add_context=True)
        pc = ProblemContext(problem_dir)
        pc.validate(add_context=True)  # should not raise

    def test_validate_add_context_fails_when_missing(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path, add_context=False)
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="Missing required"):
            pc.validate(add_context=True)

    def test_only_non_py_files_in_initial_programs_raises(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        init_dir = problem_dir / PL.INITIAL_PROGRAMS_DIR
        for py_file in init_dir.glob("*.py"):
            py_file.unlink()
        (init_dir / "readme.txt").write_text("not a program")
        pc = ProblemContext(problem_dir)
        with pytest.raises(FileNotFoundError, match="No Python files found"):
            pc.validate()


class TestConstructorAcceptsString:
    def test_accepts_string_path(self, tmp_path: Path) -> None:
        problem_dir = _make_problem_dir(tmp_path)
        pc = ProblemContext(str(problem_dir))
        assert pc.task_description == "Solve the problem."
        assert pc.problem_dir == problem_dir.resolve()
