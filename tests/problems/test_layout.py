"""Tests for gigaevo/problems/layout.py — ProblemLayout scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.problems.config import (
    ContextSpec,
    FunctionSignature,
    HelperFunctionSpec,
    InitialProgram,
    ParameterSpec,
    ProblemConfig,
    ReturnSpec,
    TaskDescription,
    UtilsConfig,
    UtilsImportSpec,
)
from gigaevo.problems.layout import ProblemLayout
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricSpec

# ---------------------------------------------------------------------------
# Helper: build minimal ProblemConfig
# ---------------------------------------------------------------------------


def _make_config(
    *,
    name: str = "test_problem",
    description: str = "A test problem",
    add_context: bool = False,
    add_helper: bool = False,
    helper_functions: list[HelperFunctionSpec] | None = None,
    context_spec: ContextSpec | None = None,
    initial_programs: list[InitialProgram] | None = None,
    utils_imports: UtilsConfig | None = None,
    extra_metrics: dict[str, MetricSpec] | None = None,
) -> ProblemConfig:
    """Build a minimal valid ProblemConfig for testing."""
    entrypoint_params = [ParameterSpec(name="data", type_hint="np.ndarray")]
    validation_params = [ParameterSpec(name="solution", type_hint="np.ndarray")]

    if add_context:
        entrypoint_params.append(ParameterSpec(name="context", type_hint="dict"))
        validation_params.append(ParameterSpec(name="context", type_hint="dict"))

    metrics: dict[str, MetricSpec] = {
        "fitness": MetricSpec(
            description="Primary fitness score",
            decimals=3,
            is_primary=True,
            higher_is_better=True,
        ),
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    return ProblemConfig(
        name=name,
        description=description,
        entrypoint=FunctionSignature(
            params=entrypoint_params,
            returns=ReturnSpec(type_hint="np.ndarray", description="Solution array"),
        ),
        validation=FunctionSignature(params=validation_params),
        metrics=metrics,
        task_description=TaskDescription(objective="Solve the test problem"),
        add_context=add_context,
        add_helper=add_helper,
        helper_functions=helper_functions,
        context_spec=context_spec,
        initial_programs=initial_programs or [],
        utils_imports=utils_imports,
    )


# ---------------------------------------------------------------------------
# TestGetTemplateDir
# ---------------------------------------------------------------------------


class TestGetTemplateDir:
    """Tests for ProblemLayout._get_template_dir."""

    def test_valid_type_returns_path(self) -> None:
        """Known type 'programs' returns a valid directory path."""
        result = ProblemLayout._get_template_dir("programs")
        assert result.is_dir()
        assert result.name == "templates"
        assert "programs" in str(result)

    def test_invalid_type_raises(self) -> None:
        """Unknown type raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown problem type.*nonexistent"):
            ProblemLayout._get_template_dir("nonexistent")


# ---------------------------------------------------------------------------
# TestRegisterJinjaFilters
# ---------------------------------------------------------------------------


class TestRegisterJinjaFilters:
    """Tests for ProblemLayout._register_jinja_filters."""

    def _make_env(self, problem_type: str = "programs"):
        from jinja2 import Environment

        env = Environment()
        ProblemLayout._register_jinja_filters(env, problem_type)
        return env

    def test_param_string_with_types(self) -> None:
        """param_string filter renders typed parameter string."""
        env = self._make_env()
        sig_dict = FunctionSignature(
            params=[
                ParameterSpec(name="x", type_hint="int"),
                ParameterSpec(name="y", type_hint="float"),
            ]
        ).model_dump()
        result = env.filters["param_string"](sig_dict, with_types=True)
        assert result == "x: int, y: float"

    def test_param_string_without_types(self) -> None:
        """param_string filter with_types=False drops type hints."""
        env = self._make_env()
        sig_dict = FunctionSignature(
            params=[
                ParameterSpec(name="x", type_hint="int"),
                ParameterSpec(name="y", type_hint="float"),
            ]
        ).model_dump()
        result = env.filters["param_string"](sig_dict, with_types=False)
        assert result == "x, y"

    def test_wildcard_import_filter(self) -> None:
        """utils_import filter with ['*'] produces wildcard import."""
        env = self._make_env("programs")
        spec_dict = UtilsImportSpec(functions=["*"]).model_dump()
        result = env.filters["utils_import"](spec_dict)
        assert result == "from gigaevo.problems.types.programs.utils import *"

    def test_named_import_filter(self) -> None:
        """utils_import filter with named functions produces explicit import."""
        env = self._make_env("programs")
        spec_dict = UtilsImportSpec(functions=["foo", "bar"]).model_dump()
        result = env.filters["utils_import"](spec_dict)
        assert "foo" in result
        assert "bar" in result
        assert "import *" not in result
        assert result.startswith("from gigaevo.problems.types.programs.utils import ")

    def test_none_import_filter(self) -> None:
        """utils_import filter with None returns empty string."""
        env = self._make_env()
        result = env.filters["utils_import"](None)
        assert result == ""


# ---------------------------------------------------------------------------
# TestBuildTemplateContext
# ---------------------------------------------------------------------------


class TestBuildTemplateContext:
    """Tests for ProblemLayout._build_template_context."""

    def test_is_valid_metric_injected(self) -> None:
        """is_valid metric is automatically added to context metrics."""
        config = _make_config()
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert VALIDITY_KEY in ctx["metrics"]
        is_valid_spec = ctx["metrics"][VALIDITY_KEY]
        assert is_valid_spec["is_primary"] is False
        assert is_valid_spec["higher_is_better"] is True

    def test_primary_key_detected(self) -> None:
        """primary_key is set to the metric with is_primary=True."""
        config = _make_config()
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert ctx["primary_key"] == "fitness"

    def test_helper_functions_included(self) -> None:
        """helper_functions are serialized when present."""
        hf = HelperFunctionSpec(
            name="compute_distance",
            description="Compute distance",
            signature=FunctionSignature(
                params=[ParameterSpec(name="a"), ParameterSpec(name="b")]
            ),
        )
        config = _make_config(add_helper=True, helper_functions=[hf])
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert ctx["helper_functions"] is not None
        assert len(ctx["helper_functions"]) == 1
        assert ctx["helper_functions"][0]["name"] == "compute_distance"

    def test_context_spec_included(self) -> None:
        """context_spec is serialized when add_context=True."""
        cs = ContextSpec(description="Training data", fields={"X_train": "numpy array"})
        config = _make_config(add_context=True, context_spec=cs)
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert ctx["context_spec"] is not None
        assert ctx["context_spec"]["description"] == "Training data"

    def test_no_primary_key_when_missing(self) -> None:
        """primary_key is None when no metric is marked primary.

        We have to construct a config that bypasses validation to test this
        edge case in _build_template_context directly.
        """
        config = _make_config()
        # Temporarily replace metrics to have none primary (bypass validator)
        original_metrics = config.metrics
        config.metrics = {
            "score": MetricSpec(
                description="Score",
                is_primary=False,
                higher_is_better=True,
            )
        }
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert ctx["primary_key"] is None
        config.metrics = original_metrics

    def test_utils_functions_collected(self) -> None:
        """utils_functions collects unique function names from all import specs."""
        utils = UtilsConfig(
            validator=UtilsImportSpec(functions=["load_data", "parse_input"]),
            helper=UtilsImportSpec(functions=["load_data"]),
        )
        config = _make_config(utils_imports=utils)
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert sorted(ctx["utils_functions"]) == ["load_data", "parse_input"]


# ---------------------------------------------------------------------------
# TestGetFileTemplateMap
# ---------------------------------------------------------------------------


class TestGetFileTemplateMap:
    """Tests for ProblemLayout._get_file_template_map."""

    def test_base_files(self) -> None:
        """Base config produces task_description, metrics, and validate."""
        config = _make_config()
        file_map = ProblemLayout._get_file_template_map(config)
        assert ProblemLayout.TASK_DESCRIPTION in file_map
        assert ProblemLayout.METRICS_FILE in file_map
        assert ProblemLayout.VALIDATOR in file_map
        assert len(file_map) == 3

    def test_add_context(self) -> None:
        """add_context=True adds context.py to file map."""
        config = _make_config(add_context=True)
        file_map = ProblemLayout._get_file_template_map(config)
        assert ProblemLayout.CONTEXT_FILE in file_map
        assert len(file_map) == 4

    def test_add_helper(self) -> None:
        """add_helper=True adds helper.py to file map."""
        config = _make_config(add_helper=True)
        file_map = ProblemLayout._get_file_template_map(config)
        assert "helper.py" in file_map
        assert len(file_map) == 4

    def test_add_context_and_helper(self) -> None:
        """Both context and helper add their files."""
        config = _make_config(add_context=True, add_helper=True)
        file_map = ProblemLayout._get_file_template_map(config)
        assert ProblemLayout.CONTEXT_FILE in file_map
        assert "helper.py" in file_map
        assert len(file_map) == 5


# ---------------------------------------------------------------------------
# TestScaffold
# ---------------------------------------------------------------------------


class TestScaffold:
    """Tests for ProblemLayout.scaffold (end-to-end scaffolding)."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        """scaffold creates target dir and initial_programs subdir."""
        config = _make_config()
        target = tmp_path / "my_problem"
        ProblemLayout.scaffold(target, config)
        assert target.is_dir()
        assert (target / ProblemLayout.INITIAL_PROGRAMS_DIR).is_dir()

    def test_overwrite_guard(self, tmp_path: Path) -> None:
        """Existing directory without overwrite=True raises ValueError."""
        config = _make_config()
        target = tmp_path / "my_problem"
        target.mkdir()
        with pytest.raises(ValueError, match="exists.*overwrite"):
            ProblemLayout.scaffold(target, config)

    def test_overwrite_allowed(self, tmp_path: Path) -> None:
        """overwrite=True succeeds on existing directory."""
        config = _make_config()
        target = tmp_path / "my_problem"
        target.mkdir()
        result = ProblemLayout.scaffold(target, config, overwrite=True)
        assert result["target_dir"] == target

    def test_return_counts(self, tmp_path: Path) -> None:
        """Return dict reports correct file and program counts."""
        config = _make_config()
        target = tmp_path / "my_problem"
        result = ProblemLayout.scaffold(target, config)
        assert result["files_generated"] == 3  # task_description, metrics, validate
        assert result["initial_programs"] == 0

    def test_generated_files_exist(self, tmp_path: Path) -> None:
        """All expected files are written to disk."""
        config = _make_config()
        target = tmp_path / "my_problem"
        ProblemLayout.scaffold(target, config)
        assert (target / ProblemLayout.TASK_DESCRIPTION).is_file()
        assert (target / ProblemLayout.METRICS_FILE).is_file()
        assert (target / ProblemLayout.VALIDATOR).is_file()

    def test_initial_programs_generated(self, tmp_path: Path) -> None:
        """Initial programs are created in initial_programs/ subdir."""
        config = _make_config(
            initial_programs=[
                InitialProgram(name="baseline", description="Simple baseline"),
                InitialProgram(name="random", description="Random approach"),
            ]
        )
        target = tmp_path / "my_problem"
        result = ProblemLayout.scaffold(target, config)
        assert result["initial_programs"] == 2
        assert (target / ProblemLayout.INITIAL_PROGRAMS_DIR / "baseline.py").is_file()
        assert (target / ProblemLayout.INITIAL_PROGRAMS_DIR / "random.py").is_file()

    def test_scaffold_with_context_and_helper(self, tmp_path: Path) -> None:
        """scaffold with add_context and add_helper generates extra files."""
        hf = HelperFunctionSpec(
            name="helper_fn",
            description="A helper",
            signature=FunctionSignature(params=[ParameterSpec(name="x")]),
        )
        cs = ContextSpec(description="ctx", fields={"key": "value"})
        config = _make_config(
            add_context=True,
            add_helper=True,
            helper_functions=[hf],
            context_spec=cs,
        )
        target = tmp_path / "my_problem"
        result = ProblemLayout.scaffold(target, config)
        assert result["files_generated"] == 5
        assert (target / ProblemLayout.CONTEXT_FILE).is_file()
        assert (target / "helper.py").is_file()

    def test_template_content_references_problem_name(self, tmp_path: Path) -> None:
        """Generated files reference the problem name from config."""
        config = _make_config(name="my_cool_problem", description="Cool problem")
        target = tmp_path / "my_cool_problem"
        ProblemLayout.scaffold(target, config)
        content = (target / ProblemLayout.TASK_DESCRIPTION).read_text()
        assert "MY_COOL_PROBLEM" in content


# ---------------------------------------------------------------------------
# TestGetUtilsImportForFile
# ---------------------------------------------------------------------------


class TestGetUtilsImportForFile:
    """Tests for ProblemLayout._get_utils_import_for_file."""

    def test_validator_import(self) -> None:
        """Validator file gets its own utils import spec."""
        utils = UtilsConfig(
            validator=UtilsImportSpec(functions=["check_bounds"]),
        )
        config = _make_config(utils_imports=utils)
        result = ProblemLayout._get_utils_import_for_file(
            config, ProblemLayout.VALIDATOR
        )
        assert result is not None
        assert "check_bounds" in result["functions"]

    def test_none_fallback(self) -> None:
        """No utils_imports returns None for any file."""
        config = _make_config()
        result = ProblemLayout._get_utils_import_for_file(
            config, ProblemLayout.VALIDATOR
        )
        assert result is None

    def test_helper_import(self) -> None:
        """helper.py gets its own utils import spec."""
        utils = UtilsConfig(
            helper=UtilsImportSpec(functions=["utility_fn"]),
        )
        config = _make_config(utils_imports=utils)
        result = ProblemLayout._get_utils_import_for_file(config, "helper.py")
        assert result is not None
        assert "utility_fn" in result["functions"]

    def test_context_import(self) -> None:
        """context.py gets its own utils import spec."""
        utils = UtilsConfig(
            context=UtilsImportSpec(functions=["load_data"]),
        )
        config = _make_config(utils_imports=utils)
        result = ProblemLayout._get_utils_import_for_file(
            config, ProblemLayout.CONTEXT_FILE
        )
        assert result is not None
        assert "load_data" in result["functions"]

    def test_unknown_file_returns_none(self) -> None:
        """Unknown file name returns None even with utils configured."""
        utils = UtilsConfig(
            validator=UtilsImportSpec(functions=["check_bounds"]),
        )
        config = _make_config(utils_imports=utils)
        result = ProblemLayout._get_utils_import_for_file(config, "unknown.py")
        assert result is None


# ---------------------------------------------------------------------------
# TestRequiredFiles
# ---------------------------------------------------------------------------


class TestRequiredFiles:
    """Tests for ProblemLayout.required_files and required_directories."""

    def test_base_required_files(self) -> None:
        """Base required files include task_description, validate, and metrics."""
        files = ProblemLayout.required_files()
        assert ProblemLayout.TASK_DESCRIPTION in files
        assert ProblemLayout.VALIDATOR in files
        assert ProblemLayout.METRICS_FILE in files
        assert len(files) == 3

    def test_required_files_with_context(self) -> None:
        """add_context=True adds context.py to required files."""
        files = ProblemLayout.required_files(add_context=True)
        assert ProblemLayout.CONTEXT_FILE in files
        assert len(files) == 4

    def test_required_directories(self) -> None:
        """required_directories returns initial_programs."""
        dirs = ProblemLayout.required_directories()
        assert ProblemLayout.INITIAL_PROGRAMS_DIR in dirs
        assert len(dirs) == 1


# ---------------------------------------------------------------------------
# TestWildcardExclusion
# ---------------------------------------------------------------------------


class TestWildcardExclusion:
    """Tests that wildcard ['*'] imports are excluded from utils_functions."""

    def test_wildcard_excluded_from_utils_functions(self) -> None:
        """Wildcard imports should NOT appear in utils_functions list."""
        utils = UtilsConfig(
            validator=UtilsImportSpec(functions=["*"]),
            helper=UtilsImportSpec(functions=["real_func"]),
        )
        config = _make_config(utils_imports=utils)
        ctx = ProblemLayout._build_template_context(config, "programs")
        assert "*" not in ctx["utils_functions"]
        assert "real_func" in ctx["utils_functions"]


# ---------------------------------------------------------------------------
# TestScaffoldContent
# ---------------------------------------------------------------------------


class TestScaffoldContent:
    """Verify generated file content, not just file existence."""

    def test_metrics_yaml_contains_primary_metric(self, tmp_path: Path) -> None:
        """Generated metrics.yaml mentions the primary metric key."""
        config = _make_config()
        target = tmp_path / "content_test"
        ProblemLayout.scaffold(target, config)
        content = (target / ProblemLayout.METRICS_FILE).read_text()
        assert "fitness" in content
        assert "is_valid" in content

    def test_validate_py_is_non_empty(self, tmp_path: Path) -> None:
        """Generated validate.py is a non-empty file."""
        config = _make_config()
        target = tmp_path / "validate_test"
        ProblemLayout.scaffold(target, config)
        content = (target / ProblemLayout.VALIDATOR).read_text()
        assert len(content) > 0
