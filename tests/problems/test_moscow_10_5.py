"""Regression and execution-smoke tests for problems/moscow_10_5."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
import numpy as np
import pytest
import yaml

from gigaevo.problems.context import ProblemContext
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.program import Program
from gigaevo.programs.stages.python_executors.execution import (
    CallProgramFunction,
    CallValidatorFunction,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "moscow_10_5"
SEED_PATHS = sorted((PROBLEM_DIR / "initial_programs").glob("*.py"))


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def problem_modules():
    helper = _load_module("_test_moscow_10_5_helper", PROBLEM_DIR / "helper.py")

    # validate.py deliberately imports the problem-local module as ``helper``,
    # matching the GigaEvo subprocess PYTHONPATH.  Restore any unrelated module
    # with that generic name immediately after loading the validator.
    missing = object()
    previous_helper = sys.modules.get("helper", missing)
    sys.modules["helper"] = helper
    try:
        validator = _load_module(
            "_test_moscow_10_5_validator",
            PROBLEM_DIR / "validate.py",
        )
    finally:
        if previous_helper is missing:
            sys.modules.pop("helper", None)
        else:
            sys.modules["helper"] = previous_helper
    return helper, validator


def test_problem_bundle_and_metric_contract() -> None:
    context = ProblemContext(PROBLEM_DIR)
    context.validate()

    assert context.metrics_context.get_primary_key() == "fitness"
    assert context.metrics_context.get_bounds("fitness") == (-10.0, 1.0)
    assert "active_log_density" in context.metrics_context.specs
    assert len(SEED_PATHS) == 5


@pytest.mark.parametrize("seed_path", SEED_PATHS, ids=lambda path: path.stem)
async def test_all_initial_programs_use_gigaevo_execution_path(
    seed_path: Path,
) -> None:
    """Execute every baseline through the isolated production stages."""

    program = Program(code=seed_path.read_text())
    program_stage = CallProgramFunction(
        function_name="entrypoint",
        python_path=[PROBLEM_DIR],
        timeout=20,
    )
    program_stage.attach_inputs({})
    payload = await program_stage.compute(program)
    assert not isinstance(payload, ProgramStageResult), (
        f"{seed_path.name} failed to execute: {payload}"
    )

    validator_stage = CallValidatorFunction(
        path=PROBLEM_DIR / "validate.py",
        timeout=20,
    )
    validator_stage.attach_inputs(
        {
            "payload": payload,
            "context": None,
        }
    )
    result = await validator_stage.compute(program)
    assert not isinstance(result, ProgramStageResult), (
        f"{seed_path.name} failed validation: {result}"
    )

    metrics, feedback = result.data
    context = ProblemContext(PROBLEM_DIR)
    assert metrics["is_valid"] == 1.0
    assert isinstance(feedback, str) and "Phi=" in feedback

    for key, value in metrics.items():
        assert key in context.metrics_context.specs
        assert math.isfinite(value)
        bounds = context.metrics_context.get_bounds(key)
        if bounds is not None:
            assert bounds[0] <= value <= bounds[1]


def test_series_parallel_seed_is_exact_equality_regression(
    problem_modules,
) -> None:
    helper, validator = problem_modules
    seed = _load_module(
        "_test_moscow_10_5_theta_seed",
        PROBLEM_DIR / "initial_programs" / "series_parallel_theta.py",
    )

    metrics, _ = validator.validate(seed.entrypoint())

    assert metrics["phi"] == pytest.approx(1.0, abs=5.0e-13)
    assert metrics["raw_margin"] == pytest.approx(0.0, abs=5.0e-13)
    assert metrics["fitness"] == pytest.approx(
        -helper.COUNTEREXAMPLE_TOL,
        abs=5.0e-13,
    )
    assert metrics["numerical_basis_count"] == 64.0
    assert metrics["basis_density"] == pytest.approx(64 / helper.NUM_SUBSETS)
    assert metrics["active_count"] == 64.0

    analysis = helper.analyze_candidate(seed.entrypoint())
    positive_scores = analysis.subset_scores[
        analysis.subset_scores > helper.NUMERICAL_BASIS_EIGENVALUE_TOL
    ]
    assert positive_scores == pytest.approx(
        np.full(64, 0.1),
        abs=5.0e-13,
    )


def test_score_depends_only_on_the_subspace(problem_modules) -> None:
    helper, _ = problem_modules
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((10, 5))
    change_of_basis = rng.standard_normal((5, 5)) + 2.0 * np.eye(5)

    reference = helper.analyze_candidate(matrix)
    transformed = helper.analyze_candidate(matrix @ change_of_basis)

    signs = np.diag(rng.choice([-1.0, 1.0], size=10))
    signed = helper.analyze_candidate(signs @ matrix)

    assert transformed.phi == pytest.approx(reference.phi, abs=2.0e-13)
    assert np.sort(transformed.subset_scores) == pytest.approx(
        np.sort(reference.subset_scores),
        abs=2.0e-13,
    )
    assert signed.phi == pytest.approx(reference.phi, abs=2.0e-13)
    assert np.sort(signed.subset_scores) == pytest.approx(
        np.sort(reference.subset_scores),
        abs=2.0e-13,
    )


def test_orthogonal_complement_self_duality(problem_modules) -> None:
    helper, _ = problem_modules
    rng = np.random.default_rng(2027)
    matrix = rng.standard_normal((10, 5))
    analysis = helper.analyze_candidate(matrix)

    _, _, right_vectors_t = np.linalg.svd(
        analysis.basis.T,
        full_matrices=True,
    )
    complement_basis = right_vectors_t[5:].T
    complement = helper.analyze_candidate(complement_basis)

    complement_lookup = {
        tuple(int(index) for index in subset): subset_index
        for subset_index, subset in enumerate(helper.SUBSETS)
    }
    all_rows = frozenset(range(helper.N_ROWS))
    for subset_index, subset in enumerate(helper.SUBSETS):
        other_rows = tuple(sorted(all_rows.difference(int(index) for index in subset)))
        other_index = complement_lookup[other_rows]
        assert analysis.subset_scores[subset_index] == pytest.approx(
            complement.subset_scores[other_index],
            abs=5.0e-13,
        )

    assert complement.phi == pytest.approx(analysis.phi, abs=5.0e-13)
    assert complement.numerical_basis_count == analysis.numerical_basis_count
    assert complement.active_count == analysis.active_count


def test_initial_programs_are_deterministic_and_cover_distinct_cells(
    problem_modules,
) -> None:
    _, validator = problem_modules
    cells: set[tuple[int, int]] = set()

    for seed_index, seed_path in enumerate(SEED_PATHS):
        seed = _load_module(
            f"_test_moscow_10_5_determinism_seed_{seed_index}",
            seed_path,
        )
        first = seed.entrypoint()
        second = seed.entrypoint()
        assert np.array_equal(first, second)

        first_metrics, _ = validator.validate(first)
        second_metrics, _ = validator.validate(second)
        assert first_metrics == second_metrics
        cells.add(
            (
                min(9, int(first_metrics["basis_density"] * 10)),
                min(9, int(first_metrics["active_log_density"] * 10)),
            )
        )

    assert len(cells) == len(SEED_PATHS) == 5


@pytest.mark.parametrize(
    "bad_candidate, message",
    [
        (np.zeros((10, 4)), "Expected a"),
        (np.zeros((10, 5)), "zero matrix"),
        (np.column_stack([np.eye(10)[:, :4], np.eye(10)[:, 0]]), "rank deficient"),
        (np.full((10, 5), np.nan), "NaN or infinite"),
        (np.full((10, 5), 1.0j), "must be real"),
    ],
)
def test_validator_rejects_malformed_candidates(
    problem_modules,
    bad_candidate: np.ndarray,
    message: str,
) -> None:
    _, validator = problem_modules
    with pytest.raises(ValueError, match=message):
        validator.validate(bad_candidate)


def test_qd_preset_is_fixed_at_exactly_100_cells() -> None:
    config_path = REPO_ROOT / "config" / "algorithm" / "moscow_10_5_qd.yaml"
    config = yaml.safe_load(config_path.read_text())
    behavior = config["behavior_space"]

    assert behavior["keys"] == ["basis_density", "active_log_density"]
    assert behavior["resolutions"] == [10, 10]
    assert behavior["bounds"] == [[0.0, 1.0], [0.0, 1.0]]
    assert behavior["dynamic"] is False
    assert config["islands"][0]["max_size"] == 100

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str((REPO_ROOT / "config").absolute()),
            version_base=None,
        ):
            composed = compose(
                config_name="config",
                overrides=[
                    "problem.name=moscow_10_5",
                    "algorithm=moscow_10_5_qd",
                ],
            )
        space = instantiate(composed.behavior_space)
    finally:
        GlobalHydra.instance().clear()

    assert space.total_cells == 100
    assert space.get_cell(
        {
            "basis_density": 0.0,
            "active_log_density": 0.0,
        }
    ) == (0, 0)
    assert space.get_cell(
        {
            "basis_density": 1.0,
            "active_log_density": 1.0,
        }
    ) == (9, 9)
