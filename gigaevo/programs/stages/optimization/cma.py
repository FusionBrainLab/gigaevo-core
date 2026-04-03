"""CMA-ES numerical constant optimization stage.

Extracts numerical constants from program code, optimizes them using CMA-ES
with a validator script as the fitness function, and returns the optimized code.

Key features of the CMA-ES integration:

* **Relative sigma** -- per-variable step-size scaling (``CMA_stds``)
  proportional to each constant's magnitude so that tiny and large constants
  are explored at an appropriate resolution.
* **Adaptive penalty** -- failed evaluations receive a penalty derived from
  the current generation's observed fitnesses rather than a fixed extreme
  value, preventing distortion of the CMA covariance matrix.
* **Optional automatic bounds** -- per-variable search bounds inferred from
  the initial constant values.
"""

from __future__ import annotations

import ast
import asyncio
import copy
from pathlib import Path
from typing import Any, cast

import cma
from loguru import logger
from pydantic import BaseModel

from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.optimization.utils import (
    OptimizationInput,
    build_eval_code,
    evaluate_single,
    make_numeric_const_node,
    read_validator,
)
from gigaevo.programs.stages.stage_registry import StageRegistry

# ---------------------------------------------------------------------------
# AST helpers -- CMA-specific constant extraction & substitution
# ---------------------------------------------------------------------------


class _ConstantInfo(BaseModel, frozen=True):
    """Bookkeeping for one extracted numerical constant."""

    value: float
    lineno: int
    col_offset: int
    is_negated: bool = False  # True when from -Constant (UnaryOp)
    was_int: bool = False  # True when the original literal was int


def _should_extract(
    value: float,
    *,
    skip_zero_one: bool,
    skip_integers: bool,
    min_abs_value: float | None,
    max_abs_value: float | None,
    raw_value: int | float,
) -> bool:
    """Return ``True`` if *value* should be extracted for CMA optimisation."""
    if isinstance(raw_value, bool):
        return False
    if skip_zero_one and value in (0.0, 1.0, -1.0):
        return False
    if skip_integers and isinstance(raw_value, int):
        return False
    if min_abs_value is not None and abs(value) < min_abs_value:
        return False
    if max_abs_value is not None and abs(value) > max_abs_value:
        return False
    return True


def _extract_constants(
    code: str,
    *,
    skip_zero_one: bool = True,
    skip_integers: bool = False,
    min_abs_value: float | None = None,
    max_abs_value: float | None = None,
) -> tuple[ast.Module, list[_ConstantInfo]]:
    """Parse *code* and collect optimisable numerical constants.

    Handles ``-Constant`` (``UnaryOp(USub, Constant)``) as a single signed
    constant so that CMA-ES can freely move through zero.
    """
    tree = ast.parse(code)
    constants: list[_ConstantInfo] = []
    # Track inner Constant nodes already consumed by a UnaryOp(USub, ...)
    _consumed: set[int] = set()

    for node in ast.walk(tree):
        # Pattern: -<numeric literal>
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            raw = node.operand.value
            fval = -float(raw)
            if _should_extract(
                fval,
                raw_value=raw,
                skip_zero_one=skip_zero_one,
                skip_integers=skip_integers,
                min_abs_value=min_abs_value,
                max_abs_value=max_abs_value,
            ):
                constants.append(
                    _ConstantInfo(
                        value=fval,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        is_negated=True,
                        was_int=isinstance(raw, int),
                    )
                )
            _consumed.add(id(node.operand))
            continue

        # Plain numeric literal
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and id(node) not in _consumed
        ):
            raw = node.value
            fval = float(raw)
            if _should_extract(
                fval,
                raw_value=raw,
                skip_zero_one=skip_zero_one,
                skip_integers=skip_integers,
                min_abs_value=min_abs_value,
                max_abs_value=max_abs_value,
            ):
                constants.append(
                    _ConstantInfo(
                        value=fval,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        was_int=isinstance(raw, int),
                    )
                )

    return tree, constants


class _ConstantParameterizer(ast.NodeTransformer):
    """Replace extracted constants with ``_cma_params[i]`` subscripts."""

    def __init__(self, constants: list[_ConstantInfo], param_name: str = "_cma_params"):
        self._param_name = param_name
        # Map (lineno, col) -> index; separate maps for UnaryOp vs Constant.
        self._neg_positions: dict[tuple[int, int], int] = {}
        self._pos_positions: dict[tuple[int, int], int] = {}
        for i, c in enumerate(constants):
            if c.is_negated:
                self._neg_positions[(c.lineno, c.col_offset)] = i
            else:
                self._pos_positions[(c.lineno, c.col_offset)] = i

    def _subscript(self, idx: int, src_node: ast.AST) -> ast.Subscript:
        node = ast.Subscript(
            value=ast.Name(id=self._param_name, ctx=ast.Load()),
            slice=ast.Constant(value=idx),
            ctx=ast.Load(),
        )
        return ast.copy_location(node, src_node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        key = (node.lineno, node.col_offset)
        if key in self._neg_positions:
            return self._subscript(self._neg_positions[key], node)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        key = (node.lineno, node.col_offset)
        if key in self._pos_positions:
            return self._subscript(self._pos_positions[key], node)
        return node


class _ConstantSubstitutor(ast.NodeTransformer):
    """Replace extracted constants with concrete optimised values.

    When the original constant was an ``int``, the optimised float is
    rounded back to ``int`` so that call-sites like ``range(n)`` remain
    valid.
    """

    def __init__(self, constants: list[_ConstantInfo], new_values: list[float]):
        self._neg_positions: dict[tuple[int, int], tuple[float, bool]] = {}
        self._pos_positions: dict[tuple[int, int], tuple[float, bool]] = {}
        for c, v in zip(constants, new_values):
            entry = (v, c.was_int)
            if c.is_negated:
                self._neg_positions[(c.lineno, c.col_offset)] = entry
            else:
                self._pos_positions[(c.lineno, c.col_offset)] = entry

    @staticmethod
    def _coerce(value: float, was_int: bool) -> int | float:
        """Round back to ``int`` when the original literal was an integer."""
        return round(value) if was_int else value

    def _make_const(self, value: float, src_node: ast.AST, *, was_int: bool) -> ast.AST:
        """Return an AST node for *value*, using ``-abs(value)`` when negative."""
        return make_numeric_const_node(value, was_int, src_node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        key = (node.lineno, node.col_offset)
        if key in self._neg_positions:
            value, was_int = self._neg_positions[key]
            return self._make_const(value, node, was_int=was_int)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        key = (node.lineno, node.col_offset)
        if key in self._pos_positions:
            value, was_int = self._pos_positions[key]
            return self._make_const(value, node, was_int=was_int)
        return node


def _parameterize(tree: ast.Module, constants: list[_ConstantInfo]) -> str:
    """Return code string with constants replaced by ``_cma_params[i]``."""
    new_tree = _ConstantParameterizer(constants).visit(copy.deepcopy(tree))
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def _substitute(
    tree: ast.Module, constants: list[_ConstantInfo], new_values: list[float]
) -> str:
    """Return code string with constants replaced by *new_values*."""
    new_tree = _ConstantSubstitutor(constants, new_values).visit(copy.deepcopy(tree))
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


# ---------------------------------------------------------------------------
# Stage I/O
# ---------------------------------------------------------------------------


class CMAOptimizationOutput(StageIO):
    """Output produced by :class:`CMANumericalOptimizationStage`."""

    optimized_code: str
    best_scores: dict[str, float]
    n_constants: int
    n_generations: int
    initial_constants: list[float]
    optimized_constants: list[float]


# ---------------------------------------------------------------------------
# The Stage
# ---------------------------------------------------------------------------


@StageRegistry.register(
    description="Optimize numerical constants in program code using CMA-ES"
)
class CMANumericalOptimizationStage(Stage):
    """Extract numerical constants from ``Program.code`` and tune them via CMA-ES.

    **How it works**

    1. The program's source is parsed to find float / int literal constants.
    2. Those constants are replaced with entries in a parameter vector.
    3. CMA-ES proposes candidate parameter vectors; each candidate is evaluated
       by running the program's entry-point function and scoring its output
       through an external *validator* script.
    4. After convergence (or budget exhaustion) the best constants are
       substituted back into the source and returned as ``optimized_code``.

    **Validator contract**

    The validator Python file must define a function (default ``validate``)
    with one of these signatures::

        def validate(program_output) -> dict[str, float]
        def validate(context, program_output) -> dict[str, float]

    The returned dict must contain ``score_key``.

    **DAG wiring**

    The only (optional) DAG input is ``context`` -- an :class:`AnyContainer`
    forwarded to both the program function and the validator.

    Parameters
    ----------
    validator_path : Path
        Path to the validator ``.py`` file.
    score_key : str
        Key in the validator's returned dict to optimise.
    minimize : bool
        If ``True`` minimise *score_key*; otherwise maximise.
        Default ``False`` (maximise).
    sigma0 : float
        CMA-ES initial step-size (default ``0.2``).
    relative_sigma : bool
        Scale step-size per variable (default ``True``).
    sigma_floor : float
        Minimum per-variable scale (default ``0.01``).
    bound_margin : float | None
        Automatic per-variable bounds margin (default ``None``).
    adaptive_penalty : bool
        Generation-adaptive penalty for failed evals (default ``True``).
    max_generations : int
        Hard cap on CMA-ES generations (default ``100``).
    population_size : int | None
        Override CMA population size (default ``None``).
    max_parallel : int
        Max concurrent evaluation sub-processes (default ``8``).
    eval_timeout : int
        Timeout in seconds per evaluation (default ``30``).
    function_name : str
        Function to call inside the program (default ``"run_code"``).
    validator_fn : str
        Function to call inside the validator (default ``"validate"``).
    skip_zero_one : bool
        Skip constants 0, 1, -1 (default ``True``).
    skip_integers : bool
        Skip integer-typed constants (default ``False``).
    min_abs_value, max_abs_value : float | None
        Filter constants by absolute value (default ``None``).
    update_program_code : bool
        Overwrite ``program.code`` in-place (default ``True``).
    penalty_fitness : float | None
        Fallback penalty for failed evaluations (default ``None``).
    python_path : list[Path] | None
        Extra ``sys.path`` entries for sub-processes (default ``None``).
    max_memory_mb : int | None
        Per-evaluation RSS memory cap in MB (default ``None``).
    """

    InputsModel = OptimizationInput
    OutputModel = CMAOptimizationOutput

    def __init__(
        self,
        *,
        validator_path: Path,
        score_key: str,
        minimize: bool = False,
        sigma0: float = 0.2,
        relative_sigma: bool = True,
        sigma_floor: float = 0.01,
        bound_margin: float | None = None,
        adaptive_penalty: bool = True,
        max_generations: int = 100,
        population_size: int | None = None,
        max_parallel: int = 8,
        eval_timeout: int = 30,
        function_name: str = "run_code",
        validator_fn: str = "validate",
        skip_zero_one: bool = True,
        skip_integers: bool = False,
        min_abs_value: float | None = None,
        max_abs_value: float | None = None,
        update_program_code: bool = True,
        penalty_fitness: float | None = None,
        python_path: list[Path] | None = None,
        max_memory_mb: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._validator_code = read_validator(validator_path)

        # -- Store hyper-parameters ----------------------------------------
        self.score_key = score_key
        self.minimize = minimize
        self.sigma0 = sigma0
        self.relative_sigma = relative_sigma
        self.sigma_floor = sigma_floor
        self.bound_margin = bound_margin
        self.adaptive_penalty = adaptive_penalty
        self.max_generations = max_generations
        self.population_size = population_size
        self.max_parallel = max_parallel
        self.eval_timeout = eval_timeout
        self.function_name = function_name
        self.validator_fn = validator_fn
        self.skip_zero_one = skip_zero_one
        self.skip_integers = skip_integers
        self.min_abs_value = min_abs_value
        self.max_abs_value = max_abs_value
        self.update_program_code = update_program_code
        self.python_path = python_path or []
        self.max_memory_mb = max_memory_mb

        # Fallback penalty in CMA space (CMA minimises: higher -> worse).
        if penalty_fitness is not None:
            self._fallback_penalty: float = (
                penalty_fitness if minimize else -penalty_fitness
            )
        else:
            self._fallback_penalty = 1e18

    def _build_eval_code(self, parameterized_code: str, params: list[float]) -> str:
        """Compose a self-contained script that runs *program -> validator*."""
        # Ensure plain Python floats (CMA-ES returns numpy.float64 whose
        # repr is ``np.float64(...)`` -- invalid in the subprocess).
        native_params = [float(p) for p in params]
        return build_eval_code(
            validator_code=self._validator_code,
            program_code=parameterized_code,
            function_name=self.function_name,
            validator_fn=self.validator_fn,
            eval_fn_name="_cma_eval",
            preamble_lines=[f"_cma_params = {native_params!r}"],
        )

    async def _evaluate_single(
        self,
        parameterized_code: str,
        candidate: list[float],
        context: Any,
    ) -> tuple[dict[str, float] | None, str | None]:
        """Run one candidate and return (score_dict, error_message)."""
        eval_code = self._build_eval_code(parameterized_code, candidate)
        return await evaluate_single(
            eval_code=eval_code,
            eval_fn_name="_cma_eval",
            context=context,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="CMA",
        )

    async def _evaluate_population(
        self,
        parameterized_code: str,
        population: list[list[float]],
        context: Any,
    ) -> tuple[list[float], int]:
        """Evaluate a whole generation, respecting *max_parallel*.

        Returns ``(fitnesses, n_ok)`` where *n_ok* is the count of
        successful (non-penalty) evaluations.
        """
        sem = asyncio.Semaphore(self.max_parallel)

        async def _bounded(candidate: list[float]):
            async with sem:
                scores, error = await self._evaluate_single(
                    parameterized_code, candidate, context
                )
                if scores is None:
                    # Return exception with error message to signal failure
                    return RuntimeError(f"Evaluation failed: {error}")
                return scores

        raw_results = await asyncio.gather(
            *(_bounded(c) for c in population), return_exceptions=True
        )

        # -- First pass: collect valid CMA fitnesses -----------------------
        valid_cma_fitnesses: list[float] = []
        for r in raw_results:
            if isinstance(r, dict) and self.score_key in r:
                score = float(r[self.score_key])
                valid_cma_fitnesses.append(score if self.minimize else -score)

        # -- Decide penalty for this generation ----------------------------
        if self.adaptive_penalty and valid_cma_fitnesses:
            worst_valid = max(valid_cma_fitnesses)
            best_valid = min(valid_cma_fitnesses)
            fitness_range = abs(worst_valid - best_valid)
            scale = max(fitness_range, abs(worst_valid)) or 1.0
            penalty = worst_valid + 3.0 * scale
        else:
            penalty = self._fallback_penalty

        # -- Second pass: build fitness list with penalty ------------------
        fitnesses: list[float] = []
        for r in raw_results:
            if isinstance(r, BaseException) or r is None:
                fitnesses.append(penalty)
            elif isinstance(r, dict) and self.score_key in r:
                score = float(r[self.score_key])
                fitnesses.append(score if self.minimize else -score)
            else:
                fitnesses.append(penalty)

        n_ok = len(valid_cma_fitnesses)
        return fitnesses, n_ok

    async def compute(self, program: Program) -> CMAOptimizationOutput:
        code = program.code
        pid = program.id[:8]

        # 1. Extract numerical constants ------------------------------------
        tree, constants = _extract_constants(
            code,
            skip_zero_one=self.skip_zero_one,
            skip_integers=self.skip_integers,
            min_abs_value=self.min_abs_value,
            max_abs_value=self.max_abs_value,
        )

        if not constants:
            logger.info(
                "[CMA][{}] No optimisable constants; returning original code.",
                pid,
            )
            return CMAOptimizationOutput(
                optimized_code=code,
                best_scores={},
                n_constants=0,
                n_generations=0,
                initial_constants=[],
                optimized_constants=[],
            )

        initial_values = [c.value for c in constants]
        n = len(initial_values)
        logger.info(
            "[CMA][{}] Found {} optimisable constants: {}", pid, n, initial_values
        )

        # 2. Parameterise ---------------------------------------------------
        parameterized_code = _parameterize(tree, constants)

        # 3. Resolve context ------------------------------------------------
        params = cast(OptimizationInput, self.params)
        ctx = params.context.data if params.context is not None else None

        # 4. Run CMA-ES ----------------------------------------------------
        opts: dict[str, Any] = {
            "maxiter": self.max_generations,
            "verbose": -9,
        }
        if self.population_size is not None:
            opts["popsize"] = self.population_size

        int_indices = [i for i, c in enumerate(constants) if c.was_int]
        if int_indices:
            opts["integer_variables"] = int_indices
            logger.debug("[CMA][{}] Integer variable indices: {}", pid, int_indices)

        cma_stds: list[float] | None = None
        if self.relative_sigma:
            cma_stds = [max(abs(v), self.sigma_floor) for v in initial_values]
            opts["CMA_stds"] = cma_stds
            eff_sigmas = [round(self.sigma0 * s, 6) for s in cma_stds]
            logger.debug(
                "[CMA][{}] Relative sigma: effective per-var sigmas={}",
                pid,
                eff_sigmas,
            )

        if self.bound_margin is not None:
            scales = cma_stds if cma_stds is not None else [1.0] * n
            lower = [v - self.bound_margin * s for v, s in zip(initial_values, scales)]
            upper = [v + self.bound_margin * s for v, s in zip(initial_values, scales)]
            opts["bounds"] = [lower, upper]
            logger.debug(
                "[CMA][{}] Auto bounds (margin={}): lower={}, upper={}",
                pid,
                self.bound_margin,
                [round(b, 6) for b in lower],
                [round(b, 6) for b in upper],
            )

        es = cma.CMAEvolutionStrategy(initial_values, self.sigma0, opts)

        baseline_scores, baseline_err = await evaluate_single(
            eval_code=self._build_eval_code(parameterized_code, initial_values),
            eval_fn_name="_cma_eval",
            context=ctx,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="CMA",
        )
        best_solution = list(initial_values)
        best_fitness = self._fallback_penalty
        best_scores: dict[str, float] = {}
        ever_succeeded = False

        if baseline_scores is not None:
            best_scores = baseline_scores
            score = float(baseline_scores[self.score_key])
            best_fitness = score if self.minimize else -score
            ever_succeeded = True
            logger.info("[CMA][{}] Baseline {}={:.6f}", pid, self.score_key, score)
        else:
            logger.info(
                "[CMA][{}] Baseline evaluation failed (original constants invalid). "
                "Proceeding with optimization to find valid constants.\nError details: {}",
                pid,
                baseline_err or "Unknown error",
            )

        generation = 0
        for generation in range(1, self.max_generations + 1):
            solutions = es.ask()
            fitnesses, n_ok = await self._evaluate_population(
                parameterized_code,
                [list(s) for s in solutions],
                ctx,
            )
            es.tell(solutions, fitnesses)

            n_pop = len(fitnesses)
            if n_ok > 0:
                ever_succeeded = True

            gen_best_idx = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
            improved = fitnesses[gen_best_idx] < best_fitness
            if improved:
                best_fitness = fitnesses[gen_best_idx]
                best_solution = list(solutions[gen_best_idx])
                # Re-evaluate best to capture the full score dict
                # Note: We use self._evaluate_single to ignore the error msg in this loop
                scores, _ = await self._evaluate_single(
                    parameterized_code, best_solution, ctx
                )
                if scores is not None:
                    best_scores = scores

            display_score = -best_fitness if not self.minimize else best_fitness

            n_fail = n_pop - n_ok
            fail_tag = f" fail={n_fail}" if n_fail else ""
            log_fn = logger.info if improved else logger.debug
            log_fn(
                "[CMA][{}] gen={} ok={}/{}{}  best {}={:.6f}",
                pid,
                generation,
                n_ok,
                n_pop,
                fail_tag,
                self.score_key,
                display_score,
            )

            stop_reasons = es.stop()
            if stop_reasons and ever_succeeded:
                logger.info("[CMA][{}] Converged early: {}", pid, dict(stop_reasons))
                break

        display_final = -best_fitness if not self.minimize else best_fitness

        # 5. Build optimised code -------------------------------------------
        optimized_code = _substitute(tree, constants, best_solution)

        # 6. Optionally update program in-place -----------------------------
        if self.update_program_code:
            program.code = optimized_code

        # 7. Summary log ----------------------------------------------------
        initial_score = (
            float(baseline_scores[self.score_key])
            if baseline_scores is not None
            else None
        )
        init_disp = f"{initial_score:.6f}" if initial_score is not None else "FAILED"
        logger.info(
            "[CMA][{}] == Done ==  gens={} constants={} {}={} -> {:.6f}  updated={}",
            pid,
            generation,
            n,
            self.score_key,
            init_disp,
            display_final,
            self.update_program_code,
        )
        logger.debug(
            "[CMA][{}] details: initial={} optimised={} scores={}",
            pid,
            [round(v, 6) for v in initial_values],
            [round(float(v), 6) for v in best_solution],
            best_scores,
        )

        return CMAOptimizationOutput(
            optimized_code=optimized_code,
            best_scores=best_scores,
            n_constants=n,
            n_generations=generation,
            initial_constants=initial_values,
            optimized_constants=best_solution,
        )
