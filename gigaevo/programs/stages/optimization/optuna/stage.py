"""OptunaOptimizationStage — LLM-guided hyperparameter optimization.

An LLM analyses program code, identifies meaningful hyperparameters, and
produces a **parameterized version** of the code where tuneable constants
are replaced by references to ``_optuna_params["name"]``.  Optuna then
tunes those parameters asynchronously.
"""

from __future__ import annotations

import ast
import asyncio
import math
from pathlib import Path
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import optuna
from optuna.importance import PedAnovaImportanceEvaluator

from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.optimization.optuna.desubstitution import (
    coerce_params,
    desubstitute_params,
    reindent_to_match_block,
    strip_line_number_prefix,
)
from gigaevo.programs.stages.optimization.optuna.models import (
    _DEFAULT_PRECISION,
    _IMPORTANCE_CHECK_MIN_TRIALS,
    _MAX_FAILURE_REASONS,
    _MIN_IMPORTANCE_CHECK_GAP,
    _MIN_PARAMS_FOR_IMPORTANCE,
    _MIN_POST_STARTUP_TRIALS,
    _OPTUNA_PARAMS_NAME,
    _PROGRESS_LOG_INTERVAL,
    OptunaOptimizationConfig,
    OptunaOptimizationOutput,
    OptunaSearchSpace,
    ParamSpec,
    default_max_params,
    default_n_startup_trials,
)
from gigaevo.programs.stages.optimization.optuna.prompts import (
    _SYSTEM_PROMPT,
    _USER_PROMPT_TEMPLATE,
)
from gigaevo.programs.stages.optimization.utils import (
    OptimizationInput,
    build_eval_code,
    evaluate_single,
    read_validator,
)
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)
from gigaevo.programs.stages.stage_registry import StageRegistry


@StageRegistry.register(
    description="LLM-guided hyperparameter optimization using Optuna"
)
class OptunaOptimizationStage(Stage):
    """Analyse program code with an LLM, then tune identified hyperparameters
    with Optuna.

    **How it works**

    1. An LLM analyses the program source and returns a structured search
       space together with a **parameterized version** of the code where
       tuneable constants are replaced by ``_optuna_params["name"]``
       references.
    2. Optuna runs ``n_trials`` asynchronous trials, each injecting
       different parameter values into the parameterized code and
       evaluating through an external validator script.
    3. The best parameter values are substituted back into the
       parameterized code (replacing ``_optuna_params["name"]`` with
       concrete literals) to produce clean ``optimized_code``.

    **Validator contract**

    Same as :class:`CMANumericalOptimizationStage` -- the validator Python
    file must define a function (default ``validate``) returning a dict
    that contains *score_key*.

    Parameters
    ----------
    llm : MultiModelRouter
        LLM wrapper for structured output calls.
    validator_path : Path
        Path to the validator ``.py`` file.
    score_key : str
        Key in the validator's returned dict to optimise.
    minimize : bool
        If ``True`` minimise *score_key*; otherwise maximise (default).
    n_trials : int
        Number of Optuna trials (default ``50``).
    max_parallel : int
        Maximum concurrent evaluation sub-processes (default ``8``).
    eval_timeout : int
        Timeout in seconds for each evaluation (default ``30``).
    function_name : str
        Function to call inside the program (default ``"run_code"``).
    validator_fn : str
        Function to call inside the validator (default ``"validate"``).
    update_program_code : bool
        If ``True`` (default), overwrite ``program.code`` in-place.
    add_tuned_comment : bool
        If ``True`` (default), append ``# tuned (Optuna)`` on lines where a
        parameter was substituted, so future LLM mutations know it was hyperparameter-tuned.
    task_description : str | None
        Optional task description forwarded to the LLM.
    python_path : list[Path] | None
        Extra ``sys.path`` entries for evaluation sub-processes.
    max_memory_mb : int | None
        Per-evaluation RSS memory cap in MB.
    llm_max_tokens : int
        Maximum tokens (thinking + output) for the LLM analysis call (default
        ``8192``).  Prevents runaway extended-thinking budget usage which can
        consume tens of thousands of tokens on a task that needs ~1000.
    """

    InputsModel = OptimizationInput
    OutputModel = OptunaOptimizationOutput

    def __init__(
        self,
        *,
        llm: MultiModelRouter,
        validator_path: Path,
        score_key: str,
        minimize: bool = False,
        n_trials: int = 50,
        max_parallel: int = 8,
        eval_timeout: int = 30,
        function_name: str = "run_code",
        validator_fn: str = "validate",
        update_program_code: bool = True,
        add_tuned_comment: bool = True,
        task_description: str | None = None,
        python_path: list[Path] | None = None,
        max_memory_mb: int | None = None,
        config: Optional[OptunaOptimizationConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._validator_code = read_validator(validator_path)

        self.llm = llm
        self.score_key = score_key
        self.minimize = minimize
        self.n_trials = n_trials
        self.max_parallel = max_parallel
        self.eval_timeout = eval_timeout
        self.function_name = function_name
        self.validator_fn = validator_fn
        self.update_program_code = update_program_code
        self.add_tuned_comment = add_tuned_comment
        self.task_description = task_description
        self.python_path = python_path or []
        self.max_memory_mb = max_memory_mb
        self.config = config or OptunaOptimizationConfig()

    # ------------------------------------------------------------------
    # Phase 1: LLM analysis
    # ------------------------------------------------------------------

    def _apply_modifications(
        self, original_code: str, search_space: OptunaSearchSpace
    ) -> str:
        """Apply the LLM's suggested line-range patches to the original code.

        Parameters
        ----------
        original_code : str
            The original program source code.
        search_space : OptunaSearchSpace
            The search space and modifications proposed by the LLM.

        Returns
        -------
        str
            The parameterized code with ``_optuna_params`` references.

        Raises
        ------
        ValueError
            If line ranges are invalid or if the resulting code has syntax errors.
        """
        lines = original_code.splitlines()
        num_lines = len(lines)
        mods = sorted(search_space.modifications, key=lambda x: x.start_line)

        for i, mod in enumerate(mods):
            if mod.start_line < 1 or mod.end_line > num_lines:
                raise ValueError(
                    f"Line range {mod.start_line}-{mod.end_line} out of bounds "
                    f"(1-{num_lines})"
                )
            if mod.start_line > mod.end_line:
                raise ValueError(
                    f"Invalid range: start_line {mod.start_line} > end_line {mod.end_line}"
                )
            if i > 0 and mod.start_line <= mods[i - 1].end_line:
                raise ValueError(
                    f"Overlapping line ranges: {mods[i - 1].start_line}-{mods[i - 1].end_line} "
                    f"and {mod.start_line}-{mod.end_line}"
                )

        new_lines = list(lines)
        for mod in reversed(mods):
            start_idx = mod.start_line - 1
            end_idx = mod.end_line
            replacement_lines = mod.parameterized_snippet.splitlines()
            # Defensive: strip any "N | " prefix if the LLM copied the numbered format
            replacement_lines = strip_line_number_prefix(replacement_lines)
            # Re-indent to match the original block so we never get "unexpected indent"
            original_block = lines[start_idx:end_idx]
            replacement_lines = reindent_to_match_block(
                replacement_lines, original_block
            )
            new_lines[start_idx:end_idx] = replacement_lines

        code = "\n".join(new_lines)
        if original_code.endswith("\n") and not code.endswith("\n"):
            code += "\n"

        if search_space.new_imports:
            imports_str = "\n".join(search_space.new_imports)
            code = f"{imports_str}\n{code}"

        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.error(
                "[Optuna] Parameterized code has syntax error: {}\nCode snippet around error:\n{}",
                e,
                "\n".join(code.splitlines()[max(0, e.lineno - 5) : e.lineno + 5])
                if e.lineno
                else "Unknown location",
            )
            raise ValueError(f"Parameterized code syntax error: {e}")

        return code

    async def _measure_baseline_runtime(
        self,
        code: str,
        context: dict[str, Any] | None,
        pid: str,
    ) -> float | None:
        """Run the original code once and return wall-clock seconds, or *None* on failure.

        Uses the existing ``build_eval_code`` / ``evaluate_single`` helpers with
        no ``_optuna_params`` preamble so the program runs with its hardcoded
        constants — i.e. the true baseline.
        """
        eval_code = build_eval_code(
            validator_code=self._validator_code,
            program_code=code,
            function_name=self.function_name,
            validator_fn=self.validator_fn,
            eval_fn_name="_optuna_eval",
            preamble_lines=None,
        )
        t0 = time.monotonic()
        result, err = await evaluate_single(
            eval_code=eval_code,
            eval_fn_name="_optuna_eval",
            context=context,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="Optuna-baseline",
        )
        elapsed = time.monotonic() - t0
        if result is None:
            logger.debug(
                "[Optuna][{}] Baseline runtime measurement failed: {}",
                pid,
                err,
            )
            return None
        logger.debug(
            "[Optuna][{}] Baseline runtime: {:.2f}s (timeout={}s)",
            pid,
            elapsed,
            self.eval_timeout,
        )
        return elapsed

    async def _analyze_code(
        self, code: str, baseline_runtime_s: float | None = None
    ) -> OptunaSearchSpace:
        """Call the LLM to propose a search space for *code*.

        Parameters
        ----------
        code : str
            The source code to analyze.
        baseline_runtime_s : float | None
            Wall-clock seconds for one baseline run, forwarded to the LLM
            prompt so it can judge runtime headroom.

        Returns
        -------
        OptunaSearchSpace
            The proposed parameters and code modifications.
        """
        # Provide line-numbered code to the LLM for precise patching
        lines = code.splitlines()
        numbered_code = "\n".join(
            f"{i + 1:4d} | {line}" for i, line in enumerate(lines)
        )

        task_section = ""
        if self.task_description:
            task_section = f"\n**Task context:**\n{self.task_description}\n"

        runtime_section = ""
        if baseline_runtime_s is not None:
            headroom = self.eval_timeout - baseline_runtime_s
            runtime_section = (
                f"\n**Runtime info:** The program currently runs in "
                f"~{baseline_runtime_s:.2f}s with a timeout of "
                f"{self.eval_timeout}s (~{headroom:.1f}s headroom). "
                f"Keep this in mind when proposing parameters that affect "
                f"runtime (e.g. iteration counts, grid sizes): ensure no "
                f"trial exceeds the timeout.\n"
            )

        direction = "minimize" if self.minimize else "maximize"
        n_startup = (
            self.config.n_startup_trials
            if self.config.n_startup_trials is not None
            else default_n_startup_trials(self.n_trials)
        )
        total_trials = n_startup + self.n_trials

        max_params = (
            self.config.max_params
            if self.config.max_params is not None
            else default_max_params(self.n_trials)
        )
        trials_per_param = total_trials // max(max_params, 1)

        user_msg = _USER_PROMPT_TEMPLATE.format(
            numbered_code=numbered_code,
            task_description_section=task_section,
            runtime_section=runtime_section,
            eval_timeout=self.eval_timeout,
            n_trials=self.n_trials,
            total_trials=total_trials,
            score_key=self.score_key,
            direction=direction,
            max_params=max_params,
        )

        structured_llm = self.llm.with_structured_output(OptunaSearchSpace)
        messages = [
            SystemMessage(
                content=_SYSTEM_PROMPT.format(
                    score_key=self.score_key,
                    eval_timeout=self.eval_timeout,
                    direction=direction,
                    max_params=max_params,
                    trials_per_param=trials_per_param,
                )
            ),
            HumanMessage(content=user_msg),
        ]
        result = await structured_llm.ainvoke(messages)
        return result

    # ------------------------------------------------------------------
    # Phase 2: Optuna evaluation
    # ------------------------------------------------------------------

    def _build_eval_code(
        self,
        parameterized_code: str,
        params: dict[str, Any],
        *,
        capture_program_output: bool = False,
    ) -> str:
        """Compose a self-contained script: params dict + program + validator.

        Parameters
        ----------
        parameterized_code : str
            The code containing ``_optuna_params`` references.
        params : dict[str, Any]
            The specific parameter values to inject for this evaluation.
        capture_program_output : bool
            If ``True``, the generated eval function returns
            ``(val_result, prog_result)`` so the caller can capture the raw
            program output alongside the validation score.

        Returns
        -------
        str
            A complete Python script ready for execution.
        """
        # Coerce int-like strings so range(k) etc. work when k comes from categorical/initial_value
        params = coerce_params(params)
        return build_eval_code(
            validator_code=self._validator_code,
            program_code=parameterized_code,
            function_name=self.function_name,
            validator_fn=self.validator_fn,
            eval_fn_name="_optuna_eval",
            preamble_lines=[f"{_OPTUNA_PARAMS_NAME} = {params!r}"],
            capture_program_output=capture_program_output,
        )

    async def _evaluate_single(
        self,
        parameterized_code: str,
        params: dict[str, Any],
        context: Optional[dict[str, Any]],
    ) -> tuple[dict[str, float] | None, Any, str | None]:
        """Run one trial capturing both scores and raw program output.

        Parameters
        ----------
        parameterized_code : str
            The code to evaluate.
        params : dict[str, Any]
            Parameters for this trial.
        context : Optional[dict[str, Any]]
            Optional evaluation context.

        Returns
        -------
        tuple[dict[str, float] | None, Any, str | None]
            ``(scores, program_output, error_message)``.
        """
        eval_code = self._build_eval_code(
            parameterized_code, params, capture_program_output=True
        )
        args = [context] if context is not None else []
        try:
            result, _, _ = await run_exec_runner(
                code=eval_code,
                function_name="_optuna_eval",
                args=args,
                kwargs={},
                python_path=list(self.python_path),
                timeout=self.eval_timeout,
                max_memory_mb=self.max_memory_mb,
            )
            val_result, prog_output = result
            # Validator may return (metrics_dict, artifact) — unwrap.
            if isinstance(val_result, tuple):
                val_result = val_result[0]
            if isinstance(val_result, dict) and self.score_key in val_result:
                return val_result, prog_output, None
            return None, None, f"Unexpected result type: {type(val_result).__name__}"
        except asyncio.TimeoutError:
            return None, None, "Timeout"
        except ExecRunnerError as exc:
            last_line = (exc.stderr or "").strip().rsplit("\n", 1)[-1]
            return None, None, f"{exc} | {last_line}"

    async def _run_optuna(
        self,
        parameterized_code: str,
        param_specs: list[ParamSpec],
        context: Optional[dict[str, Any]],
        pid: str,
    ) -> tuple[dict[str, Any], dict[str, float], int, int, Any]:
        """Run Optuna optimization.

        Parameters
        ----------
        parameterized_code : str
            The code to optimize.
        param_specs : list[ParamSpec]
            Specifications of parameters to tune.
        context : Optional[dict[str, Any]]
            Optional evaluation context.
        pid : str
            Short program ID for logging.

        Returns
        -------
        tuple[dict[str, Any], dict[str, float], int, int, Any]
            Best parameters, best scores, number of successful trials,
            total trials run, and the raw program output from the best trial.
        """
        direction = "minimize" if self.minimize else "maximize"

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # TPE with configurable startup trials and multivariate
        from_config = self.config.n_startup_trials is not None
        n_startup = (
            self.config.n_startup_trials
            if from_config
            else default_n_startup_trials(self.n_trials)
        )
        # Total trials = startup (random) + n_trials (TPE); startup trials are extra, not counted in n_trials.
        total_trials = n_startup + self.n_trials
        logger.debug(
            "[Optuna][{}] TPE sampler: n_startup_trials={} ({}), total_trials={} ({} + {} TPE)",
            pid,
            n_startup,
            "from config" if from_config else "default",
            total_trials,
            n_startup,
            self.n_trials,
        )
        has_categorical = any(p.param_type == "categorical" for p in param_specs)
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup,
            multivariate=self.config.multivariate,
            group=has_categorical,
            constant_liar=True,
            seed=self.config.random_state,
        )
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
        )

        sem = asyncio.Semaphore(self.max_parallel)

        best_scores: dict[str, float] = {}
        best_value: float | None = None
        best_params: dict[str, Any] = {p.name: p.initial_value for p in param_specs}
        best_prog_output: Any = None

        def _is_better(score: float) -> bool:
            if best_value is None:
                return True
            if direction == "minimize":
                return score < best_value
            return score > best_value

        # -- Two-phase importance schedule ------------------------------------
        # Early check (~33% into TPE) with conservative threshold to catch
        # clearly unimportant params; late check (~75% into TPE) with standard
        # threshold for final cleanup.  Both anchored to TPE phase, not total.
        n_tpe = self.n_trials

        config = self.config

        def _compute_check_point(override: int | None, fraction: float) -> int:
            raw = (
                override
                if override is not None
                else n_startup + max(_MIN_POST_STARTUP_TRIALS, int(n_tpe * fraction))
            )
            clamped = max(
                raw, n_startup + _MIN_POST_STARTUP_TRIALS, _IMPORTANCE_CHECK_MIN_TRIALS
            )
            if override is not None and clamped != override:
                logger.warning(
                    "[Optuna][{}] importance_check_at={} clamped to {} "
                    "(below minimum floor)",
                    pid,
                    override,
                    clamped,
                )
            return clamped

        importance_check_early = _compute_check_point(
            config.importance_check_at,
            config.early_tpe_fraction,
        )
        importance_check_late = _compute_check_point(
            config.importance_check_late_at,
            config.late_tpe_fraction,
        )
        importance_check_late = max(
            importance_check_late, importance_check_early + _MIN_IMPORTANCE_CHECK_GAP
        )

        logger.debug(
            "[Optuna][{}] Importance checks scheduled at trials {} (early) and {} (late) "
            "(n_startup={}, n_tpe={}, total={})",
            pid,
            importance_check_early,
            importance_check_late,
            n_startup,
            n_tpe,
            total_trials,
        )

        # Schedule: (check_at, threshold_multiplier, ped_anova_quantile, label)
        # Filter out checkpoints that exceed total_trials (can happen for small budgets
        # where _MIN_POST_STARTUP_TRIALS pushes the checkpoint beyond the run).
        _importance_schedule: list[tuple[int, float, float, str]] = []
        for _ck, _tm, _pq, _lb in [
            (
                importance_check_early,
                config.early_threshold_multiplier,
                config.ped_anova_early_quantile,
                "early",
            ),
            (importance_check_late, 1.0, config.ped_anova_late_quantile, "late"),
        ]:
            if _ck < total_trials:
                _importance_schedule.append((_ck, _tm, _pq, _lb))

        if self.config.importance_freezing and self.n_trials < 30:
            logger.warning(
                "[Optuna][{}] n_trials={} is small — importance freezing may be "
                "unreliable (PED-ANOVA needs sufficient completed trials)",
                pid,
                self.n_trials,
            )

        if self.config.importance_freezing and not _importance_schedule:
            logger.debug(
                "[Optuna][{}] Importance checks disabled — both checkpoints ({}, {}) "
                "exceed total_trials ({})",
                pid,
                importance_check_early,
                importance_check_late,
                total_trials,
            )

        frozen_params: dict[str, Any] = {}
        _fired_phases: set[str] = set()
        _importance_lock = asyncio.Lock()
        _ask_lock = asyncio.Lock()

        failure_reasons: list[str] = []
        failure_reasons_set: set[str] = set()
        n_completed = 0
        _completed_lock = asyncio.Lock()

        # Early stopping: cancel remaining trials after `patience` without improvement
        _patience = self.config.early_stopping_patience
        _trials_since_improvement = 0
        _stop_event = asyncio.Event()

        # Trial deduplication: skip re-evaluation of identical param combos
        _seen_params: set[frozenset] = set()

        async def _log_progress() -> None:
            nonlocal n_completed
            async with _completed_lock:
                n_completed += 1

                # Dynamic Feature Importance: two-phase check
                # Uses >= (not ==) so a burst of failures cannot permanently
                # skip a checkpoint.  _fired_phases prevents double-firing.
                for check_at, thresh_mult, ped_quantile, phase in _importance_schedule:
                    if not (
                        self.config.importance_freezing
                        and phase not in _fired_phases
                        and n_completed >= check_at
                        and len(param_specs) >= _MIN_PARAMS_FOR_IMPORTANCE
                    ):
                        continue
                    _fired_phases.add(phase)
                    try:
                        completed_trials = [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                        # PED-ANOVA needs ≥5 trials in the top quantile
                        # bucket.  Compute the per-phase floor dynamically.
                        min_required = max(
                            self.config.min_trials_for_importance,
                            math.ceil(5.0 / ped_quantile),
                        )
                        if len(completed_trials) < min_required:
                            logger.debug(
                                "[Optuna][{}][{}] Skipping importance — only {} "
                                "completed trials (need {} for quantile={})",
                                pid,
                                phase,
                                len(completed_trials),
                                min_required,
                                ped_quantile,
                            )
                            continue
                        importances = optuna.importance.get_param_importances(
                            study,
                            evaluator=PedAnovaImportanceEvaluator(
                                target_quantile=ped_quantile
                            ),
                        )
                        avg_importance = 1.0 / max(len(importances), 1)
                        effective_ratio = (
                            self.config.importance_threshold_ratio * thresh_mult
                        )
                        threshold = avg_importance * effective_ratio

                        # Params missing from PED-ANOVA output have zero
                        # discriminative power — freeze them too.
                        all_param_names = {p.name for p in param_specs}
                        reported_names = set(importances.keys())
                        silent_zero = all_param_names - reported_names

                        async with _importance_lock:
                            for name in silent_zero:
                                if name not in frozen_params:
                                    frozen_val = best_params.get(name)
                                    frozen_params[name] = frozen_val
                                    logger.info(
                                        "[Optuna][{}][{}] Freezing '{}' "
                                        "(absent from PED-ANOVA — zero discriminative power)",
                                        pid,
                                        phase,
                                        name,
                                    )
                            abs_thresh = self.config.importance_absolute_threshold
                            for name, imp in importances.items():
                                if name not in frozen_params and (
                                    imp < threshold or imp < abs_thresh
                                ):
                                    frozen_val = best_params.get(name)
                                    frozen_params[name] = frozen_val
                                    reason = (
                                        f"< abs_thresh={abs_thresh:.4f}"
                                        if imp >= threshold
                                        else f"< rel_thresh={threshold:.4f}"
                                    )
                                    logger.info(
                                        "[Optuna][{}][{}] Freezing '{}' "
                                        "(importance={:.4f}, {})",
                                        pid,
                                        phase,
                                        name,
                                        imp,
                                        reason,
                                    )
                        logger.debug(
                            "[Optuna][{}][{}] Importance done — {} frozen: {}",
                            pid,
                            phase,
                            len(frozen_params),
                            list(frozen_params.keys()),
                        )
                        # If all params are frozen, no search space left —
                        # stop early to avoid wasting budget on pruned duplicates.
                        if len(frozen_params) >= len(param_specs):
                            _stop_event.set()
                            logger.info(
                                "[Optuna][{}] All {} parameters frozen — stopping early",
                                pid,
                                len(param_specs),
                            )
                    except Exception as e:
                        logger.debug(
                            "[Optuna][{}][{}] Importance check failed: {}",
                            pid,
                            phase,
                            e,
                        )

                if (
                    n_completed % _PROGRESS_LOG_INTERVAL == 0
                    or n_completed == total_trials
                ):
                    logger.info(
                        "[Optuna][{}] Progress: {}/{} trials run, best {}={:.{prec}g}",
                        pid,
                        n_completed,
                        total_trials,
                        self.score_key,
                        best_value if best_value is not None else float("nan"),
                        prec=_DEFAULT_PRECISION,
                    )

        async def _run_trial(trial_number: int) -> None:
            nonlocal \
                best_scores, \
                best_value, \
                best_params, \
                best_prog_output, \
                _trials_since_improvement
            trial = None
            k = trial_number + 1
            try:
                if _stop_event.is_set():
                    return
                async with sem:
                    async with _importance_lock:
                        current_frozen = dict(frozen_params)

                    # Enqueue frozen values so suggest_*() records them in
                    # the trial (keeps Optuna's trial data complete for TPE
                    # and importance computation).
                    # _ask_lock ensures enqueue+ask is atomic so concurrent
                    # trials don't steal each other's enqueued values.
                    async with _ask_lock:
                        if current_frozen:
                            study.enqueue_trial(current_frozen)
                        trial = study.ask()

                    values: dict[str, Any] = {}
                    for p in param_specs:
                        if p.param_type == "float":
                            v = trial.suggest_float(p.name, p.low, p.high)
                            if v != 0 and math.isfinite(v):
                                v = float(f"{v:.{_DEFAULT_PRECISION}g}")
                            values[p.name] = v
                        elif p.param_type == "int":
                            values[p.name] = trial.suggest_int(
                                p.name, int(p.low), int(p.high)
                            )
                        elif p.param_type == "log_float":
                            v = trial.suggest_float(p.name, p.low, p.high, log=True)
                            if v != 0 and math.isfinite(v):
                                v = float(f"{v:.{_DEFAULT_PRECISION}g}")
                            values[p.name] = v
                        elif p.param_type == "categorical":
                            values[p.name] = trial.suggest_categorical(
                                p.name, p.choices
                            )

                    logger.trace(
                        "[Optuna][{}][trial {}] Evaluating: {}",
                        pid,
                        trial.number,
                        values,
                    )

                    # Dedup: skip evaluation if we've already seen these params
                    param_key = frozenset(
                        (k_, repr(v_)) for k_, v_ in sorted(values.items())
                    )
                    if param_key in _seen_params:
                        logger.debug(
                            "[Optuna][{}] Trial {}/{} skipped (duplicate params)",
                            pid,
                            k,
                            total_trials,
                        )
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        return
                    _seen_params.add(param_key)

                    status = "random" if trial.number <= n_startup else "TPE"
                    logger.debug(
                        "[Optuna][{}] Trial {}/{} started (evaluating, mode={})",
                        pid,
                        trial.number + 1,
                        total_trials,
                        status,
                    )
                    scores, prog_output, error = await self._evaluate_single(
                        parameterized_code, values, context
                    )

                # After releasing sem — bookkeeping (tell is fast/synchronous)
                if scores is None:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    reason = f"Evaluation failed: {error}"
                    if reason not in failure_reasons_set:
                        failure_reasons_set.add(reason)
                        failure_reasons.append(reason)
                    logger.debug(
                        "[Optuna][{}] Trial {}/{} pruned", pid, k, total_trials
                    )
                else:
                    score = float(scores[self.score_key])
                    study.tell(trial, score)
                    async with _completed_lock:
                        if _is_better(score):
                            best_value = score
                            best_scores = scores
                            best_params = dict(values)
                            best_prog_output = prog_output
                            _trials_since_improvement = 0
                        else:
                            _trials_since_improvement += 1
                            if (
                                _patience is not None
                                and _trials_since_improvement >= _patience
                            ):
                                _stop_event.set()
                                logger.info(
                                    "[Optuna][{}] Early stopping: no improvement for {} trials",
                                    pid,
                                    _patience,
                                )
                    logger.debug(
                        "[Optuna][{}] Trial {}/{} completed, {}={:.{prec}g}",
                        pid,
                        k,
                        total_trials,
                        self.score_key,
                        score,
                        prec=_DEFAULT_PRECISION,
                    )
                await _log_progress()
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if reason not in failure_reasons_set:
                    failure_reasons_set.add(reason)
                    failure_reasons.append(reason)
                if trial is not None:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                logger.debug(
                    "[Optuna][{}] Trial {}/{} failed: {}",
                    pid,
                    k,
                    total_trials,
                    reason,
                )

        # 1. Evaluate baseline (parameterized code with initial values).
        baseline_values = {p.name: p.initial_value for p in param_specs}

        baseline_result, baseline_prog, baseline_err = await self._evaluate_single(
            parameterized_code, baseline_values, context
        )

        if baseline_result is not None:
            baseline_score = float(baseline_result[self.score_key])
            if _is_better(baseline_score):
                best_value = baseline_score
                best_scores = baseline_result
                best_params = dict(baseline_values)
                best_prog_output = baseline_prog
            # Tell the study about the baseline so TPE can learn from it.
            try:
                study.enqueue_trial(baseline_values)
                baseline_trial = study.ask()
                for p in param_specs:
                    if p.param_type == "float":
                        baseline_trial.suggest_float(p.name, p.low, p.high)
                    elif p.param_type == "int":
                        baseline_trial.suggest_int(p.name, int(p.low), int(p.high))
                    elif p.param_type == "log_float":
                        baseline_trial.suggest_float(p.name, p.low, p.high, log=True)
                    elif p.param_type == "categorical":
                        baseline_trial.suggest_categorical(p.name, p.choices)
                study.tell(baseline_trial, baseline_score)
            except Exception as e:
                logger.warning(
                    "[Optuna][{}] Could not record baseline in study: {}",
                    pid,
                    e,
                )
            logger.info(
                "[Optuna][{}] Baseline {}={:.{prec}f}",
                pid,
                self.score_key,
                baseline_score,
                prec=_DEFAULT_PRECISION,
            )
        else:
            # Enhanced logging for baseline failure
            logger.info(
                "[Optuna][{}] Baseline evaluation failed (original parameters invalid). "
                "Proceeding with optimization to find valid parameters.\n"
                "Error details: {}",
                pid,
                baseline_err or "Unknown error (check debug logs)",
            )

        # Run trials: total = n_startup (random) + n_trials (TPE).
        logger.info(
            "[Optuna][{}] Running {} trials total ({} random + {} TPE, up to {} in parallel)...",
            pid,
            total_trials,
            n_startup,
            self.n_trials,
            self.max_parallel,
        )
        tasks = [asyncio.create_task(_run_trial(i)) for i in range(total_trials)]
        await asyncio.gather(*tasks, return_exceptions=True)

        n_complete = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )

        if n_complete == 0:
            reasons_str = "\n".join(
                f"- {r}" for r in failure_reasons[:_MAX_FAILURE_REASONS]
            )
            if len(failure_reasons) > _MAX_FAILURE_REASONS:
                reasons_str += (
                    f"\n- ... and {len(failure_reasons) - _MAX_FAILURE_REASONS} more"
                )

            logger.warning(
                "[Optuna][{}] No trials completed successfully; "
                "returning original code.\nCommon errors:\n{}",
                pid,
                reasons_str,
            )
            return best_params, best_scores, 0, total_trials, best_prog_output

        logger.debug(
            "[Optuna][{}] Best trial: {} {}={}",
            pid,
            best_params,
            self.score_key,
            best_value,
        )

        return best_params, best_scores, n_complete, total_trials, best_prog_output

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    async def compute(self, program: Program) -> OptunaOptimizationOutput:
        """Analyze code with LLM and tune hyperparameters using Optuna.

        Parameters
        ----------
        program : Program
            The program to optimize.

        Returns
        -------
        OptunaOptimizationOutput
            Results including optimized code, best parameters, and trial stats.
        """
        code = program.code
        pid = program.id[:8]

        # 0. Resolve context early (needed for baseline runtime measurement)
        ctx = self.params.context.data if self.params.context is not None else None

        # 1. Measure baseline runtime for the LLM prompt
        baseline_runtime_s = await self._measure_baseline_runtime(code, ctx, pid)

        # 2. LLM analysis
        logger.debug("[Optuna][{}] Analysing code with LLM...", pid)
        try:
            search_space = await self._analyze_code(code, baseline_runtime_s)
            parameterized_code = self._apply_modifications(code, search_space)
        except Exception as exc:
            raise RuntimeError(
                f"[Optuna][{pid}] LLM analysis or patching failed: {exc}"
            ) from exc

        if not search_space.parameters:
            raise ValueError(f"[Optuna][{pid}] LLM found no tuneable parameters")

        param_specs = search_space.parameters
        # parameterized_code is already computed in try-block above
        n = len(param_specs)

        logger.debug(
            "[Optuna][{}] LLM proposed {} parameters: {}",
            pid,
            n,
            [p.name for p in param_specs],
        )
        logger.debug("[Optuna][{}] LLM reasoning: {}", pid, search_space.reasoning)

        # 3. Run Optuna
        (
            best_params,
            best_scores,
            n_complete,
            total_trials,
            best_prog_output,
        ) = await self._run_optuna(parameterized_code, param_specs, ctx, pid)

        if not best_scores:
            raise ValueError(f"[Optuna][{pid}] Optuna produced no usable scores")

        # 4. Build optimised code (desubstitute params into clean code)
        param_types = {p.name: p.param_type for p in param_specs}
        optimized_code = desubstitute_params(
            parameterized_code,
            best_params,
            param_types,
            add_tuned_comment=self.add_tuned_comment,
        )

        # 5. Optionally update program in-place
        if self.update_program_code:
            program.code = optimized_code

        # 6. Summary
        search_summary = [
            {
                "name": p.name,
                "param_type": p.param_type,
                "initial_value": p.initial_value,
                "optimized_value": best_params.get(p.name),
                "low": p.low,
                "high": p.high,
                "choices": p.choices,
            }
            for p in param_specs
        ]

        display_score = (
            float(best_scores[self.score_key])
            if self.score_key in best_scores
            else None
        )
        logger.info(
            "[Optuna][{}] == Done ==  trials={}/{} (+ baseline) params={} {}={}  updated={}",
            pid,
            n_complete,
            total_trials,
            n,
            self.score_key,
            f"{display_score:.{_DEFAULT_PRECISION}f}"
            if display_score is not None
            else "N/A",
            self.update_program_code,
        )

        return OptunaOptimizationOutput(
            optimized_code=optimized_code,
            best_scores=best_scores,
            best_params=best_params,
            n_params=n,
            n_trials=n_complete,
            search_space_summary=search_summary,
            best_program_output=best_prog_output,
        )
