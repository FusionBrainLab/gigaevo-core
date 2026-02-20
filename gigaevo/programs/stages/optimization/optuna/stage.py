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
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import optuna

from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.optimization.optuna.desubstitution import (
    _coerce_params,
    _reindent_to_match_block,
    _strip_line_number_prefix,
    desubstitute_params,
)
from gigaevo.programs.stages.optimization.optuna.models import (
    _DEFAULT_PRECISION,
    _OPTUNA_PARAMS_NAME,
    OptunaOptimizationConfig,
    OptunaOptimizationOutput,
    OptunaSearchSpace,
    ParamSpec,
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
            replacement_lines = _strip_line_number_prefix(replacement_lines)
            # Re-indent to match the original block so we never get "unexpected indent"
            original_block = lines[start_idx:end_idx]
            replacement_lines = _reindent_to_match_block(
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

    async def _analyze_code(self, code: str) -> OptunaSearchSpace:
        """Call the LLM to propose a search space for *code*.

        Parameters
        ----------
        code : str
            The source code to analyze.

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
            task_section = (
                f"\n**Task description** (the metric to optimize is `{self.score_key}`):\n"
                f"{self.task_description}\n"
            )

        user_msg = _USER_PROMPT_TEMPLATE.format(
            numbered_code=numbered_code,
            task_description_section=task_section,
        )

        structured_llm = self.llm.with_structured_output(OptunaSearchSpace)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(score_key=self.score_key)),
            HumanMessage(content=user_msg),
        ]
        result = await structured_llm.ainvoke(messages)
        return result

    # ------------------------------------------------------------------
    # Phase 2: Optuna evaluation
    # ------------------------------------------------------------------

    def _build_eval_code(self, parameterized_code: str, params: dict[str, Any]) -> str:
        """Compose a self-contained script: params dict + program + validator.

        Parameters
        ----------
        parameterized_code : str
            The code containing ``_optuna_params`` references.
        params : dict[str, Any]
            The specific parameter values to inject for this evaluation.

        Returns
        -------
        str
            A complete Python script ready for execution.
        """
        # Coerce int-like strings so range(k) etc. work when k comes from categorical/initial_value
        params = _coerce_params(params)
        return build_eval_code(
            validator_code=self._validator_code,
            program_code=parameterized_code,
            function_name=self.function_name,
            validator_fn=self.validator_fn,
            eval_fn_name="_optuna_eval",
            preamble_lines=[f"{_OPTUNA_PARAMS_NAME} = {params!r}"],
        )

    async def _evaluate_single(
        self,
        parameterized_code: str,
        params: dict[str, Any],
        context: Optional[dict[str, Any]],
    ) -> tuple[Optional[dict[str, float]], Optional[str]]:
        """Run one trial and return (score_dict, error_message).

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
        tuple[Optional[dict[str, float]], Optional[str]]
            A tuple of (scores, error_message).
        """
        eval_code = self._build_eval_code(parameterized_code, params)
        return await evaluate_single(
            eval_code=eval_code,
            eval_fn_name="_optuna_eval",
            context=context,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="Optuna",
        )

    async def _run_optuna(
        self,
        parameterized_code: str,
        param_specs: list[ParamSpec],
        context: Optional[dict[str, Any]],
        pid: str,
    ) -> tuple[dict[str, Any], dict[str, float], int, int]:
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
        tuple[dict[str, Any], dict[str, float], int, int]
            Best parameters, best scores, number of successful trials, and total trials run.
        """
        direction = "minimize" if self.minimize else "maximize"

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # TPE with configurable startup trials and multivariate
        from_config = self.config.n_startup_trials is not None
        n_startup = (
            self.config.n_startup_trials
            if from_config
            else min(25, max(10, self.n_trials // 2))
        )
        # Total trials = startup (random) + n_trials (TPE); startup trials are extra, not counted in n_trials.
        total_trials = n_startup + self.n_trials
        logger.debug(
            "[Optuna][{}] TPE sampler: n_startup_trials={} ({}), total_trials={} ({} + {} TPE)",
            pid,
            n_startup,
            "from config" if from_config else "default min(25, max(10, n_trials//2))",
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

        def _is_better(score: float) -> bool:
            if best_value is None:
                return True
            if direction == "minimize":
                return score < best_value
            return score > best_value

        # Run importance only after TPE phase has produced at least one completed trial.
        importance_check_at = (
            self.config.importance_check_at
            if self.config.importance_check_at is not None
            else max(10, total_trials // 3)
        )
        importance_check_at = max(importance_check_at, n_startup + 1)
        frozen_params: dict[str, Any] = {}
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

                # Dynamic Feature Importance: freeze unimportant parameters
                if (
                    self.config.importance_freezing
                    and n_completed == importance_check_at
                    and len(param_specs) > 3
                ):
                    try:
                        completed_trials = [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                        if (
                            len(completed_trials)
                            >= self.config.min_trials_for_importance
                        ):
                            importances = optuna.importance.get_param_importances(study)
                            # Only freeze if the parameter is statistically insignificant
                            # (i.e., its importance is a tiny fraction of the average expected importance)
                            avg_importance = 1.0 / len(importances)
                            threshold = (
                                avg_importance * self.config.importance_threshold_ratio
                            )

                            async with _importance_lock:
                                for name, imp in importances.items():
                                    if (
                                        imp < threshold
                                        or imp
                                        < self.config.importance_absolute_threshold
                                    ):
                                        # Freeze at best-so-far value
                                        frozen_val = best_params.get(
                                            name,
                                            next(
                                                p.initial_value
                                                for p in param_specs
                                                if p.name == name
                                            ),
                                        )
                                        frozen_params[name] = frozen_val
                                        logger.info(
                                            "[Optuna][{}] Freezing low-impact parameter '{}' (importance={:.3f}, thresh={:.3f}) at best-so-far",
                                            pid,
                                            name,
                                            imp,
                                            threshold,
                                        )
                    except Exception as e:
                        logger.debug("[Optuna][{}] Importance check failed: {}", pid, e)

                if n_completed % 10 == 0 or n_completed == total_trials:
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
            nonlocal best_scores, best_value, best_params, _trials_since_improvement
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
                    scores, error = await self._evaluate_single(
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

        baseline_eval_code = self._build_eval_code(parameterized_code, baseline_values)
        baseline_result, baseline_err = await evaluate_single(
            eval_code=baseline_eval_code,
            eval_fn_name="_optuna_eval",
            context=context,
            score_key=self.score_key,
            python_path=self.python_path,
            timeout=self.eval_timeout,
            max_memory_mb=self.max_memory_mb,
            log_tag="Optuna",
        )

        if baseline_result is not None:
            baseline_score = float(baseline_result[self.score_key])
            if _is_better(baseline_score):
                best_value = baseline_score
                best_scores = baseline_result
                best_params = dict(baseline_values)
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
            reasons_str = "\n".join(f"- {r}" for r in failure_reasons[:5])
            if len(failure_reasons) > 5:
                reasons_str += f"\n- ... and {len(failure_reasons) - 5} more"

            logger.warning(
                "[Optuna][{}] No trials completed successfully; "
                "returning original code.\nCommon errors:\n{}",
                pid,
                reasons_str,
            )
            return best_params, best_scores, 0, total_trials

        logger.debug(
            "[Optuna][{}] Best trial: {} {}={}",
            pid,
            best_params,
            self.score_key,
            best_value,
        )

        return best_params, best_scores, n_complete, total_trials

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

        # 1. LLM analysis
        logger.debug("[Optuna][{}] Analysing code with LLM...", pid)
        try:
            search_space = await self._analyze_code(code)
            parameterized_code = self._apply_modifications(code, search_space)
        except Exception as exc:
            logger.warning(
                "[Optuna][{}] LLM analysis or patching failed: {}; returning original code",
                pid,
                exc,
            )
            return OptunaOptimizationOutput(
                optimized_code=code,
                best_scores={},
                best_params={},
                n_params=0,
                n_trials=0,
                search_space_summary=[],
            )

        if not search_space.parameters:
            logger.info(
                "[Optuna][{}] LLM found no tuneable parameters; "
                "returning original code.",
                pid,
            )
            return OptunaOptimizationOutput(
                optimized_code=code,
                best_scores={},
                best_params={},
                n_params=0,
                n_trials=0,
                search_space_summary=[],
            )

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

        # 2. Resolve context
        ctx = self.params.context.data if self.params.context is not None else None

        # 3. Run Optuna
        best_params, best_scores, n_complete, total_trials = await self._run_optuna(
            parameterized_code, param_specs, ctx, pid
        )

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
        )
