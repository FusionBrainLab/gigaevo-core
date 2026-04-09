"""Adversarial co-evolution pipeline builder.

Extends DefaultPipelineBuilder — adds two adversarial stages:

  FetchOpponentIdsStage (NO_CACHE)
    Samples fresh opponent IDs on every DAG run.

  FetchOpponentResultsStage (InputHashCache)
    Executes opponent code; reruns only when opponent IDs change.
    Cache key = hash of opponent IDs → automatic re-evaluation when
    the archive changes, without any engine-level hooks.

CallValidatorFunction is reconfigured to call evaluate.py (not validate.py).
"""

from __future__ import annotations

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.adversarial.stages import FetchOpponentIdsStage, FetchOpponentResultsStage
from gigaevo.entrypoint.constants import (
    DEFAULT_SIMPLE_STAGE_TIMEOUT,
    MAX_MEMORY_MB,
    MAX_OUTPUT_SIZE,
)
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.stages.python_executors.execution import CallValidatorFunction


class AdversarialPipelineBuilder(DefaultPipelineBuilder):
    """Standard pipeline + two adversarial stages for co-evolution.

    Inherits DefaultPipelineBuilder (gets all standard stages + edges + deps).
    Adds FetchOpponentIdsStage → FetchOpponentResultsStage wired as context
    to CallValidatorFunction.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        opponent_provider: OpponentArchiveProvider,
        n_opponents: int = 5,
        per_opponent_timeout: float = 10.0,
        fallback_dir: str = "fallback",
        archive_reeval: bool = True,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
    ):
        super().__init__(ctx, dag_timeout=dag_timeout, stage_timeout=stage_timeout)
        fallback_codes = self._load_fallback_codes(fallback_dir)
        self._add_adversarial_stages(
            opponent_provider,
            n_opponents,
            per_opponent_timeout,
            fallback_codes,
            archive_reeval,
        )

    def _load_fallback_codes(self, fallback_dir: str) -> list[str]:
        d = self.ctx.problem_ctx.problem_dir / fallback_dir
        if not d.exists():
            logger.debug("[AdversarialPipeline] no fallback dir: {}", d)
            return []
        codes = [f.read_text() for f in sorted(d.glob("*.py"))]
        logger.info(
            "[AdversarialPipeline] loaded {} fallback opponents from {}", len(codes), d
        )
        return codes

    def _add_adversarial_stages(
        self,
        provider: OpponentArchiveProvider,
        n_opponents: int,
        per_opponent_timeout: float,
        fallback_codes: list[str],
        archive_reeval: bool = True,
    ) -> None:
        problem_dir = self.ctx.problem_ctx.problem_dir
        stage_timeout = self._stage_timeout

        # Replace CallValidatorFunction to use evaluate.py instead of validate.py
        evaluate_path = problem_dir / "evaluate.py"
        self.replace_stage(
            "CallValidatorFunction",
            lambda: CallValidatorFunction(
                path=evaluate_path,
                function_name="evaluate",
                timeout=stage_timeout,
                max_memory_mb=MAX_MEMORY_MB,
                max_output_size=MAX_OUTPUT_SIZE,
            ),
        )

        # Stage 1: FetchOpponentIdsStage — always fresh, NO_CACHE
        self.add_stage(
            "FetchOpponentIdsStage",
            lambda: FetchOpponentIdsStage(
                opponent_provider=provider,
                n_opponents=n_opponents,
                timeout=stage_timeout,
            ),
        )

        # Stage 2: FetchOpponentResultsStage
        # archive_reeval=True  → InputHashCache (reruns only when opponent IDs change)
        # archive_reeval=False → NO_CACHE (always reruns — adversarial-v2 baseline)
        total_timeout = per_opponent_timeout * n_opponents + 30
        _archive_reeval = archive_reeval  # capture for lambda closure
        self.add_stage(
            "FetchOpponentResultsStage",
            lambda: FetchOpponentResultsStage(
                opponent_provider=provider,
                n_opponents=n_opponents,
                fallback_codes=fallback_codes,
                per_opponent_timeout=per_opponent_timeout,
                python_path=[problem_dir.resolve()],
                max_memory_mb=MAX_MEMORY_MB,
                timeout=total_timeout,
                archive_reeval=_archive_reeval,
            ),
        )

        # Wire IDs → eval → validator
        self.add_data_flow_edge(
            "FetchOpponentIdsStage", "FetchOpponentResultsStage", "opponent_ids"
        )
        self.add_data_flow_edge(
            "FetchOpponentResultsStage", "CallValidatorFunction", "context"
        )

        # FetchOpponentIdsStage runs after ValidateCodeStage (parallel with CallProgramFunction)
        self.add_exec_dep(
            "FetchOpponentIdsStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )
        # FetchOpponentResultsStage depends on FetchOpponentIdsStage (via data flow)
        self.add_exec_dep(
            "FetchOpponentResultsStage",
            ExecutionOrderDependency.on_success("FetchOpponentIdsStage"),
        )
