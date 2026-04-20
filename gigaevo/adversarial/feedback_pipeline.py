"""Adversarial co-evolution pipeline with bidirectional opponent feedback.

Extends AdversarialPipelineBuilder by adding OpponentFeedbackStage, which
injects top-K opponent source codes into the mutation prompt (analogous to
GAN gradient flow between Generator and Discriminator).

Both populations (Constructors and Improvers) see each other's code:
  - Constructor (pop_a): sees top-K Improver codes → OPPONENT ATTACK REPORT
  - Improver (pop_b): sees top-K Constructor codes → TARGET ANALYSIS REPORT

The feedback block replaces FormatterStage's output in the MutationContextStage
'formatted' slot (FormatterStage returns nothing useful for adversarial problems
since evaluate.py does not return an artifact).
"""

from __future__ import annotations

from typing import Literal

from gigaevo.adversarial.feedback_stage import OpponentFeedbackStage
from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.adversarial.pipeline import AdversarialPipelineBuilder
from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency


class AdversarialFeedbackPipelineBuilder(AdversarialPipelineBuilder):
    """Standard adversarial pipeline + OpponentFeedbackStage (bidirectional feedback).

    Inherits AdversarialPipelineBuilder (gets FetchOpponentResultsStage wired
    to CallValidatorFunction), then adds OpponentFeedbackStage wired to
    MutationContextStage.formatted.

    The FormatterStage → MutationContextStage.formatted edge is removed because:
    1. Adversarial evaluate.py returns a tuple (no artifact), so FormatterStage
       always returns ProgramStageResult.skipped() — no useful content.
    2. OpponentFeedbackStage provides richer mutation context (opponent source code).

    Args:
        ctx: Evolution context.
        opponent_provider: Archive provider (shared with FetchOpponentResultsStage).
        opponent_feedback_k: Number of top opponents to show per mutation (K).
        population_role: "constructor" or "improver" — controls report framing.
        n_opponents: Number of opponents for evaluate.py (passed to parent).
        per_opponent_timeout: Timeout per opponent execution (passed to parent).
        fallback_dir: Directory with fallback opponent codes (passed to parent).
        dag_timeout: Total DAG execution timeout.
        stage_timeout: Per-stage execution timeout.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        opponent_provider: OpponentArchiveProvider,
        opponent_feedback_k: int = 3,
        population_role: Literal["constructor", "improver"] = "constructor",
        n_opponents: int = 5,
        per_opponent_timeout: float = 10.0,
        fallback_dir: str = "fallback",
        archive_reeval: bool = True,
        *,
        opponent_result_mode: Literal["exec", "cached"] = "exec",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        opponent_sources: list[dict[str, int | str]] | None = None,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
    ):
        super().__init__(
            ctx,
            opponent_provider,
            n_opponents,
            per_opponent_timeout,
            fallback_dir,
            archive_reeval,
            opponent_result_mode=opponent_result_mode,
            redis_host=redis_host,
            redis_port=redis_port,
            opponent_sources=opponent_sources,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
        )
        # Derive sort direction from the primary metric so we always rank by the
        # right polarity (e.g. a loss metric where lower=better would use reverse=False).
        higher_is_better = (
            ctx.problem_ctx.metrics_context.get_primary_spec().higher_is_better
        )
        self._add_feedback_stages(
            opponent_provider=opponent_provider,
            k=opponent_feedback_k,
            role=population_role,
            stage_timeout=stage_timeout,
            higher_is_better=higher_is_better,
        )

    def _add_feedback_stages(
        self,
        opponent_provider: OpponentArchiveProvider,
        k: int,
        role: Literal["constructor", "improver"],
        stage_timeout: float,
        higher_is_better: bool,
    ) -> None:
        # Remove FormatterStage → MutationContextStage edge (no artifact in adversarial eval)
        self.remove_data_flow_edge("FormatterStage", "MutationContextStage")

        # Add OpponentFeedbackStage
        self.add_stage(
            "OpponentFeedbackStage",
            lambda: OpponentFeedbackStage(
                opponent_provider=opponent_provider,
                k=k,
                role=role,
                higher_is_better=higher_is_better,
                timeout=stage_timeout,
            ),
        )

        # Wire feedback output to MutationContextStage.formatted slot
        self.add_data_flow_edge(
            "OpponentFeedbackStage", "MutationContextStage", "formatted"
        )

        # Run after ValidateCodeStage succeeds (no point fetching feedback for invalid code)
        self.add_exec_dep(
            "OpponentFeedbackStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )
