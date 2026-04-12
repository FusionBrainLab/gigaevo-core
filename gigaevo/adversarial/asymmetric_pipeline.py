"""Adversarial asymmetric pipeline builder.

Extends AdversarialPipelineBuilder:
  - D runs: adds SourceCodeInjectionStage (forked from FetchOpponentIdsStage)
  - G runs, Arm C: adds GradientInPromptStage (reads D's archive)
  - G runs, Arm A: composition injection handled as pre-step hook (not pipeline)

Key data flow (D runs):
  FetchOpponentIdsStage → FetchOpponentResultsStage (evaluation)
  FetchOpponentIdsStage → SourceCodeInjectionStage (mutation prompt)

Parametric: n_opponents=k, source_prompt_k=l (l<=k).
"""

from __future__ import annotations

from typing import Literal

from loguru import logger

from gigaevo.adversarial.gradient_prompt import GradientInPromptStage
from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.adversarial.pipeline import AdversarialPipelineBuilder
from gigaevo.adversarial.source_injection import SourceCodeInjectionStage
from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency


class AdversarialAsymmetricPipelineBuilder(AdversarialPipelineBuilder):
    """Asymmetric adversarial pipeline with source injection and gradient feedback.

    For D (Improver) runs:
      - SourceCodeInjectionStage receives opponent IDs from FetchOpponentIdsStage
        (same IDs used for evaluation) and shows top-l source codes in D's
        mutation prompt.

    For G (Constructor) runs with feedback_mode="gradient_in_prompt" (Arm C):
      - GradientInPromptStage reads D's best from D's archive and shows it
        in G's mutation prompt.

    For G (Constructor) runs with feedback_mode="composition" (Arm A):
      - No pipeline-level changes. CompositionInjectionHook is wired at the
        engine level (not here).

    Args:
        ctx: Evolution context.
        opponent_provider: Archive provider for the opponent population.
        d_provider: Archive provider for D's population (needed for Arm C G runs).
        population_role: "constructor" or "improver".
        feedback_mode: "composition" (Arm A) or "gradient_in_prompt" (Arm C).
        n_opponents: Number of opponents for evaluation (k).
        source_prompt_k: Number of source codes to show in D's prompt (l <= k).
        per_opponent_timeout: Timeout per opponent execution.
        fallback_dir: Directory with fallback opponent codes.
        archive_reeval: Whether to use InputHashCache for opponent results.
        dag_timeout: Total DAG execution timeout.
        stage_timeout: Per-stage execution timeout.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        opponent_provider: OpponentArchiveProvider,
        d_provider: OpponentArchiveProvider | None = None,
        population_role: Literal["constructor", "improver"] = "constructor",
        feedback_mode: Literal["composition", "gradient_in_prompt"] = "composition",
        n_opponents: int = 1,
        source_prompt_k: int = 1,
        per_opponent_timeout: float = 10.0,
        fallback_dir: str = "fallback",
        archive_reeval: bool = False,
        *,
        dag_timeout: float = 7200.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
    ):
        super().__init__(
            ctx,
            opponent_provider,
            n_opponents,
            per_opponent_timeout,
            fallback_dir,
            archive_reeval,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
        )

        if population_role == "improver":
            self._add_source_injection(
                opponent_provider, source_prompt_k, stage_timeout
            )

        if population_role == "constructor" and feedback_mode == "gradient_in_prompt":
            if d_provider is None:
                raise ValueError(
                    "gradient_in_prompt feedback mode requires d_provider "
                    "(OpponentArchiveProvider pointing to D's archive)"
                )
            self._add_gradient_prompt(d_provider, stage_timeout)

        logger.info(
            "[AsymmetricPipeline] role={} feedback={} n_opp={} source_prompt_k={}",
            population_role,
            feedback_mode,
            n_opponents,
            source_prompt_k,
        )

    def _add_source_injection(
        self,
        opponent_provider: OpponentArchiveProvider,
        source_prompt_k: int,
        stage_timeout: float,
    ) -> None:
        self.remove_data_flow_edge("FormatterStage", "MutationContextStage")
        self.add_stage(
            "SourceCodeInjectionStage",
            lambda: SourceCodeInjectionStage(
                opponent_provider=opponent_provider,
                source_prompt_k=source_prompt_k,
                timeout=stage_timeout,
            ),
        )
        self.add_data_flow_edge(
            "FetchOpponentIdsStage", "SourceCodeInjectionStage", "opponent_ids"
        )
        self.add_data_flow_edge(
            "SourceCodeInjectionStage", "MutationContextStage", "formatted"
        )
        self.add_exec_dep(
            "SourceCodeInjectionStage",
            ExecutionOrderDependency.on_success("FetchOpponentIdsStage"),
        )

    def _add_gradient_prompt(
        self,
        d_provider: OpponentArchiveProvider,
        stage_timeout: float,
    ) -> None:
        self.remove_data_flow_edge("FormatterStage", "MutationContextStage")
        self.add_stage(
            "GradientInPromptStage",
            lambda: GradientInPromptStage(
                opponent_provider=d_provider,
                timeout=stage_timeout,
            ),
        )
        self.add_data_flow_edge(
            "GradientInPromptStage", "MutationContextStage", "formatted"
        )
        self.add_exec_dep(
            "GradientInPromptStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )
