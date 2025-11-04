from typing import Literal, Optional

from loguru import logger

from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.exceptions import MutationError
from gigaevo.llm.agents.mutation import MutationAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program

MutationMode = Literal["rewrite", "diff"]


class LLMMutationOperator(MutationOperator):
    """Mutation operator using LangGraph-based MutationAgent.

    This class maintains backward compatibility while using the new agent architecture.
    All existing interfaces and logging are preserved.
    """

    def __init__(
        self,
        *,
        llm_wrapper: MultiModelRouter,
        mutation_mode: MutationMode = "rewrite",
        fallback_to_rewrite: bool = True,
        context_key: str = MUTATION_CONTEXT_METADATA_KEY,
        problem_context: ProblemContext,
    ):
        self.problem_context = problem_context
        self.llm_wrapper = llm_wrapper
        self.mutation_mode = mutation_mode
        self.fallback_to_rewrite = fallback_to_rewrite
        self.context_key = context_key
        self.metrics_context = problem_context.metrics_context

        metrics_formatter = MetricsFormatter(self.metrics_context)
        metrics_description = metrics_formatter.format_metrics_description()
        self.system_prompt = problem_context.mutation_system_prompt.format(
            task_definition=problem_context.task_description,
            task_hints=problem_context.task_hints,
            metrics_description=metrics_description,
        )
        self.user_prompt_template = problem_context.mutation_user_prompt

        self.agent = MutationAgent(
            llm=llm_wrapper,
            mutation_mode=mutation_mode,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
        )

        logger.info(
            f"[LLMMutationOperator] Initialized with mode: {mutation_mode} "
            "(using LangGraph agent)"
        )

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> Optional[MutationSpec]:
        """Generate a single mutation from the selected parents.

        Args:
            selected_parents: List of parent programs to mutate

        Returns:
            MutationSpec if successful, None if no mutation could be generated
        """
        if not selected_parents:
            logger.warning("[LLMMutationOperator] No parents provided for mutation")
            return None

        try:
            if self.mutation_mode == "diff" and len(selected_parents) != 1:
                raise MutationError(
                    "Diff-based mutation requires exactly 1 parent program"
                )

            logger.debug(
                f"[LLMMutationOperator] Running mutation agent for {len(selected_parents)} parents"
            )

            result = await self.agent.arun(
                input=selected_parents, mutation_mode=self.mutation_mode
            )

            final_code: str = result["code"].strip()
            if not final_code:
                raise MutationError(
                    "Failed to extract code from LLM response. No code found."
                )
            mutation_spec = MutationSpec(
                code=final_code,
                parents=selected_parents,
                name=f"LLM Mutation: {self.mutation_mode} | {self.llm_wrapper.__class__.__name__}",
            )
            return mutation_spec
        except Exception as e:
            raise MutationError(f"Failed to mutate: {e}") from e
