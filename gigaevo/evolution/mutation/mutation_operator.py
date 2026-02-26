from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Literal
from pathlib import Path

from loguru import logger

from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.evolution.mutation.utils import _DocstringRemover
from gigaevo.exceptions import MutationError
from gigaevo.llm.agents.factories import create_mutation_agent
from gigaevo.llm.agents.mutation import MUTATION_OUTPUT_METADATA_KEY
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.llm.bandit import MutationOutcome

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
        strip_comments_and_docstrings: bool = False,
        prompts_dir: str | Path | None = None,
    ):
        self.problem_context = problem_context
        self.llm_wrapper = llm_wrapper
        self.mutation_mode = mutation_mode
        self.fallback_to_rewrite = fallback_to_rewrite
        self.context_key = context_key
        self.metrics_context = problem_context.metrics_context
        self.strip_comments_and_docstrings = strip_comments_and_docstrings

        self.agent = create_mutation_agent(
            llm=llm_wrapper,
            task_description=problem_context.task_description,
            metrics_context=self.metrics_context,
            mutation_mode=mutation_mode,
            prompts_dir=prompts_dir,
        )

        logger.info(
            f"[LLMMutationOperator] Initialized with mode: {mutation_mode}, "
            f"strip_comments_and_docstrings: {strip_comments_and_docstrings} "
            "(using LangGraph agent)"
        )

    @staticmethod
    def _canonicalize_code(code: str) -> str:
        """Remove comments and docstrings from Python code.

        Args:
            code: Python source code as string

        Returns:
            Canonicalized code with comments and docstrings removed
        """
        try:
            tree = ast.parse(code)
            remover = _DocstringRemover()
            tree = remover.visit(tree)
            canonicalized = ast.unparse(tree)
            return canonicalized
        except SyntaxError as e:
            logger.warning(
                f"[LLMMutationOperator] Failed to canonicalize code due to syntax error: {e}. "
                "Returning original code."
            )
            return code

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
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

            # Capture model name (works for both standard and bandit routers)
            model_name = self.llm_wrapper.get_last_model()

            final_code: str = result["code"].strip()
            if not final_code:
                raise MutationError(
                    "Failed to extract code from LLM response. No code found."
                )

            # Canonicalize code if requested
            if self.strip_comments_and_docstrings:
                logger.debug(
                    "[LLMMutationOperator] Canonicalizing code (removing comments and docstrings)"
                )
                final_code = self._canonicalize_code(final_code)

            # Extract structured mutation metadata
            structured_output = result.get("structured_output")
            mutation_metadata: dict[str, object] = {}
            if structured_output:
                mutation_metadata[MUTATION_OUTPUT_METADATA_KEY] = structured_output
                archetype = result.get("archetype", "unknown")
                logger.debug(f"[LLMMutationOperator] Mutation archetype: {archetype}.")
            if model_name:
                mutation_metadata["mutation_model"] = model_name

            mutation_spec = MutationSpec(
                code=final_code,
                parents=selected_parents,
                name=f"LLM Mutation: {self.mutation_mode} | {self.llm_wrapper.__class__.__name__}",
                metadata=mutation_metadata,
            )
            return mutation_spec
        except Exception as e:
            raise MutationError(f"Failed to mutate: {e}") from e

    async def on_program_ingested(
        self,
        program: Program,
        storage: ProgramStorage,
        outcome: MutationOutcome | None = None,
    ) -> None:
        """Fetch parents and forward to the router's mutation outcome callback."""
        parent_ids = program.lineage.parents
        if not parent_ids:
            return
        parents = await storage.mget(parent_ids)
        self.llm_wrapper.on_mutation_outcome(
            program, [p for p in parents if p], outcome=outcome
        )
