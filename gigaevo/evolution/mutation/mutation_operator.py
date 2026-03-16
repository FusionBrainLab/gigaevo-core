from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
    from gigaevo.prompts.fetcher import PromptFetcher

MutationMode = Literal["rewrite", "diff"]

#: Metadata key for storing the prompt tracking ID in program metadata
PROMPT_ID_METADATA_KEY = "prompt_id"


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
        prompt_fetcher: PromptFetcher | None = None,
    ):
        self.problem_context = problem_context
        self.llm_wrapper = llm_wrapper
        self.mutation_mode = mutation_mode
        self.fallback_to_rewrite = fallback_to_rewrite
        self.context_key = context_key
        self.metrics_context = problem_context.metrics_context
        self.strip_comments_and_docstrings = strip_comments_and_docstrings
        self._prompt_fetcher = prompt_fetcher

        self.agent = create_mutation_agent(
            llm=llm_wrapper,
            task_description=problem_context.task_description,
            metrics_context=self.metrics_context,
            mutation_mode=mutation_mode,
            prompts_dir=prompts_dir,
            prompt_fetcher=prompt_fetcher,
        )

        logger.info(
            "[LLMMutationOperator] Initialized with mode: {}, "
            "strip_comments_and_docstrings: {} "
            "(using LangGraph agent)",
            mutation_mode,
            strip_comments_and_docstrings,
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
                "[LLMMutationOperator] Failed to canonicalize code due to syntax error: {}. "
                "Returning original code.",
                e,
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
                "[LLMMutationOperator] Running mutation agent for {} parents",
                len(selected_parents),
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
                logger.debug("[LLMMutationOperator] Mutation archetype: {}", archetype)
            if model_name:
                mutation_metadata["mutation_model"] = model_name
            # Stamp prompt tracking ID if present
            prompt_id = result.get("prompt_id")
            if prompt_id is not None:
                mutation_metadata[PROMPT_ID_METADATA_KEY] = prompt_id

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
        """Fetch parents and forward to the router's mutation outcome callback.

        Also records prompt outcome stats for the prompt co-evolution run,
        if a dynamic prompt_fetcher is configured.
        """
        parent_ids = program.lineage.parents
        parents: list[Program] = []
        if parent_ids:
            parents = await storage.mget(parent_ids)
            self.llm_wrapper.on_mutation_outcome(
                program, [p for p in parents if p], outcome=outcome
            )

        # Record prompt outcome for co-evolution stats tracking
        _fetcher = getattr(self, "_prompt_fetcher", None)
        if _fetcher is not None and _fetcher.is_dynamic and outcome is not None:
            prompt_id = program.metadata.get(PROMPT_ID_METADATA_KEY)
            if prompt_id:
                primary_key = self.metrics_context.get_primary_key()
                higher_is_better = self.metrics_context.is_higher_better(primary_key)
                child_fitness = program.metrics.get(primary_key, 0.0)
                parent_fitness_values = [
                    p.metrics.get(primary_key, 0.0)
                    for p in parents
                    if p and primary_key in p.metrics
                ]
                best_parent_fitness = (
                    max(parent_fitness_values) if parent_fitness_values else 0.0
                )
                _fetcher.record_outcome(
                    prompt_id=prompt_id,
                    child_fitness=child_fitness,
                    parent_fitness=best_parent_fitness,
                    higher_is_better=higher_is_better,
                    outcome=outcome,
                )
