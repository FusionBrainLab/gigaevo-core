"""Mutation operator that asks the LLM for a schema-constrained diff, never raw genome text."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.evolution.mutation.allowed_changes import AllowedChanges
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.exceptions import MutationError
from gigaevo.llm.agents.structured_diff import DiffMutationAgent
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.prompts.fetcher import PromptFetcher


class StructuredDiffMutationOperator(MutationOperator):
    """`AllowedChanges` builds the per-call schema, renders parents, and applies diffs;
    this operator stays genome-agnostic.

    The diff schema carries no archetype field: any archetype the prompt asks the
    LLM to name lives in the diff's prose, so MutationSpec.mutation_archetype
    stays None by design."""

    def __init__(
        self,
        *,
        llm_wrapper,
        allowed_changes: AllowedChanges,
        problem_context: ProblemContext,
        prompts_dir: str | Path | None = None,
        prompt_fetcher: PromptFetcher | None = None,
    ):
        self.allowed_changes = allowed_changes
        self.agent = DiffMutationAgent(
            llm=llm_wrapper,
            allowed_changes=allowed_changes,
            task_description=problem_context.task_description,
            metrics_context=problem_context.metrics_context,
            prompts_dir=prompts_dir,
            prompt_fetcher=prompt_fetcher,
        )
        logger.info(
            "[StructuredDiffMutationOperator] Initialized with {}",
            type(allowed_changes).__name__,
        )

    async def mutate_single(
        self,
        selected_parents: list[Program],
        memory_instructions: str | None = None,
    ) -> MutationSpec | None:
        if not selected_parents:
            logger.warning("[StructuredDiffMutationOperator] No parents provided")
            return None
        parents_map = {
            chr(ord("A") + i): parent.code for i, parent in enumerate(selected_parents)
        }
        try:
            schema = self.allowed_changes.build_schema(parents_map)
        except MutationError:
            raise
        except Exception as e:
            raise MutationError(f"diff_schema_error: {e}") from e
        try:
            result = await self.agent.arun(
                parents=selected_parents, parents_map=parents_map, diff_schema=schema
            )
        except MutationError:
            raise
        except Exception as e:
            raise MutationError(f"llm_call_error: {e}") from e
        metadata: dict = {MutationSpec.META_OUTPUT: result["diff"]}
        model_used = result["metadata"].get("model_used")
        if model_used:
            metadata[MutationSpec.META_MODEL] = model_used
        return MutationSpec(
            code=result["code"],
            parents=selected_parents,
            name="structured_diff",
            metadata=metadata,
        )
