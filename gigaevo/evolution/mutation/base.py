from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.llm.bandit import MutationOutcome


class MutationSpec(BaseModel):
    """Container for a single mutation result returned by a `MutationOperator`."""

    # Canonical metadata key names — use these instead of bare strings.
    META_MODEL: ClassVar[str] = "mutation_model"
    META_OUTPUT: ClassVar[str] = "mutation_output"
    META_PROMPT_ID: ClassVar[str] = "prompt_id"

    code: str = Field(description="The code of the mutated program")
    parents: list[Program] = Field(description="List of parent programs")
    name: str = Field(description="Description of the mutation")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured mutation metadata (archetype, justification, etc.)",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def mutation_model(self) -> str | None:
        """Model used to generate this mutation, or None."""
        return self.metadata.get(self.META_MODEL)

    @property
    def mutation_archetype(self) -> str | None:
        """Archetype label from structured LLM output, or None."""
        output = self.metadata.get(self.META_OUTPUT)
        if isinstance(output, dict):
            return output.get("archetype")
        return None

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        """Allow easy unpacking: ``code, parents, name = spec``."""
        return iter((self.code, self.parents, self.name))


class MutationOperator(ABC):
    """Abstract mutation operator that produces child programs from parents."""

    @abstractmethod
    async def mutate_single(
        self,
        selected_parents: list[Program],
        memory_instructions: str | None = None,
    ) -> MutationSpec | None:
        """Generate a single mutation from the selected parents.

        Args:
            selected_parents: List of parent programs to mutate
            memory_instructions: Optional memory text to guide mutation

        Returns:
            MutationSpec if successful, None if no mutation could be generated
        """

    async def on_program_ingested(
        self,
        program: Program,
        storage: ProgramStorage,
        outcome: MutationOutcome | None = None,
    ) -> None:
        """Hook called after a mutated program completes evaluation.

        Called for **every** outcome (accepted, rejected by strategy, rejected
        by acceptor) so the mutation operator can provide feedback (e.g. bandit
        reward).  Default is a no-op.
        """
