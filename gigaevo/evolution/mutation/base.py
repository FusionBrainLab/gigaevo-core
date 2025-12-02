from abc import ABC, abstractmethod
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.programs.program import Program


class MutationSpec(BaseModel):
    """Container for a single mutation result returned by a `MutationOperator`."""

    code: str = Field(description="The code of the mutated program")
    parents: list[Program] = Field(description="List of parent programs")
    name: str = Field(description="Description of the mutation")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured mutation metadata (archetype, justification, etc.)",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __iter__(self) -> Iterable:
        """Allow easy unpacking: ``code, parents, name = spec``."""
        return iter((self.code, self.parents, self.name))


class MutationOperator(ABC):
    """Abstract mutation operator that produces child programs from parents."""

    @abstractmethod
    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        """Generate a single mutation from the selected parents.

        Args:
            selected_parents: List of parent programs to mutate

        Returns:
            MutationSpec if successful, None if no mutation could be generated
        """
