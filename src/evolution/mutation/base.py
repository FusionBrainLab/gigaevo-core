from abc import ABC, abstractmethod
import asyncio
from typing import List, Optional, Iterable
from pydantic import ConfigDict
from pydantic import BaseModel

from src.programs.program import Program
from src.evolution.mutation.parent_selector import ParentSelector


class MutationSpec(BaseModel):
    """Container for a single mutation result returned by a `MutationOperator`."""

    code: str # the code of the mutated program
    parents: List[Program] # list of programs that were mutated to produce this one
    name: str # description of the mutation
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __iter__(self) -> Iterable:
        """Allow easy unpacking: ``code, parents, name = spec``."""
        return iter((self.code, self.parents, self.name))


class MutationOperator(ABC):
    """Abstract mutation operator that produces child programs from parents."""

    @abstractmethod
    async def mutate_single(self, available_parents: List[Program], parent_selector: ParentSelector) -> Optional[MutationSpec]:
        """Generate a single mutation using the parent selector.
        
        Args:
            available_parents: List of parent programs available for mutation
            parent_selector: Strategy for selecting which parents to use
            
        Returns:
            MutationSpec if successful, None if no mutation could be generated
        """
        pass

