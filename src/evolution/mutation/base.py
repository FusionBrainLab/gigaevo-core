from abc import ABC, abstractmethod
import asyncio
from typing import List, Optional, Iterable
from pydantic import ConfigDict
from pydantic import BaseModel

from src.programs.program import Program



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
    async def mutate_batch(self, parents: list[Program]) -> list[MutationSpec]:
        """Return child programs derived from *parents*."""


class DummyMutationOperator(MutationOperator):
    async def mutate_batch(self, parents: List[Program]) -> List[MutationSpec]:
        """Return trivial children that append a comment."""
        children: List[Program] = []
        for p in parents:
            new_code = p.code + "\n# mutated"
            children.append(Program.from_mutation_spec(MutationSpec(code=new_code, parents=[p], name="append comment")))
        return children
