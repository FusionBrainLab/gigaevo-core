from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, TypeVar

from loguru import logger

from src.database.program_storage import ProgramStorage
from src.programs.core_types import ProgramStageResult, StageIO, VoidInput, VoidOutput
from src.programs.program import Program
from src.programs.stages.ancestry_selector import AncestrySelector
from src.programs.stages.base import Stage
from src.programs.stages.common import StringList
from src.programs.stages.stage_registry import StageRegistry

T = TypeVar("T")


class RelatedCollectorBase(Stage):
    """
    Two-phase collector:
      1) _collect_programs(program)  -> list[Program]
      2) _process(program, programs) -> StageIO | ProgramStageResult

    Subclasses set a concrete OutputModel and override the two abstract methods.
    """

    InputsModel = VoidInput
    OutputModel = VoidOutput
    cacheable: bool = False  # lineage-derived sets usually change over time

    def __init__(self, *, storage: ProgramStorage, **kwargs: Any):
        super().__init__(**kwargs)
        self.storage = storage

    @abstractmethod
    async def _collect_programs(self, program: Program) -> list[Program]: ...

    @abstractmethod
    async def _process(
        self, program: Program, programs: list[Program]
    ) -> StageIO | ProgramStageResult: ...

    async def compute(self, program: Program) -> StageIO | ProgramStageResult:
        related = await self._collect_programs(program)
        return await self._process(program, related)


@StageRegistry.register(description="Collect related Program IDs (List[str])")
class ProgramIdsCollector(RelatedCollectorBase):
    OutputModel = StringList

    async def _process(self, program: Program, programs: List[Program]) -> StringList:
        return StringList(items=[p.id for p in programs])


@StageRegistry.register(description="Collect ids of descendant Programs")
class DescendantProgramIds(ProgramIdsCollector):
    cacheable: bool = False

    def __init__(self, *, selector: AncestrySelector, **kwargs: Any):
        super().__init__(**kwargs)
        self.selector = selector

    async def _collect_programs(self, program: Program) -> list[Program]:
        selected = await self.selector.select(
            await self.storage.mget(program.lineage.children)
        )
        logger.info(
            f"[DescendantProgramIds] Selected {len(selected)} programs for {program.id} with children {program.lineage.children}"
        )
        return selected


@StageRegistry.register(description="Collect ids of ancestor Programs")
class AncestorProgramIds(ProgramIdsCollector):
    cacheable: bool = False

    def __init__(self, *, selector: AncestrySelector, **kwargs: Any):
        super().__init__(**kwargs)
        self.selector = selector

    async def _collect_programs(self, program: Program) -> list[Program]:
        selected = await self.selector.select(
            await self.storage.mget(program.lineage.parents)
        )
        logger.info(
            f"[AncestorProgramIds] Selected {len(selected)} programs for {program.id} with parents {program.lineage.parents}"
        )
        return selected
