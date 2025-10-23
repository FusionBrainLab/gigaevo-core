"""Collector stages for aggregating outputs across related programs."""

from abc import abstractmethod
from datetime import datetime

from loguru import logger

from src.programs.core_types import ProgramStageResult, StageState
from src.programs.program import Program
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.database.program_storage import ProgramStorage
from src.llm.agents.lineage import LineageAnalysis


class CollectorStage(Stage):
    """Base class for aggregating stage outputs across related programs.
    
    Collectors fetch related programs (ancestors/descendants) and aggregate
    outputs from a specified source stage.
    
    Subclasses define:
    - _collect_programs(): Which programs to aggregate (children/parents)
    
    Subclasses can optionally add filtering/sorting in _collect_programs().
    """
    
    def __init__(
        self,
        storage: ProgramStorage,
        source_stage_name: str,
        **kwargs
    ):
        """Initialize collector.
        
        Args:
            storage: Program storage for fetching related programs
            source_stage_name: Name of stage whose outputs to aggregate
            **kwargs: Additional Stage arguments (stage_name, timeout, etc.)
        """
        super().__init__(**kwargs)
        self.storage = storage
        self.source_stage_name = source_stage_name
    
    @property
    def cacheable(self) -> bool:
        """Collectors are never cacheable - always fetch fresh family data."""
        return False
    
    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        """Collectors have no mandatory DAG inputs."""
        return []
    
    @abstractmethod
    async def _collect_programs(self, program: Program) -> list[Program]:
        """Fetch related programs (children/parents).
        
        Args:
            program: Current program
            
        Returns:
            List of related programs (can apply filtering/sorting here)
        """
        pass
    
    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute collection and aggregation.
        
        1. Collect related programs via _collect_programs()
        2. Extract outputs from source stage for each related program
        3. Return aggregated list
        """
        # Collect related programs
        related = await self._collect_programs(program)
        
        # Collect outputs from source stage
        outputs = []
        for prog in related:
            result = prog.stage_results.get(self.source_stage_name)
            if result and result.output:
                outputs.append(result.output)
        
        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output=outputs,
            stage_name=self.stage_name
        )


class DescendantCollector(CollectorStage):
    """Collects stage outputs from all descendants (children)."""
    
    async def _collect_programs(self, program: Program) -> list[Program]:
        """Fetch all descendants from program.lineage.children."""
        descendants = []
        
        if not program.lineage.children:
            logger.info(
                f"[{self.stage_name}] No descendants found for {program.id[:8]} (children list is empty)"
            )
            return descendants
            
        logger.info(
            f"[{self.stage_name}] Collecting {len(program.lineage.children)} descendants for {program.id[:8]}"
        )
        
        for child_id in program.lineage.children:
            child = await self.storage.get(child_id)
            if child:
                descendants.append(child)
                
        logger.info(
            f"[{self.stage_name}] Collected {len(descendants)} descendants for {program.id[:8]}"
        )
        return descendants
    
    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute collection and aggregation for descendants.
        
        Returns dict mapping child_id to their lineage analyses where current program is the parent.
        Each child's lineage stage output is a dict[parent_id, LineageAnalysis]
        where we extract the analysis for current program as parent.
        """
        # Collect related programs
        related = await self._collect_programs(program)
        
        if not related:
            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output={},
                stage_name=self.stage_name
            )
        
        # Collect outputs from source stage and build dict mapping child_id to their analysis
        child_analyses: dict[str, LineageAnalysis] = {}
        current_program_id = program.id
        
        for child_program in related:
            stage_result = child_program.stage_results.get(self.source_stage_name)
            if stage_result and stage_result.output:
                result: dict[str, LineageAnalysis] = stage_result.output
                if current_program_id in result:
                    child_analyses[child_program.id] = result[current_program_id]
        
        logger.info(
            f"[{self.stage_name}] Extracted {len(child_analyses)} lineage analyses for {program.id[:8]}"
        )
        
        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output=child_analyses,
            stage_name=self.stage_name
        )


class AncestorCollector(CollectorStage):
    """Collects stage outputs from all ancestors (parents)."""
    
    async def _collect_programs(self, program: Program) -> list[Program]:
        """Fetch all ancestors from program.lineage.parents."""
        ancestors = []
        if program.lineage and program.lineage.parents:
            for parent_id in program.lineage.parents:
                parent = await self.storage.get(parent_id)
                if parent:
                    ancestors.append(parent)
        return ancestors

