from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.evolution.mutation.context import (
    CompositeMutationContext,
    FamilyTreeMutationContext,
    InsightsMutationContext,
    MetricsMutationContext,
    MutationContext,
)
import random
from src.llm.agents.insights import ProgramInsights
from src.llm.agents.lineage import LineageAnalysis
from src.programs.core_types import ProgramStageResult, StageState
from src.programs.metrics.context import MetricsContext
from src.programs.metrics.formatter import MetricsFormatter
from src.programs.program import Program
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.llm.agents.insights import ProgramInsights
from src.exceptions import StageError
from src.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY

if TYPE_CHECKING:
    from src.database.program_storage import ProgramStorage

class MutationContextStage(Stage):
    """Assembles mutation context from available upstream stage outputs.
    
    Automatically aggregates whatever data is available: metrics, insights, lineage.
    Fully extensible - new context types can be added without modifying this class.
    
    This stage is marked as non-cacheable because it depends on lineage data
    which can change over time as new descendants are created.
    """
    
    def __init__(
        self,
        storage: "ProgramStorage",
        metrics_context: MetricsContext,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.storage = storage
        self.metadata_key = MUTATION_CONTEXT_METADATA_KEY
        self.metrics_context = metrics_context
    
    @property
    def cacheable(self) -> bool:
        return False
    
    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return []
    
    @classmethod
    def optional_inputs(cls) -> list[str]:
        return [
            "metrics",
            "insights",
            "lineage_insights",
            "descendant_lineages",
        ]
    
    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        contexts: list[MutationContext] = []
        
        # Metrics context
        metrics_output = self.get_input_optional("metrics")
        if metrics_output and isinstance(metrics_output, dict):
            metrics_keys = metrics_output.get("metrics_keys", [])
            actual_metrics = {k: v for k, v in metrics_output.items() if k in metrics_keys}
            
            if actual_metrics:
                formatter = MetricsFormatter(self.metrics_context)
                contexts.append(MetricsMutationContext(
                    metrics=actual_metrics,
                    metrics_formatter=formatter
                ))
        
        # Insights context
        insights_output = self.get_input_optional("insights")
        if not isinstance(insights_output, ProgramInsights):
            raise StageError(f"Insights output for {program.id} is not a {ProgramInsights.__class__.__name__}")

        contexts.append(InsightsMutationContext(insights=insights_output))

        # Family tree context (ancestors + descendants)
        # Ancestors: Get from current program's own lineage stage output (dict[parent_id, LineageAnalysis])
        lineage_dict = self.get_input_optional("lineage_insights") or {}
        if not isinstance(lineage_dict, dict):
            raise StageError(f"Lineage insights output for {program.id} is not a dict")
        
        # Validate that all values are LineageAnalysis instances
        for key, value in lineage_dict.items():
            if not isinstance(value, LineageAnalysis):
                raise StageError(f"Lineage insights output for {program.id} contains non-LineageAnalysis value at key '{key}'")

        ancestor_lineages: list[LineageAnalysis] = list(lineage_dict.values())

        descendants = self.get_input_optional("descendant_lineages") or {}
        if not isinstance(descendants, dict):
            raise StageError(f"Descendant lineages output for {program.id} is not a dict")
        
        # Validate that all values are LineageAnalysis instances
        for key, value in descendants.items():
            if not isinstance(value, LineageAnalysis):
                raise StageError(f"Descendant lineages output for {program.id} contains non-LineageAnalysis value at key '{key}'")
        
        descendant_lineages: list[LineageAnalysis] = list(descendants.values())

        
        if ancestor_lineages or descendant_lineages:
            formatter = MetricsFormatter(self.metrics_context)
            selected_descendants = random.sample(descendant_lineages, min(2, len(descendant_lineages)))
            
            contexts.append(FamilyTreeMutationContext(
                ancestors=random.sample(ancestor_lineages, min(1, len(ancestor_lineages))),
                descendants=selected_descendants,
                metrics_formatter=formatter
            ))
        
        if not contexts:
            logger.warning(f"No context data available for {program.id}")
        
        context = CompositeMutationContext(contexts=contexts)
        program.set_metadata(self.metadata_key, context)
  
        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output=context,
            stage_name=self.stage_name
        )


