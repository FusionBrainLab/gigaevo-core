"""MetaEvolve Stages Package

This package contains all stage implementations for the MetaEvolve pipeline system.
Importing this module ensures all stages are registered with the StageRegistry.
"""

# Import all stage modules to ensure decorators are executed
from src.programs.stages import base
from src.programs.stages import collector
from src.programs.stages import complexity
from src.programs.stages import execution
from src.programs.stages import insights
from src.programs.stages import insights_lineage
from src.programs.stages import json_processing
from src.programs.stages import llm_score
from src.programs.stages import metrics
from src.programs.stages import validation
from src.programs.stages import worker_pool

# Import commonly used stage classes for convenience
from src.programs.stages.execution import (
    RunProgramCodeWithOptionalProducedData,
    ValidatorCodeExecutor,
    RunConstantPythonCode,
)
from src.programs.stages.metrics import EnsureMetricsStage, NormalizeMetricsStage
from src.programs.stages.validation import ValidateCodeStage
from src.programs.stages.insights import create_insights_stage
from src.programs.stages.insights_lineage import create_lineage_stage
from src.programs.stages.llm_score import ScoringStage
from src.programs.stages.collector import AncestorCollector, DescendantCollector
# WorkerPoolStage is a base class, not exported for direct use
from src.programs.stages.complexity import GetCodeLengthStage, ComputeComplexityStage
from src.programs.stages.json_processing import MergeDictStage, ParseJSONStage, StringifyJSONStage

__all__ = [
    # Base classes
    "base",
    # Stage modules
    "collector",
    "complexity",
    "execution", 
    "insights",
    "insights_lineage",
    "json_processing",
    "llm_score",
    "metrics",
    "validation",
    "worker_pool",
    # Commonly used stage classes
    "RunProgramCodeWithOptionalProducedData",
    "ValidatorCodeExecutor", 
    "RunConstantPythonCode",
    "EnsureMetricsStage",
    "NormalizeMetricsStage",
    "ValidateCodeStage",
    "create_insights_stage",
    "create_lineage_stage",
    "ScoringStage",
    "AncestorCollector",
    "DescendantCollector",
    "GetCodeLengthStage",
    "ComputeComplexityStage",
    "MergeDictStage",
    "ParseJSONStage",
    "StringifyJSONStage",
]