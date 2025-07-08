from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from src.programs.stages.base import Stage, ProgramStageResult, StageError, StageState, build_stage_result
from src.llm.wrapper import LLMInterface
from src.programs.program import Program



class LLMInsightValidationResult(BaseModel):
    """General-purpose structured descriptors extracted from a generated program via LLM."""

    used_insight_types: List[str] = Field(
        default_factory=list,
        description="Types of insights used in the program (e.g., geometric, algorithmic, evolutionary)"
    )

    lineage_strategy: Optional[Literal[
        "imitation", "generalization", "avoidance", "hybrid", "novel", "none"
    ]] = Field(
        default="none",
        description="How the program relates to its parent(s)"
    )

    architecture_summary: Optional[str] = Field(
        default=None,
        description="Brief human-readable summary of spatial or algorithmic structure"
    )

    code_modality: Optional[Literal[
        "manual_layout",     # Hardcoded coordinates or static geometry
        "procedural",        # Deterministic, step-by-step logic
        "heuristic",         # Rule-based approximations
        "optimization_loop", # Iterative improvement, e.g., annealing or gradient-based
        "symbolic",          # Equations, closed-form math, analytical modeling
        "search_based",      # Greedy, random, or stochastic search
        "unknown"            # Could not be determined
    ]] = Field(
        default="unknown",
        description="Type of code generation strategy"
    )

    novelty_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Subjective 0–1 score indicating novelty vs known approaches"
    )

    insight_fidelity: Optional[str] = Field(
        default=None,
        description="LLM summary of whether the program reflects provided insights"
    )

    failure_mode_summary: Optional[str] = Field(
        default=None,
        description="If applicable, explanation of why the program failed or performed poorly"
    )


class LLMValidatorConfig(BaseModel):
    llm_wrapper: LLMInterface
    task_description: str
    metadata_key: str = "validation_summary"
    output_format: Literal["json"] = "json"  # Only JSON supported
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMInsightValidatorStage(Stage):
    def __init__(self, config: LLMValidatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""You are an expert in Python code evaluation for evolutionary tasks.

Your goal is to extract structured, high-level descriptors from a Python program generated during a search process.

You will analyze:
- Code structure
- Execution output
- Parent lineage (if present)

Return a JSON dictionary with the following fields:
- used_insight_types: List[str] (e.g., geometric, algorithmic)
- lineage_strategy: One of ["imitation", "generalization", "avoidance", "hybrid", "novel", "none"]
- architecture_summary: short description of core spatial or logical structure
- code_modality: One of ["manual_layout", "procedural", "heuristic", "optimization_loop", "symbolic", "unknown"]
- novelty_score: Float between 0.0 and 1.0
- insight_fidelity: short sentence describing how well the code aligns with prior insights
- failure_mode_summary: optional sentence describing what failure occurred, if any

Task: {self.config.task_description}
""".strip()





