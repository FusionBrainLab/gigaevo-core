from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
import json

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.programs.program import Program, ProgramStageResult, StageState
from src.database.program_storage import ProgramStorage
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.exceptions import StageError
from src.llm.wrapper import LLMInterface


DEFAULT_SYSTEM_PROMPT_TEMPLATE_JSON = """
You are an expert in Python code analysis, mathematical modeling, and software evolution.

You operate within an autonomous system that evolves Python programs to solve complex scientific or mathematical problems.
Your task is to extract structured, mutation-relevant insights that guide future improvements.

You must:
- Analyze the program's logic, abstractions, and algorithmic structure
- Detect flaws, bottlenecks, or fragilities that impair performance or scalability
- Identify missed opportunities for generalization, modularity, or reuse
- Spot dangerous or fragile constructs (e.g., hardcoded logic, undefined cases)
- Suggest improvements that increase robustness and evolvability

Avoid high-level summaries, vague advice, or cosmetic style comments.
Focus strictly on algorithmic, structural, or mathematical substance.

The current optimization objective is:
{evolutionary_task_description}

Return a JSON list. Each item must include:
- "type": one of {insight_types}
- "insight": a short, specific suggestion (≤ 25 words)
""".strip()

DEFAULT_SYSTEM_PROMPT_TEMPLATE_TEXT = """
You are an expert in Python code analysis, mathematical modeling, and evolutionary optimization.
Your task is to generate a set of **categorized, causal insights** to guide future code mutations.

The current optimization goal is:
{evolutionary_task_description}

Each insight must be:
- Short (≤25 words)
- Categorized (use one of: {insight_types})
- Causal and actionable (cause → effect → suggestion)

Only return a **bullet list** of insights in the format:
- <category>: <concrete causal insight>

Use relevance tags:
[beneficial], [harmful], or [neutral]
""".strip()


DEFAULT_USER_PROMPT_TEMPLATE_JSON = """
Analyze the following Python program and extract structured insights to guide further evolution.

Program:
```python
{code}
```
**Current Optimization Metrics:**
{metrics}

**Current Errors:**
{error_section}

Return a list of 3–{max_insights} insights in JSON format like:
[
  {"type": "geometric", "insight": "Try recursive Apollonian packing instead of radial symmetry."},
  ...
]

Guidelines:
- Do not repeat generic patterns unless they are applied in novel ways
- Avoid vague advice or stylistic feedback
- Each insight must be concise, actionable, and materially impactful
{error_focus}
""".strip()

DEFAULT_USER_PROMPT_TEMPLATE_TEXT = """
Analyze the following Python program and generate 3–{max_insights} actionable insights to guide future improvements.

Program:
```python
{code}
```
**Current Optimization Metrics:**
{metrics}

**Current Errors:**
{error_section}

🧠 Identify patterns and structural weaknesses in the program.

Each insight must follow this format:
- <category>: <causal insight>

Example:
 - geometric: Central clustering leaves voids along edges — try offset arc layout to increase wall coverage.

{error_focus}
""".strip()


class InsightsConfig(BaseModel):
    llm_wrapper: LLMInterface
    evolutionary_task_description: str
    excluded_error_stages: Optional[List[str]] = None
    metadata_key: str = "insights"
    output_format: Literal["json", "text"] = "text"
    max_insights: int = 7
    insight_types: List[Literal[
        "geometric", "algorithmic", "evolution", "math", "implementation", "error_fix"
    ]] = Field(default_factory=lambda: [
        "geometric", "algorithmic", "evolution", "math", "implementation", "error_fix"
    ])
    metrics_to_display: Dict[str, str] = Field(default_factory=dict)
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None
    class Config:
        arbitrary_types_allowed = True

    @field_validator("output_format")
    def validate_output_format(cls, v):
        if v not in {"json", "text"}:
            raise ValueError("output_format must be 'json' or 'text'")
        return v


class GenerateLLMInsightsStage(Stage):
    def __init__(self, config: InsightsConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        if self.config.system_prompt_template is not None:
            self.system_prompt_template = self.config.system_prompt_template
        else:
            if self.config.output_format == "json":
                self.system_prompt_template = DEFAULT_SYSTEM_PROMPT_TEMPLATE_JSON
            else:
                self.system_prompt_template = DEFAULT_SYSTEM_PROMPT_TEMPLATE_TEXT

        if self.config.user_prompt_template is not None:
            self.user_prompt_template = self.config.user_prompt_template
        else:
            if self.config.output_format == "json":
                self.user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE_JSON
            else:
                self.user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE_TEXT

        
        self.system_prompt = self.system_prompt_template.format(
            evolutionary_task_description=self.config.evolutionary_task_description,
            insight_types=", ".join(self.config.insight_types)
        )

        logger.info(
            f"[{self.stage_name}] Initialized LLM insights stage for task: {self.config.evolutionary_task_description[:100]}..."
        )

    async def _execute_stage(self, program: Program, started_at: datetime) -> ProgramStageResult:
        try:
            logger.debug(f"[{self.stage_name}] Generating insights for program {program.id}")
            user_prompt = self._render_user_prompt(program)
            response = await self.config.llm_wrapper.generate_async(user_prompt, system_prompt=self.system_prompt)
            parsed = response.strip()

            if not parsed:
                raise StageError("LLM returned empty or whitespace-only response")
            
            if self.config.output_format == "json":
                try:
                    parsed = json.loads(parsed)
                    assert isinstance(parsed, list)
                    for item in parsed:
                        assert "type" in item and "insight" in item
                except Exception as e:
                    raise StageError(f"Failed to parse JSON insight list: {e}\nRaw: {parsed}") from e

            logger.info(f"[{self.stage_name}] Successfully generated insights for program {program.id}")

            program.set_metadata(self.config.metadata_key, parsed)
            logger.debug(f"[{self.stage_name}] Stored insights in program.metadata['{self.config.metadata_key}']")

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=parsed,
                stage_name=self.stage_name,
                metadata={
                    self.config.metadata_key: parsed,
                    f"{self.config.metadata_key}_model_used": self.config.llm_wrapper.model, 
                    f"{self.config.metadata_key}_output_format": self.config.output_format,
                },
            )

        except Exception as e:
            error_msg = f"Error generating insights for program {program.id}: {e}"
            logger.error(f"[{self.stage_name}] {error_msg}")
            raise StageError(error_msg) from e

    def _render_user_prompt(self, program: Program) -> str:
        metrics_str = self._format_metrics(program.metrics)
        return self.user_prompt_template.format(
            code=program.code,
            metrics=metrics_str,
            error_section=self._format_error_section(program),
            error_focus=self._format_error_focus(program),
            max_insights=self.config.max_insights
        )

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        lines = []
        for metric_key, metric_description in self.config.metrics_to_display.items():
            v = metrics[metric_key]
            lines.append(f"- {metric_key} : {v} ({metric_description})")

        return "\n".join(lines)

    def _format_error_section(self, program: Program) -> str:
        excluded = set(self.config.excluded_error_stages or [])
        failed = [s for s in program.get_failed_stages() if s not in excluded]
        if not failed:
            return ""
        blocks = []
        for stage in failed:
            summary = program.get_stage_error_summary(stage)
            if summary:
                blocks.append(f"=== Stage: {stage} ===\n{summary}")
        return f"\nStage Execution Errors:\n{chr(10).join(blocks)}\n" if blocks else ""

    def _format_error_focus(self, program: Program) -> str:
        failed = [s for s in program.get_failed_stages() if s not in (self.config.excluded_error_stages or [])]
        if not failed:
            return ""
        return f"\n- **Error Analysis**: Focus on fixing or avoiding failure modes from stages: {', '.join(failed)}"

    def get_task_description(self) -> str:
        return self.config.evolutionary_task_description

    def get_formatted_system_prompt(self) -> str:
        return self.system_prompt

    def get_excluded_error_stages(self) -> List[str]:
        return self.config.excluded_error_stages or []

    def is_stage_error_excluded(self, stage_name: str) -> bool:
        return stage_name in (self.config.excluded_error_stages or [])

    def get_formatted_user_prompt(self, program: Program) -> str:
        return self._render_user_prompt(program)