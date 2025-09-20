from datetime import datetime
import json
from typing import Any, Optional, Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.exceptions import StageError
from src.llm.wrapper import LLMInterface
from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.programs.metrics.context import MetricsContext
from src.programs.metrics.formatter import MetricsFormatter

DEFAULT_SYSTEM_PROMPT_TEMPLATE_JSON = """
You are an expert in Python code analysis and evolutionary optimization.

TASK: Extract structured insights to guide program evolution for: {evolutionary_task_description}

ANALYZE:
- Logic, abstractions, and algorithmic structure
- Flaws, bottlenecks, or fragilities affecting performance
- Missed opportunities for generalization or modularity
- Fragile constructs (hardcoded logic, undefined cases)

REQUIREMENTS:
- Return JSON list with "type" (one of {insight_types}) and "insight" (≤25 words)
- Focus on algorithmic/structural substance, not style
- Provide specific, actionable suggestions
""".strip()

DEFAULT_SYSTEM_PROMPT_TEMPLATE_TEXT = """
You are an expert in Python code analysis and evolutionary optimization.

TASK: Generate causal insights to guide code mutations for: {evolutionary_task_description}

CATEGORIES ({insight_types}):
- structural: code architecture, modularity, control flow patterns limiting effectiveness
- algorithmic: point placement logic, geometric construction methods, optimization procedures
- optimization: search strategy weaknesses, local minima traps, exploration vs exploitation imbalance
- numerical: parameter sensitivity, precision issues, convergence criteria, scaling problems
- semantic: geometric assumptions, constraint violations, domain-specific logic flaws
- error_handling: edge cases, validation gaps, robustness failures under boundary conditions

REQUIREMENTS:
- Each insight ≤25 words with concrete evidence (values, metrics, structures)
- Be causal and actionable - link cause, effect, and solution
- Ground observations in code/metrics/errors, not speculation
- **Prioritize fitness-impacting patterns** over minor refinements

Strict rules:
- Do **not** suggest changes based on speculative efficiency (e.g., "this is faster") unless directly tied to the optimization objective
- You do **not** have access to runtime speed — avoid assumptions about performance or memory usage
- Do **not** hallucinate structural flaws without evidence from code, metrics, or error outputs
- Do **not** include summaries, general advice, or stylistic feedback
- **Focus on geometric reasoning** - triangle areas, point distributions, constraint satisfaction

✅ Only recommend structural or numerical improvements that clearly support maximizing minimum triangle area

TAGS: beneficial, harmful, neutral, fragile, rigid
SEVERITY: high (direct fitness impact), medium (geometric reasoning), low (implementation details)

**TAG DEFINITIONS (evolutionary action guidance)**:
- **beneficial**: Current pattern is good - PRESERVE/EXTEND this approach
- **harmful**: Current pattern is bad - REMOVE/AVOID this approach  
- **fragile**: Current pattern is risky - IMPROVE/ROBUSTIFY this approach
- **rigid**: Current pattern is inflexible - MAKE MORE ADAPTABLE
- **neutral**: Current pattern has no clear impact - IGNORE for evolution

FORMAT: - <category> [tag] (severity): <insight with evidence>
EXAMPLE: - algorithmic [harmful] (high): Grid placement creates systematic collinearity at y=0.33, min_area=0.001 vs target>0.01
""".strip()


DEFAULT_USER_PROMPT_TEMPLATE_JSON = """
Analyze this Python program and extract 3-{max_insights} structured insights for evolution.

PROGRAM:
```python
{code}
```

METRICS: {metrics}
ERRORS: {error_section}

Return JSON format:
[
  {{"type": "category", "insight": "specific suggestion with evidence (≤25 words)"}},
  ...
]

Requirements: concise, actionable, evidence-based insights only.
{error_focus}
""".strip()

DEFAULT_USER_PROMPT_TEMPLATE_TEXT = """
Analyze this program and extract 3-{max_insights} actionable insights for evolutionary refinement.

PROGRAM:
```python
{code}
```

METRICS: {metrics}
ERRORS: {error_section}

Extract insights with concrete evidence from code/metrics/errors only. No speculation.

{error_focus}
""".strip()


class InsightsConfig(BaseModel):
    llm_wrapper: LLMInterface
    evolutionary_task_description: str
    excluded_error_stages: Optional[list[str]] = None
    metadata_key: str = "insights"
    output_format: Literal["json", "text"] = "text"
    max_insights: int = 7
    insight_types: list[
        Literal[
            "structural",
            "algorithmic",
            "optimization",
            "numerical",
            "semantic",
            "error_handling",
        ]
    ] = Field(
        default_factory=lambda: [
            "structural",
            "algorithmic",
            "optimization",
            "numerical",
            "semantic",
            "error_handling",
        ]
    )
    metrics_context: MetricsContext
    metrics_formatter: Optional[MetricsFormatter] = None
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
        self.formatter = config.metrics_formatter or MetricsFormatter(
            config.metrics_context, use_range_normalization=False
        )

        # Select templates based on output format
        self.system_prompt_template = self._select_template(
            custom=config.system_prompt_template,
            json_default=DEFAULT_SYSTEM_PROMPT_TEMPLATE_JSON,
            text_default=DEFAULT_SYSTEM_PROMPT_TEMPLATE_TEXT,
        )

        self.user_prompt_template = self._select_template(
            custom=config.user_prompt_template,
            json_default=DEFAULT_USER_PROMPT_TEMPLATE_JSON,
            text_default=DEFAULT_USER_PROMPT_TEMPLATE_TEXT,
        )

        self.system_prompt = self.system_prompt_template.format(
            evolutionary_task_description=self.config.evolutionary_task_description,
            insight_types=", ".join(self.config.insight_types),
        )

        logger.info(
            f"[{self.stage_name}] Initialized LLM insights stage for task: {self.config.evolutionary_task_description[:100]}..."
        )

    def _select_template(
        self, custom: Optional[str], json_default: str, text_default: str
    ) -> str:
        """Select appropriate template based on config."""
        if custom is not None:
            return custom
        return (
            json_default
            if self.config.output_format == "json"
            else text_default
        )

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        try:
            logger.debug(
                f"[{self.stage_name}] Generating insights for program {program.id}"
            )
            user_prompt = self._render_user_prompt(program)
            response = await self.config.llm_wrapper.generate_async(
                user_prompt, system_prompt=self.system_prompt
            )
            parsed = response.strip()

            if not parsed:
                raise StageError(
                    "LLM returned empty or whitespace-only response"
                )

            if self.config.output_format == "json":
                try:
                    parsed = json.loads(parsed)
                    assert isinstance(parsed, list)
                    for item in parsed:
                        assert "type" in item and "insight" in item
                except Exception as e:
                    raise StageError(
                        f"Failed to parse JSON insight list: {e}\nRaw: {parsed}"
                    ) from e

            logger.info(
                f"[{self.stage_name}] Successfully generated insights for program {program.id}"
            )

            program.set_metadata(self.config.metadata_key, parsed)
            logger.debug(
                f"[{self.stage_name}] Stored insights in program.metadata['{self.config.metadata_key}']"
            )

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
            error_msg = (
                f"Error generating insights for program {program.id}: {e}"
            )
            logger.error(f"[{self.stage_name}] {error_msg}")
            raise StageError(error_msg) from e

    def _render_user_prompt(self, program: Program) -> str:
        """Render the user prompt with program data using the configured formatter."""
        metrics_str = self.formatter.format_metrics_block(program.metrics)
        return self.user_prompt_template.format(
            code=program.code,
            metrics=metrics_str,
            error_section=self._format_error_section(program),
            error_focus=self._format_error_focus(program),
            max_insights=self.config.max_insights,
        )

    def _get_filtered_failed_stages(self, program: Program) -> list[str]:
        """Get failed stages excluding configured exclusions."""
        excluded = set(self.config.excluded_error_stages or [])
        return [s for s in program.get_failed_stages() if s not in excluded]

    def _format_error_section(self, program: Program) -> str:
        """Format error section for prompt."""
        failed = self._get_filtered_failed_stages(program)
        if not failed:
            return ""

        blocks = []
        for stage in failed:
            summary = program.get_stage_error_summary(stage)
            if summary:
                blocks.append(f"=== Stage: {stage} ===\n{summary}")

        return (
            f"\nStage Execution Errors:\n{chr(10).join(blocks)}\n"
            if blocks
            else ""
        )

    def _format_error_focus(self, program: Program) -> str:
        """Format error focus section for prompt."""
        failed = self._get_filtered_failed_stages(program)
        if not failed:
            return ""
        return f"\n- **Error Analysis**: Focus on fixing or avoiding failure modes from stages: {', '.join(failed)}"
