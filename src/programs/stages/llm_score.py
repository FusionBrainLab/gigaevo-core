from datetime import datetime
import json
import re
from typing import Optional

from loguru import logger
from pydantic import BaseModel, field_validator

from src.exceptions import StageError
from src.llm.wrapper import LLMInterface
from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.runner.stage_registry import StageRegistry

STRICT_SCORE_PROMPT_TEMPLATE = """
You are evaluating the following Python program based on the trait:

🧠 Trait: {trait_description}

Please assign a score between 0.0 and {max_score} that quantifies the program's quality in this trait.

Only return a **single JSON number**, with no explanation or formatting.

Program:
```python
{code}
```

Return only:
```json
{{"score": <float>}}
```
""".strip()


class ScoreStageConfig(BaseModel):
    llm_wrapper: LLMInterface
    trait_description: str  # e.g., "code novelty", "creative reuse", etc.
    score_metric_name: str  # key for program.metrics
    prompt_template: Optional[str] = None
    max_expected_score: float = 1.0

    class Config:
        arbitrary_types_allowed = True

    @field_validator("max_expected_score")
    def validate_max_score(cls, v):
        if not (v > 0):
            raise ValueError("max_expected_score must be positive")
        return v


@StageRegistry.register(
    description="Generate LLM-based scores for program evaluation"
)
class GenerateLLMScoreStage(Stage):
    def __init__(self, config: ScoreStageConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.prompt_template = (
            config.prompt_template or STRICT_SCORE_PROMPT_TEMPLATE
        )

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        try:
            logger.debug(
                f"[{self.stage_name}] Generating score for {self.config.score_metric_name} on program {program.id}"
            )
            user_prompt = self._render_user_prompt(program)
            response = await self.config.llm_wrapper.generate_async(user_prompt)

            parsed_score = self._parse_score(response)
            logger.info(
                f"[{self.stage_name}] Parsed score = {parsed_score:.3f}"
            )

            program.metrics[self.config.score_metric_name] = parsed_score

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=parsed_score,
                stage_name=self.stage_name,
                metadata={
                    f"{self.config.score_metric_name}_model_used": self.config.llm_wrapper.model,
                    "raw_llm_response": response.strip(),
                },
            )

        except Exception as e:
            error_msg = f"Error scoring program {program.id}: {e}"
            logger.error(f"[{self.stage_name}] {error_msg}")
            raise StageError(error_msg) from e

    def _render_user_prompt(self, program: Program) -> str:
        return self.prompt_template.format(
            trait_description=self.config.trait_description,
            code=program.code,
            max_score=self.config.max_expected_score,
        )

    def _parse_score(self, response: str) -> float:
        raw = response.strip()

        # Attempt JSON parsing first
        try:
            if raw.startswith("{") or raw.startswith("["):
                data = json.loads(raw)
                if isinstance(data, (int, float)):
                    return float(data)
                elif isinstance(data, dict) and "score" in data:
                    score_value = data["score"]
                    if isinstance(score_value, (int, float)):
                        return float(score_value)
                    else:
                        raise ValueError(
                            f"Score value is not numeric: {score_value}"
                        )
        except json.JSONDecodeError as e:
            logger.debug(f"[{self.stage_name}] JSON parsing failed: {e}")
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"[{self.stage_name}] JSON structure invalid: {e}")

        # Fallback: regex float extraction
        # Fixed regex pattern - removed double backslashes
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            raise StageError(
                f"Could not extract float score from LLM response: '{raw}'"
            )

        try:
            value = float(match.group())
        except ValueError as e:
            raise StageError(
                f"Could not convert matched string to float: '{match.group()}' from '{raw}'"
            ) from e

        # Clamp to valid range [0, max_expected_score]
        return min(max(value, 0.0), self.config.max_expected_score)
