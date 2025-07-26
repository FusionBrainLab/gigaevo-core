from datetime import datetime
import difflib
import json
import random
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel

from src.database.program_storage import ProgramStorage
from src.exceptions import StageError
from src.llm.wrapper import LLMInterface
from src.programs.program import Program, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result

DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT = """
You are an expert in evolutionary programming and performance-guided code optimization.

Your task is to analyze a code transition between two Python programs. You are given unified diff blocks and a performance metric delta showing how the child's behavior changed relative to its parent.

🎯 Return **3–5 concise, causal insights** explaining how specific changes likely contributed to the metric change.

Each insight must:
- Be formatted as a JSON object with:
  - `"strategy"`: one of `"imitation"`, `"avoidance"`, or `"generalization"`
  - `"description"`: a **brief, causal explanation** of a specific architectural or algorithmic change and how it may have affected performance
- Be ≤ 25 words and focus on concrete mechanisms (e.g., perturbation logic, layout symmetry, parameter adaptation)
- **Optionally** include a diff range in the description, like: `(@@ -32,7 +33,9 @@)`

📦 Example response:
[
  {
    "strategy": "imitation",
    "description": "(@@ -45,10 +45,12 @@) Introduced adaptive sampling interval, improving convergence stability and diversity of explored solutions."
  },
  {
    "strategy": "avoidance",
    "description": "Removed annealing schedule, causing premature convergence and degraded triangle distribution."
  }
]

🛑 Do NOT include:
- Diffs
- Metric values

⚠️ Avoid these types of weak insights:
- "This change improved the metric"
- "Better performance due to optimization"
- "Improved speed or accuracy"
- Any vague statement without a concrete architectural reference

Respond with **only valid JSON**, no markdown or extra text.
""".strip()

DEFAULT_USER_PROMPT_LINEAGE_TEXT = """
🔬 TASK CONTEXT:
{task_description}

📈 METRIC:
Optimization target = `{metric_name}` ({metric_description})  
Observed change = {delta:+.5f}

Below are unified diff blocks showing changes from the parent to the child program. Each block begins with a line like:

  @@ -32,7 +33,9 @@

Analyze these diffs and return high-quality causal insights explaining how the changes likely led to the metric delta. Each insight may optionally mention the diff range it relates to.

{diff_blocks}
""".strip()


def format_structured_insight_block(
    child_insights: Optional[Dict[str, Any]],
    ancestor_insights: List[Dict[str, Any]]
) -> str:
    lines = []

    # Always show ancestors section, even if empty, for consistency
    lines.append("## ANCESTORS")
    if ancestor_insights:
        for ancestor in ancestor_insights:
            delta = ancestor.get("delta", "unknown")
            title = _generate_readable_title("parent", delta)
            lines.append(f"### {title}")

            insights = ancestor.get("insights", [])
            for insight in insights:
                strategy = insight.get("strategy", "unknown")
                description = insight.get("description", "No description")
                lines.append(f"- **[{strategy}]** {description}")

            diff_blocks = ancestor.get("diff_blocks", [])
            if diff_blocks:
                lines.append("")
                lines.append("#### Diff Blocks")
                for i, block in enumerate(diff_blocks):
                    lines.append(f"--- Diff Block {i+1} ---\n```diff\n{block}\n```")

            lines.append("")
    else:
        lines.append("*No ancestor information available (likely initial/seed program)*")
        lines.append("")

    if child_insights:
        lines.append("## DESCENDANTS")
        delta = child_insights.get("delta", "unknown")
        title = _generate_readable_title("child", delta)
        lines.append(f"### {title}")

        insights = child_insights.get("insights", [])
        for insight in insights:
            strategy = insight.get("strategy", "unknown")
            description = insight.get("description", "No description")
            lines.append(f"- **[{strategy}]** {description}")

        diff_blocks = child_insights.get("diff_blocks", [])
        if diff_blocks:
            lines.append("")
            lines.append("#### Diff Blocks")
            for i, block in enumerate(diff_blocks):
                lines.append(f"--- Diff Block {i+1} ---\n```diff\n{block}\n```")

    if not child_insights and not ancestor_insights:
        lines.append("*No lineage information available*")

    return "\n".join(lines)


def _generate_readable_title(relationship: str, delta: str) -> str:
    """Generate human-readable titles for lineage transitions instead of showing program hashes."""
    try:
        delta_value = float(delta.replace("+", ""))
    except (ValueError, AttributeError):
        delta_value = 0.0
    
    # Determine performance change direction (magnitude-agnostic)
    if abs(delta_value) < 1e-6:  # Essentially zero
        change_desc = "No Change"
    elif delta_value > 0:
        change_desc = "Improvement"
    else:
        change_desc = "Decline"
    
    # Create descriptive title based on relationship
    if relationship == "parent":
        base = "Direct Parent"
    else:  # child
        if delta_value > 0:
            base = "Improved Child"
        elif delta_value < 0:
            base = "Child Variant"
        else:
            base = "Unchanged Child"
    
    return f"{base} ({change_desc} Δ{delta})"


class LineageInsightsConfig(BaseModel):
    llm_wrapper: LLMInterface
    metric_key: str
    metric_description: str
    metadata_key_raw: str = "lineage_insights_raw"
    metadata_key: str = "lineage_insights"
    parent_selection_strategy: Literal["first", "random", "best_fitness"] = "first"
    higher_is_better: bool = True
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT
    user_prompt_template: str = DEFAULT_USER_PROMPT_LINEAGE_TEXT
    fitness_selector_metric: Optional[str] = None
    fitness_selector_higher_is_better: Optional[bool] = None
    task_description: str

    class Config:
        arbitrary_types_allowed = True


class GenerateLineageInsightsStage(Stage):
    MAX_DESCENDANTS = 3

    def __init__(self, config: LineageInsightsConfig, storage: ProgramStorage, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.storage = storage

    async def _select_parent_id(self, program: Program) -> Optional[str]:
        """Select parent ID based on configured strategy."""
        if not program.lineage or not program.lineage.parents:
            return None

        if self.config.parent_selection_strategy == "first":
            return program.lineage.parents[0]
        elif self.config.parent_selection_strategy == "random":
            return random.choice(program.lineage.parents)
        elif self.config.parent_selection_strategy == "best_fitness":
            # Find parent with highest fitness (or lowest if higher_is_better=False)
            best_parent_id = None
            best_fitness = None
            fitness_key = self.config.fitness_selector_metric or self.config.metric_key
            higher_better = self.config.fitness_selector_higher_is_better
            if higher_better is None:
                higher_better = self.config.higher_is_better

            for parent_id in program.lineage.parents:
                parent = await self.storage.get(parent_id)
                if parent and fitness_key in parent.metrics:
                    fitness = float(parent.metrics[fitness_key])
                    if best_fitness is None or (
                        (higher_better and fitness > best_fitness) or
                        (not higher_better and fitness < best_fitness)
                    ):
                        best_fitness = fitness
                        best_parent_id = parent_id
            return best_parent_id
        else:
            raise ValueError(f"Unknown parent selection strategy: {self.config.parent_selection_strategy}")

    async def _set_empty_lineage_and_return(self, program: Program, started_at: datetime, message: str):
        """Helper to set empty lineage metadata, persist to storage, and return completed result."""
        program.set_metadata(self.config.metadata_key_raw, {"ancestors": [], "descendants": []})
        program.set_metadata(self.config.metadata_key, format_structured_insight_block(None, []))
        await self.storage.update(program)
        return build_stage_result(StageState.COMPLETED, started_at, message, self.stage_name)

    async def _set_child_lineage_and_return(self, program: Program, child_ancestors: List, child_insights: Dict, started_at: datetime, parent_id: str):
        """Helper to set child lineage metadata, persist to storage, and return completed result."""
        raw_key = self.config.metadata_key_raw
        program.set_metadata(raw_key, {"ancestors": child_ancestors})
        program.set_metadata(self.config.metadata_key, format_structured_insight_block(None, child_ancestors))
        await self.storage.update(program)
        
        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output=child_insights,
            stage_name=self.stage_name,
            metadata={
                self.config.metadata_key: child_insights,
                "parent_id": parent_id,
                "selected_parent_strategy": self.config.parent_selection_strategy,
            },
        )

    def _compute_diff_blocks(self, parent_code: str, child_code: str) -> List[str]:
        if parent_code.strip() == child_code.strip():
            return []

        diff_lines = list(difflib.unified_diff(
            parent_code.strip().splitlines(),
            child_code.strip().splitlines(),
            lineterm="",
            n=3
        ))

        if not diff_lines:
            return []

        content = [line for line in diff_lines if not line.startswith(('---', '+++'))]
        if not content:
            return []

        hunks = []
        current_hunk = []
        for line in content:
            if line.startswith('@@'):
                if current_hunk:
                    if self._hunk_has_changes(current_hunk):
                        hunks.append('\n'.join(current_hunk))
                current_hunk = [line]
            else:
                current_hunk.append(line)
        if current_hunk and self._hunk_has_changes(current_hunk):
            hunks.append('\n'.join(current_hunk))

        return hunks

    def _hunk_has_changes(self, hunk_lines: List[str]) -> bool:
        return any(line.startswith(('+', '-')) and not line.startswith(('+++', '---')) for line in hunk_lines)

    def _render_user_prompt(self, delta: float, diff_blocks: List[str]) -> str:
        rendered_blocks = "\n\n".join([f"--- Block {i+1} ---\n```diff\n{b}\n```" for i, b in enumerate(diff_blocks)])
        return self.config.user_prompt_template.format(
            task_description=self.config.task_description,
            metric_name=self.config.metric_key,
            metric_description=self.config.metric_description,
            delta=delta,
            diff_blocks=rendered_blocks,
        )

    async def _execute_stage(self, program: Program, started_at: datetime):
        try:
            parent_id = await self._select_parent_id(program)
            if not parent_id:
                # Handle programs with no parents (initial/seed programs) by setting empty lineage insights
                return await self._set_empty_lineage_and_return(program, started_at, "<No valid parent - initial program>")
            
            parent = await self.storage.get(parent_id)
            if not parent:
                # Set empty lineage insights even when parent not found
                return await self._set_empty_lineage_and_return(program, started_at, f"<Parent {parent_id} not found>")

            metric_key = self.config.metric_key
            pm = float(parent.metrics[metric_key])
            cm = float(program.metrics[metric_key])
            delta = cm - pm

            diff_blocks = self._compute_diff_blocks(parent.code, program.code)
            if not diff_blocks:
                # Even with no code differences, we still have a lineage relationship
                # Continue with empty diff_blocks and empty insights
                diff_blocks = []
                insights = []
            else:
                prompt = self._render_user_prompt(delta, diff_blocks)
                response = await self.config.llm_wrapper.generate_async(prompt, system_prompt=self.config.system_prompt_template)

                try:
                    insights = json.loads(response)
                except Exception as e:
                    raise StageError(f"Failed to parse LLM JSON output: {e}\nRaw output:\n{response}") from e

            child_insights = {
                "from": parent.id,
                "to": program.id,
                "delta": f"{delta:+.4f}",
                "diff_blocks": diff_blocks,
                "insights": insights,
            }

            raw_key = self.config.metadata_key_raw
            parent_raw = parent.metadata.get(raw_key, {})
            parent_descendants = sorted(
                parent_raw.get("descendants", []), 
                key=lambda x: float(x.get("delta", "0").replace("+", "")), 
                reverse=self.config.higher_is_better
            )
            if len(parent_descendants) >= self.MAX_DESCENDANTS:
                parent_descendants = parent_descendants[-(self.MAX_DESCENDANTS - 1):]
            parent_descendants.append(child_insights)
            parent.set_metadata(raw_key, {"descendants": parent_descendants})
            ancestor_block = parent_raw.get("ancestors", [])
            parent.set_metadata(self.config.metadata_key, format_structured_insight_block(child_insights, ancestor_block))
            await self.storage.update(parent)

            # Only store immediate parent (depth 1 ancestors)
            child_ancestors = [child_insights]
            return await self._set_child_lineage_and_return(program, child_ancestors, child_insights, started_at, parent.id)

        except Exception as e:
            raise StageError(f"[{self.stage_name}] Failed: {e}") from e
