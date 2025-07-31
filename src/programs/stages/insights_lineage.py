from datetime import datetime
import difflib
import json
import random
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
import re
from pydantic import BaseModel

from src.database.program_storage import ProgramStorage
from src.exceptions import StageError
from src.llm.wrapper import LLMInterface
from src.programs.program import Program, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result

def safe_json_extract_from_llm_output(text: str):
    """
    Extracts and safely parses a JSON array from an LLM response,
    even if it's wrapped in markdown or slightly malformed.
    """
    # Remove markdown fences
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Try to extract valid JSON array
    try:
        # Remove trailing commas before closing brackets/braces
        text = re.sub(r",\s*(\]|\})", r"\1", text)

        # Optionally: trim to first matching [ ... ] block (for safety)
        array_match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
        if array_match:
            text = array_match.group(0)

        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM output: {e}\n---\n{text}")

DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT = """
You are an expert in evolutionary programming analyzing code transitions between parent-child programs.

TASK: Produce 3-5 JSON insights analyzing what strategies were used in code changes and how they contributed to observed metric/error changes.

STRATEGY TYPES (describing what the child did):
- "imitation": preserved/extended parent logic (kept successful patterns)
- "avoidance": removed/bypassed parent logic (eliminated failing patterns)
- "generalization": added abstraction/parameterization (made approach more flexible)
- "exploration": introduced novel mechanisms (tried completely new approaches)

ANALYSIS FOCUS:
- **Causal mechanisms**: How specific code changes led to metric delta
- **Geometric impact**: How changes affected point distribution, triangle areas, constraints
- **Risk assessment**: Whether changes introduced fragility or improved robustness
- **Success patterns**: What made positive changes work and negative changes fail

REQUIREMENTS:
- Each insight: JSON object with "strategy" and "description" (≤30 words for complex geometric reasoning)
- Reference diff blocks (e.g., "(@@ -32,7 +33,9 @@)") when possible
- Cite concrete evidence (code lines, metric deltas, error messages, geometric properties)
- Focus on causal relationships with **quantified impact** when available
- One focused change per insight - don't combine multiple modifications

EXAMPLE:
[
  {
    "strategy": "imitation", 
    "description": "(@@ -45,10 +47,12 @@) Preserved parent's boundary-aware placement, maintaining min_distance>0.1; explains +0.012 improvement."
  },
  {
    "strategy": "avoidance",
    "description": "(@@ -78,6 +0,0 @@) Removed fixed grid causing collinearity, eliminating systematic zero-area triangles."
  },
  {
    "strategy": "generalization",
    "description": "(@@ -22,1 +24,3 @@) Parameterized step_size via alpha, enabling adaptive convergence; correlates with +0.007 gain."
  }
]

Respond with only valid JSON, no commentary.
""".strip()

DEFAULT_USER_PROMPT_LINEAGE_TEXT = """
TASK: {task_description}

METRIC TARGET: {metric_name} ({metric_description})
Observed delta (child - parent): {delta:+.5f}

ADDITIONAL METRICS:
{additional_metrics}

ERROR CHANGES:
Parent Errors: {parent_errors}
Child Errors: {child_errors}

**ANALYSIS INSTRUCTIONS:**
Analyze what strategies the child used compared to the parent and how these changes contributed to the metric delta or error changes. 

For POSITIVE deltas (+): Focus on what successful patterns were preserved, what failing patterns were removed, or what new mechanisms were introduced that led to improvement.

For NEGATIVE deltas (-): Focus on what successful patterns were lost, what problematic patterns were introduced, or what mechanisms failed.

For ERROR changes: Identify how code modifications resolved errors (positive) or introduced new failures (negative).

{diff_blocks}

PARENT PROGRAM (reference):
```python
{parent_code}
```
""".strip()


def format_structured_insight_block(
    child_insights: Optional[Dict[str, Any]],
    ancestor_insights: List[Dict[str, Any]]
) -> str:
    """
    Format structured lineage insights into a readable markdown block
    with distinct ANCESTORS and DESCENDANTS sections, each including:
    - Performance delta and type (Improvement, Decline, etc.)
    - Strategy-tagged insights with optional diff ranges
    - Diff blocks for context (separated from insights)
    """
    lines = []

    # === ANCESTORS SECTION ===
    lines.append("## ANCESTORS")
    if ancestor_insights:
        for ancestor in ancestor_insights:
            delta = ancestor.get("delta", "unknown")
            title = _generate_readable_title("parent", delta)
            lines.append(f"### Transition from Ancestor → Child ({title})")

            # Include additional metrics (if any)
            add_mets = ancestor.get("additional_metric_deltas", {})
            if add_mets:
                lines.append("#### Additional Metrics")
                for m_key, m_delta in add_mets.items():
                    lines.append(f"- {m_key}: Δ{m_delta}")

            insights = ancestor.get("insights", [])
            if insights:
                lines.append("#### Insights")
                for insight in insights:
                    strategy = insight.get("strategy", "unknown")
                    description = insight.get("description", "No description")
                    lines.append(f"- **[{strategy}]** {description}")
            else:
                lines.append("*No extracted insights.*")

            # diff_blocks = ancestor.get("diff_blocks", [])
            # if diff_blocks:
            #     lines.append("")
            #     lines.append("#### Diff Blocks")
            #     for i, block in enumerate(diff_blocks):
            #         lines.append(f"--- Diff Block {i+1} ---\n```diff\n{block}\n```")
            # lines.append("")
    else:
        lines.append("*No ancestor information available (likely initial/seed program)*")
        lines.append("")

    # === DESCENDANTS SECTION ===
    if child_insights:
        lines.append("## DESCENDANTS")
        delta = child_insights.get("delta", "unknown")
        title = _generate_readable_title("child", delta)
        lines.append(f"### Transition from Parent → Child ({title})")

        # Include additional metrics for child transition
        child_add_mets = child_insights.get("additional_metric_deltas", {})
        if child_add_mets:
            lines.append("#### Additional Metrics")
            for m_key, m_delta in child_add_mets.items():
                lines.append(f"- {m_key}: Δ{m_delta}")

        insights = child_insights.get("insights", [])
        if insights:
            lines.append("#### Insights")
            for insight in insights:
                strategy = insight.get("strategy", "unknown")
                description = insight.get("description", "No description")
                lines.append(f"- **[{strategy}]** {description}")
        else:
            lines.append("*No extracted insights.*")

        # diff_blocks = child_insights.get("diff_blocks", [])
        # if diff_blocks:
        #     lines.append("")
        #     lines.append("#### Diff Blocks")
        #     for i, block in enumerate(diff_blocks):
        #         lines.append(f"--- Diff Block {i+1} ---\n```diff\n{block}\n```")

    # Final fallback if empty
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
    additional_metrics: Dict[str, str] = {}
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

    def _render_user_prompt(self, delta: float, diff_blocks: List[str], parent: Program, child: Program) -> str:
        rendered_blocks = "\n\n".join([f"--- Block {i+1} ---\n```diff\n{b}\n```" for i, b in enumerate(diff_blocks)])
        
        # Get error summaries for both programs
        parent_errors = parent.get_all_errors_summary()
        child_errors = child.get_all_errors_summary()

        # Build additional metrics overview (apart from the primary optimisation metric)
        add_metrics_lines: List[str] = []
        for m_key, m_desc in self.config.additional_metrics.items():
            # Skip the main optimisation metric if it appears in this mapping to avoid duplication
            if m_key == self.config.metric_key:
                continue

            try:
                parent_val = float(parent.metrics[m_key])
                child_val = float(child.metrics[m_key])
                m_delta = child_val - parent_val
                add_metrics_lines.append(f"- {m_key} ({m_desc}) {m_delta:+.5f}")
            except (KeyError, ValueError):
                # If the metric is missing or cannot be converted to float we silently ignore it
                continue

        # Join into a single string expected by the template placeholder
        additional_metrics_str = "\n".join(add_metrics_lines) if add_metrics_lines else "N/A"
 
        return self.config.user_prompt_template.format(
            task_description=self.config.task_description,
            metric_name=self.config.metric_key,
            metric_description=self.config.metric_description,
            delta=delta,
            parent_errors=parent_errors,
            child_errors=child_errors,
            additional_metrics=additional_metrics_str,
            diff_blocks=rendered_blocks,
            parent_code=parent.code,
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
                prompt = self._render_user_prompt(delta, diff_blocks, parent, program)
                response = await self.config.llm_wrapper.generate_async(prompt, system_prompt=self.config.system_prompt_template)

                try:
                    insights = safe_json_extract_from_llm_output(response)
                except Exception as e:
                    raise StageError(f"Failed to parse LLM JSON output: {e}\nRaw output:\n{response}") from e

            # Calculate deltas for any configured additional metrics (excluding the primary metric)
            additional_metric_deltas: Dict[str, str] = {}
            for m_key in self.config.additional_metrics.keys():
                if m_key == self.config.metric_key:
                    # Skip the primary metric; it's already captured by the 'delta' field
                    continue

                try:
                    parent_val = float(parent.metrics[m_key])
                    child_val = float(program.metrics[m_key])
                    additional_metric_deltas[m_key] = f"{child_val - parent_val:+.5f}"
                except (KeyError, ValueError):
                    # Ignore metrics that are missing or non-numeric
                    continue

            child_insights = {
                "from": parent.id,
                "to": program.id,
                "delta": f"{delta:+.5f}",
                # Include per-metric deltas for configured additional metrics
                "additional_metric_deltas": additional_metric_deltas,
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
            ancestor_block = parent_raw.get("ancestors", [])
            # Fix: Preserve existing ancestors data when updating descendants
            parent.set_metadata(raw_key, {
                "descendants": parent_descendants,
                "ancestors": ancestor_block
            })
            parent.set_metadata(self.config.metadata_key, format_structured_insight_block(child_insights, ancestor_block))
            await self.storage.update(parent)

            # Only store immediate parent (depth 1 ancestors)
            child_ancestors = [child_insights]
            return await self._set_child_lineage_and_return(program, child_ancestors, child_insights, started_at, parent.id)

        except Exception as e:
            raise StageError(f"[{self.stage_name}] Failed: {e}") from e
