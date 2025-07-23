
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
You are an expert in performance-guided evolutionary programming.

You will analyze **localized code transitions** between a parent and a child Python program.
For each diff block, write a structured JSON object describing the effect of that change.

For each diff block, return:

- `strategy`: one of `"imitation"`, `"avoidance"`, or `"generalization"`
- `description`: ≤ 25 words describing how this change may have influenced the metric
- `impact`: a string representing the delta (e.g., "+0.017", "-0.005")

📌 You do **not** need to include the full diff again — it's already shown in the input prompt for each block.

Return a JSON **list of objects**, each object corresponding to a diff block.
Respond with **valid JSON only** — no markdown or extra text.
""".strip()


DEFAULT_USER_PROMPT_LINEAGE_TEXT = """
Analyze the following transition between two Python programs. Each block represents a localized code difference between a parent and a child.

The overall metric change is {delta:+.4f}.

For each block:
- Analyze the diff
- Identify the functional change
- Describe its likely causal effect
- Choose appropriate strategy: "imitation" (positive change to copy), "avoidance" (negative change to avoid), or "generalization" (neutral/structural change)

--- DIFF BLOCKS ---
{diff_blocks}
""".strip()


# def _format_code_reference(side: str, ref: Dict[str, Any]) -> List[str]:
#     """Format a code reference snippet if present."""
#     lines = []
#     if "code_snippet" in ref:
#         label = "Parent" if side == "parent_reference" else "Child"
#         lines.append(f"**{label} Code Snippet:**")
#         lines.append("```python")
#         lines.append(ref["code_snippet"].strip())
#         lines.append("```")
#     return lines


def _format_insight(insight: Dict[str, Any]) -> List[str]:
    """Format a single insight with its diff and references."""
    lines = []
    
    # Add diff block
    diff_content = insight.get('diff', '<diff omitted>').strip()
    if diff_content:
        lines.append("```diff")
        lines.append(diff_content)
        lines.append("```")
    
    # Add insight summary
    strategy = insight.get('strategy', 'unknown')
    description = insight.get('description', 'No description')
    impact = insight.get('impact', 'unknown')
    lines.append(f"- **[{strategy}]** {description} ({impact})")
    
    # # Add code references if present
    # for side in ["parent_reference", "child_reference"]:
    #     ref = insight.get(side, {})
    #     lines.extend(_format_code_reference(side, ref))
    
    return lines


def format_structured_insight_block(
    child_insights: Optional[Dict[str, Any]],
    ancestor_insights: List[Dict[str, Any]]
) -> str:
    """
    Format lineage insights into a structured markdown block.
    
    Args:
        child_insights: Insights about transitions to descendants (optional)
        ancestor_insights: List of insights about transitions from ancestors
        
    Returns:
        Formatted markdown string with ancestors and descendants sections
    """
    lines = []
    
    # Format ancestors section
    if ancestor_insights:
        lines.append("## ANCESTORS")
        for i, ancestor in enumerate(ancestor_insights):
            if i > 0:  # Add spacing between different ancestors
                lines.append("")
            
            from_id = ancestor.get('from', 'unknown')
            delta = ancestor.get('delta', 'unknown')
            lines.append(f"### From Parent `{from_id}` (Δ {delta}):")
            
            insights = ancestor.get("insights", [])
            for j, insight in enumerate(insights):
                if j > 0:  # Add spacing between insights
                    lines.append("")
                lines.extend(_format_insight(insight))
    
    # Format descendants section
    if child_insights:
        if lines:  # Add section separator if ancestors exist
            lines.append("")
        
        lines.append("## DESCENDANTS")
        to_id = child_insights.get('to', 'unknown')
        delta = child_insights.get('delta', 'unknown')
        lines.append(f"### To Child `{to_id}` (Δ {delta}):")
        
        insights = child_insights.get("insights", [])
        for i, insight in enumerate(insights):
            if i > 0:  # Add spacing between insights
                lines.append("")
            lines.extend(_format_insight(insight))
    
    # Handle empty case
    if not lines:
        lines.append("## LINEAGE")
        lines.append("*No lineage information available*")
    
    return "\n".join(lines)


class LineageInsightsConfig(BaseModel):
    llm_wrapper: LLMInterface
    metric_key: str
    metadata_key_raw: str = "lineage_insights_raw"
    metadata_key: str = "lineage_insights"
    parent_selection_strategy: Literal["first", "random", "best_fitness"] = "first"
    higher_is_better: bool = True
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT
    user_prompt_template: str = DEFAULT_USER_PROMPT_LINEAGE_TEXT
    fitness_selector_metric: Optional[str] = None
    fitness_selector_higher_is_better: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True


class GenerateLineageInsightsStage(Stage):
    MAX_DESCENDANTS = 3

    def __init__(self, config: LineageInsightsConfig, storage: ProgramStorage, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.storage = storage

    async def _select_parent_id(self, program: Program) -> Optional[str]:
        if not program.lineage or not program.lineage.parents:
            return None
        parents = program.lineage.parents
        if self.config.parent_selection_strategy == "first":
            return parents[0]
        elif self.config.parent_selection_strategy == "random":
            return random.choice(parents)
        elif self.config.parent_selection_strategy == "best_fitness":
            return await self._select_best_fitness_parent(parents)
        return parents[0]

    async def _select_best_fitness_parent(self, parent_ids: List[str]) -> Optional[str]:
        metric = self.config.fitness_selector_metric or self.config.metric_key
        better = self.config.fitness_selector_higher_is_better if self.config.fitness_selector_higher_is_better is not None else self.config.higher_is_better
        best_id, best_val = None, None
        for pid in parent_ids:
            parent = await self.storage.get(pid)
            try:
                val = float(parent.metrics[metric])
                if best_val is None or (val > best_val if better else val < best_val):
                    best_id, best_val = pid, val
            except Exception:
                continue
        return best_id

    def _compute_diff_blocks(self, parent_code: str, child_code: str) -> List[str]:
        """Compute diff blocks between parent and child code, filtering out meaningless changes."""
        
        # Handle identical code case
        if parent_code.strip() == child_code.strip():
            return []
            
        diff_lines = list(difflib.unified_diff(
            parent_code.strip().splitlines(),
            child_code.strip().splitlines(),
            lineterm="",
            n=3
        ))
        
        # If no diff generated, return empty
        if not diff_lines:
            return []
            
        # Remove file headers (--- and +++ lines) and find hunks
        diff_content = []
        for line in diff_lines:
            if not (line.startswith('---') or line.startswith('+++')):
                diff_content.append(line)
        
        # If no content after removing headers, return empty
        if not diff_content:
            return []
            
        # Split into hunks based on @@ lines
        hunks = []
        current_hunk = []
        
        for line in diff_content:
            if line.startswith('@@'):
                # Save previous hunk if it exists and has changes
                if current_hunk:
                    hunk_content = '\n'.join(current_hunk)
                    if self._hunk_has_changes(current_hunk):
                        hunks.append(hunk_content)
                # Start new hunk
                current_hunk = [line]
            else:
                current_hunk.append(line)
        
        # Don't forget the last hunk
        if current_hunk:
            hunk_content = '\n'.join(current_hunk)
            if self._hunk_has_changes(current_hunk):
                hunks.append(hunk_content)
        
        return hunks
    
    def _hunk_has_changes(self, hunk_lines: List[str]) -> bool:
        """Check if a hunk contains actual changes (+ or - lines)."""
        for line in hunk_lines:
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                return True
        return False

    def _is_meaningful_diff(self, diff_block: str) -> bool:
        """Check if a diff block contains actual changes (+ or - lines)."""
        return any(line.startswith(('+', '-')) and not line.startswith(('+++', '---')) for line in diff_block.splitlines())

    def _render_user_prompt(self, delta: float, blocks: List[str]) -> str:
        rendered_blocks = "\n\n".join([f"--- Block {i+1} ---\n```diff\n{b}\n```" for i, b in enumerate(blocks)])
        return self.config.user_prompt_template.format(delta=delta, diff_blocks=rendered_blocks)

    async def _execute_stage(self, program: Program, started_at: datetime):
        try:
            parent_id = await self._select_parent_id(program)
            if not parent_id:
                return build_stage_result(StageState.COMPLETED, started_at, "<No valid parent>", self.stage_name)
            parent = await self.storage.get(parent_id)
            if not parent:
                return build_stage_result(StageState.COMPLETED, started_at, f"<Parent {parent_id} not found>", self.stage_name)

            metric_key = self.config.metric_key
            pm = float(parent.metrics[metric_key])
            cm = float(program.metrics[metric_key])
            delta = cm - pm

            diff_blocks = self._compute_diff_blocks(parent.code, program.code)
            
            # Debug logging for diff blocks
            logger.debug(f"Generated {len(diff_blocks)} diff blocks for {parent.id} -> {program.id}")
            for i, block in enumerate(diff_blocks):
                logger.debug(f"Diff block {i+1} length: {len(block)} chars, meaningful: {self._is_meaningful_diff(block)}")
            
            # Skip insight generation if there are no meaningful differences
            if not diff_blocks:
                logger.info(f"No diff blocks found between parent {parent.id} and child {program.id}, skipping insight generation")
                return build_stage_result(
                    StageState.COMPLETED, 
                    started_at, 
                    "<No code differences found>", 
                    self.stage_name
                )
            
            prompt = self._render_user_prompt(delta, diff_blocks)

            response = await self.config.llm_wrapper.generate_async(prompt, system_prompt=self.config.system_prompt_template)
            try:
                parsed_insights = json.loads(response)
            except Exception as e:
                raise StageError(f"Failed to parse LLM JSON output: {e}\nRaw output:\n{response}") from e

            # Validate that we have insights
            if not parsed_insights:
                logger.warning(f"LLM returned no insights for parent {parent.id} -> child {program.id}")
                return build_stage_result(
                    StageState.COMPLETED,
                    started_at,
                    "<No insights generated>",
                    self.stage_name
                )

            # Associate each insight with its corresponding diff block (1:1 mapping)
            insights_with_diffs = []
            for i, insight in enumerate(parsed_insights):
                insight_with_diff = insight.copy()
                
                if i < len(diff_blocks):
                    # Use the corresponding diff block (1:1 mapping)
                    diff_block = diff_blocks[i]
                    
                    # Validate that the diff block has meaningful content
                    if self._is_meaningful_diff(diff_block):
                        insight_with_diff['diff'] = diff_block
                    else:
                        logger.warning(f"Diff block {i+1} is empty or contains only headers, using placeholder")
                        insight_with_diff['diff'] = f"<Empty diff block {i+1}>"
                else:
                    # More insights than diff blocks - log warning and use placeholder
                    logger.warning(f"Insight {i+1} has no corresponding diff block (only {len(diff_blocks)} blocks available)")
                    insight_with_diff['diff'] = f"<No diff block available for insight {i+1}>"
                
                insights_with_diffs.append(insight_with_diff)

            # Log mismatch if insights and diff blocks don't align with expected 1:1 ratio
            if len(parsed_insights) != len(diff_blocks):
                logger.warning(f"Expected {len(diff_blocks)} insights (1 per block) but got {len(parsed_insights)} for {parent.id} -> {program.id}")

            child_insights = {
                "from": parent.id,
                "to": program.id,
                "delta": f"{delta:+.4f}",
                "insights": insights_with_diffs,
            }

            raw_key = self.config.metadata_key_raw
            parent_raw = parent.metadata.get(raw_key, {})
            parent_descendants = parent_raw.get("descendants", [])
            if len(parent_descendants) >= self.MAX_DESCENDANTS:
                parent_descendants = parent_descendants[-(self.MAX_DESCENDANTS - 1):]
            parent_descendants.append(child_insights)
            parent.set_metadata(raw_key, {"descendants": parent_descendants})

            ancestor_block = parent_raw.get("ancestors", [])
            full_block = format_structured_insight_block(child_insights, ancestor_block)
            parent.set_metadata(self.config.metadata_key, full_block)
            await self.storage.update(parent)

            program.set_metadata(raw_key, {"ancestors": [child_insights]})
            program.set_metadata(self.config.metadata_key, format_structured_insight_block(None, [child_insights]))

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=full_block,
                stage_name=self.stage_name,
                metadata={
                    self.config.metadata_key: full_block,
                    "parent_id": parent.id,
                    "selected_parent_strategy": self.config.parent_selection_strategy,
                },
            )

        except Exception as e:
            raise StageError(f"[{self.stage_name}] Failed: {e}") from e
