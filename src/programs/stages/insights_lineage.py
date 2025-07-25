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

You will analyze a transition between two Python programs based on a set of unified diff blocks. Your goal is to extract high-level causal insights about how the changes may have contributed to a metric change.

🎯 Return 3–5 short insights, each describing a structural or algorithmic change and its likely effect on performance.

Each insight must be:
- Prefixed with a `strategy`: "imitation", "avoidance", or "generalization"
- Followed by a concise `description` (≤ 25 words)

📦 Respond with a **JSON list** of objects like:
[
  {
    "strategy": "generalization",
    "description": "Replaced hardcoded parameters with computed values, enabling adaptive behavior."
  },
  ...
]

🛑 Do not include the diffs or metric change — those will be handled separately.

Respond with **valid JSON only** — no markdown or text commentary.
""".strip()

DEFAULT_USER_PROMPT_LINEAGE_TEXT = """
🔬 TASK CONTEXT:
{task_description}

📈 METRIC:
Optimization target = `{metric_name} ({metric_description})`
Delta = {delta:+.5f}

The following diff blocks show the changes from the parent to the child program:

{diff_blocks}

Analyze the diffs and summarize the key architectural or algorithmic changes that likely caused the observed metric shift.
""".strip()


def format_structured_insight_block(
    child_insights: Optional[Dict[str, Any]],
    ancestor_insights: List[Dict[str, Any]]
) -> str:
    lines = []

    if ancestor_insights:
        lines.append("## ANCESTORS")
        for ancestor in ancestor_insights:
            from_id = ancestor.get("from", "unknown")
            delta = ancestor.get("delta", "unknown")
            lines.append(f"### From Parent `{from_id}` (Δ {delta})")

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

    if child_insights:
        lines.append("## DESCENDANTS")
        to_id = child_insights.get("to", "unknown")
        delta = child_insights.get("delta", "unknown")
        lines.append(f"### To Child `{to_id}` (Δ {delta})")

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

    if not lines:
        lines.append("*No lineage information available*")

    return "\n".join(lines)


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
            if parent is None:
                continue
            try:
                val = float(parent.metrics[metric])
                if best_val is None or (val > best_val if better else val < best_val):
                    best_id, best_val = pid, val
            except Exception:
                continue
        return best_id

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
                return build_stage_result(StageState.COMPLETED, started_at, "<No valid parent>", self.stage_name)
            parent = await self.storage.get(parent_id)
            if not parent:
                return build_stage_result(StageState.COMPLETED, started_at, f"<Parent {parent_id} not found>", self.stage_name)

            metric_key = self.config.metric_key
            pm = float(parent.metrics[metric_key])
            cm = float(program.metrics[metric_key])
            delta = cm - pm

            diff_blocks = self._compute_diff_blocks(parent.code, program.code)
            if not diff_blocks:
                return build_stage_result(StageState.COMPLETED, started_at, "<No code differences found>", self.stage_name)

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

            program.set_metadata(raw_key, {"ancestors": [child_insights]})
            program.set_metadata(self.config.metadata_key, format_structured_insight_block(None, [child_insights]))

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=child_insights,
                stage_name=self.stage_name,
                metadata={
                    self.config.metadata_key: child_insights,
                    "parent_id": parent.id,
                    "selected_parent_strategy": self.config.parent_selection_strategy,
                },
            )

        except Exception as e:
            raise StageError(f"[{self.stage_name}] Failed: {e}") from e
