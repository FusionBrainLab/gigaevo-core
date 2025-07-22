from datetime import datetime
import random
from typing import List, Dict, Optional, Any, Literal
import difflib

from loguru import logger
from pydantic import BaseModel

from src.programs.program import Program, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.exceptions import StageError
from src.llm.wrapper import LLMInterface
from src.database.program_storage import ProgramStorage

DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT = """
You are an expert in performance-guided evolutionary programming.

Your task is to analyze a single **code transition** from a parent Python program to its mutated child. This transition includes a measured change in a key performance metric (e.g., fitness).

🎯 Generate **3 concise insights** explaining how specific code changes likely caused the metric change.

📌 Prefix each insight with one of:
- Lineage (imitation): → for ideas that worked and should be repeated
- Lineage (avoidance): → for failed strategies to avoid
- Lineage (generalization): → for potentially reusable or adaptable patterns

🧠 Each insight must:
- Include the **direction and magnitude** of the metric change (e.g., “+0.23”, “–0.17”)
- Reference a **specific architectural or algorithmic change** (use line numbers or function names when helpful)
- Be ≤ 25 words
- Avoid vague, stylistic, or speculative claims

✅ Example:
Lineage (imitation): +0.28 from line 35 — added jitter to spacing → reduced cluster collapse.""".strip()

DEFAULT_USER_PROMPT_LINEAGE_TEXT = """
Analyze the following code transition between a parent and child Python program.

The transition includes the metric delta and a unified diff. Write exactly **3 concise insights** explaining how structural code changes likely caused the metric change.

👇 Output:
- One line per insight
- Each line must start with: `Lineage (imitation):`, `Lineage (avoidance):`, or `Lineage (generalization):`
- Each insight must include the metric change (e.g., “+0.12”) and reference concrete code changes (line number, function, or operation)
- Use no more than 25 words per insight

--- TRANSITION CONTEXT ---
{examples_block}
""".strip()


class LineageInsightsConfig(BaseModel):
    llm_wrapper: LLMInterface
    metric_key: str
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
    MAX_CHILD_INSIGHT_BLOCKS = 5

    def __init__(self, config: LineageInsightsConfig, storage: ProgramStorage, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.storage = storage
        self.system_prompt = config.system_prompt_template or ""
        self.user_prompt_template = config.user_prompt_template or ""

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
        fitness_metric = self.config.fitness_selector_metric or self.config.metric_key
        higher_is_better = self.config.fitness_selector_higher_is_better
        if higher_is_better is None:
            higher_is_better = self.config.higher_is_better

        best_id, best_value = None, None
        for pid in parent_ids:
            parent = await self.storage.get(pid)
            if not parent or not parent.metrics or fitness_metric not in parent.metrics:
                continue
            try:
                val = float(parent.metrics[fitness_metric])
                if best_value is None or (val > best_value if higher_is_better else val < best_value):
                    best_id, best_value = pid, val
            except Exception:
                continue
        return best_id

    def _compute_diff(self, parent_code: str, child_code: str) -> str:
        return "\n".join(difflib.unified_diff(
            parent_code.strip().splitlines(),
            child_code.strip().splitlines(),
            lineterm=""
        ))

    def _create_transition(self, parent: Program, child: Program) -> Optional[Dict[str, Any]]:
        try:
            # Validate metric exists and is numeric
            if self.config.metric_key not in parent.metrics:
                logger.warning(f"Metric '{self.config.metric_key}' not found in parent {parent.id}")
                return None
            if self.config.metric_key not in child.metrics:
                logger.warning(f"Metric '{self.config.metric_key}' not found in child {child.id}")
                return None
            
            try:
                pm = float(parent.metrics[self.config.metric_key])
                cm = float(child.metrics[self.config.metric_key])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid metric values for '{self.config.metric_key}': parent={parent.metrics[self.config.metric_key]}, child={child.metrics[self.config.metric_key]} - {e}")
                return None
                
            delta = cm - pm
            improvement = delta > 0 if self.config.higher_is_better else delta < 0
            return {
                "from_id": parent.id,
                "to_id": child.id,
                "delta": delta,
                "improvement": improvement,
                "parent_value": pm,
                "child_value": cm,
                "diff": self._compute_diff(parent.code, child.code),
                "parent_error_summary": parent.get_all_errors_summary(),
                "child_error_summary": child.get_all_errors_summary(),
            }
        except Exception as e:
            logger.warning(f"Failed to compute transition: {e}")
            return None

    def _render_user_prompt(self, transition: Dict[str, Any]) -> str:
        block = f"""--- Transition ---
From program: {transition['from_id']}
To program: {transition['to_id']}
{self.config.metric_key}: {transition['parent_value']:.4f} → {transition['child_value']:.4f} (change: {transition['delta']:+.4f})
Result: {'Improved' if transition['improvement'] else 'Degraded'}

[Diff between Parent and Child Code]
```diff
{transition['diff']}
```

[Parent Error Summary]
{transition['parent_error_summary']}

[Child Error Summary]
{transition['child_error_summary']}
"""
        return self.user_prompt_template.format(
            metric_key=self.config.metric_key,
            examples_block=block
        )

    async def _execute_stage(self, program: Program, started_at: datetime):
        try:
            parent_id = await self._select_parent_id(program)
            if not parent_id:
                return build_stage_result(StageState.COMPLETED, started_at, "<No valid parent>", self.stage_name)

            parent = await self.storage.get(parent_id)
            if not parent:
                return build_stage_result(StageState.COMPLETED, started_at, f"<Parent {parent_id} not found>", self.stage_name)

            transition = self._create_transition(parent, program)
            if not transition:
                raise StageError("Transition creation failed")

            user_prompt = self._render_user_prompt(transition)
            response = await self.config.llm_wrapper.generate_async(user_prompt, system_prompt=self.system_prompt)
            insights = response.strip()

            # Store in child
            program.set_metadata(self.config.metadata_key, f"## ANCESTORS\n\n### Parent {parent.id} insights:\n{insights}")

            # Store in parent
            key = self.config.metadata_key
            existing = parent.metadata.get(key, "")
            descendant_entries = []

            if "## DESCENDANTS" in existing:
                parts = existing.split("## DESCENDANTS", maxsplit=1)
                if parts[1].strip():
                    # Split and filter out empty entries (first split result is often empty)
                    entries = parts[1].strip().split("\n\n### Child ")
                    descendant_entries = [entry for entry in entries if entry.strip()]

            if len(descendant_entries) >= self.MAX_CHILD_INSIGHT_BLOCKS:
                descendant_entries = descendant_entries[-(self.MAX_CHILD_INSIGHT_BLOCKS - 1):]

            descendant_entries.append(f"### Child {program.id} insights:\n{insights}")
            parent_metadata = f"## DESCENDANTS\n\n" + "\n\n".join(descendant_entries).strip()
            parent.set_metadata(key, parent_metadata)
            await self.storage.update(parent)

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=insights,
                stage_name=self.stage_name,
                metadata={
                    self.config.metadata_key: insights,
                    "parent_id": parent.id,
                    "selected_parent_strategy": self.config.parent_selection_strategy,
                },
            )

        except Exception as e:
            raise StageError(f"[{self.stage_name}] Failed: {e}") from e