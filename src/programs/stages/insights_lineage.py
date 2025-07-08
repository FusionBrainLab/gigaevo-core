from datetime import datetime
import random
from typing import List, Dict, Optional, Literal, Any

from pydantic import BaseModel
from loguru import logger

from src.programs.program import Program, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result
from src.exceptions import StageError
from src.llm.wrapper import LLMInterface
from src.database.program_storage import ProgramStorage

DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT = """
You are an expert in evolutionary programming and performance-guided software optimization.

You will analyze a series of **code transitions** representing mutations from a **parent** to a **child** Python program, along with the observed change in a key performance metric (e.g., fitness).

🎯 Your task:
For each transition, generate **3 concise, causal insights** explaining how structural code changes likely caused the observed metric change.

📌 Prefix each insight with one of:
- Lineage (imitation): → for strategies worth **repeating**
- Lineage (avoidance): → for failed ideas to **avoid**
- Lineage (generalization): → for useful patterns that should be **expanded or modified**

🧠 Insight Criteria:
- Must mention the **direction and magnitude** of the metric change (e.g., "+0.25", "–0.12")
- Must reference a **concrete architectural or algorithmic change**
- Avoid vague, stylistic, or speculative commentary
- ≤ 25 words per insight

✅ Format example:
Lineage (imitation): +0.31 from golden-angle spiral → reduced voids and improved central compactness.
Lineage (avoidance): –0.18 due to hard-coded angles → rigid hex orientations reduced adaptability.
Lineage (generalization): +0.24 from boundary clustering → apply with loosened arc spacing for flexibility.

These insights will guide future LLM-based mutations. Ensure precision, traceability, and structural clarity.
"""

DEFAULT_USER_PROMPT_LINEAGE_TEXT = """
Analyze the following **code transitions** between parent and child Python programs.

Each transition includes the observed change in **{metric_key}**, e.g., "+0.23".

Your task:
- For each transition, write **3 concise insights**
- Each must explain how a structural change in code likely caused the metric delta
- Begin each insight with one of:
  - Lineage (imitation):
  - Lineage (avoidance):
  - Lineage (generalization):

🧠 Each insight must:
- Include the metric delta (e.g., "+0.23", "–0.15")
- Mention the **specific design/algorithmic factor** responsible
- Be ≤ 25 words
- Avoid redundant phrasing or vague speculation

👇 Output ONLY 3 insight lines per transition. DO NOT repeat instructions or format. No extra text.

Transitions (each shows a parent → child mutation with metric delta, oldest to newest):
{examples_block}
"""


class LineageInsightsConfig(BaseModel):
    """Configuration for lineage insights generation with flexible parent selection.
    
    PARENT SELECTION STRATEGIES:
    - "first": Always select the first parent (fastest, deterministic)
    - "random": Randomly select among available parents (introduces variability)
    - "best_fitness": Select parent with best fitness value (most informed choice)
    
    EXAMPLES:
    
    1. Simple lineage analysis using fitness metric:
        config = LineageInsightsConfig(
            llm_wrapper=llm,
            metric_key="fitness",
            parent_selection_strategy="best_fitness"
        )
    
    2. Select parents based on different metric than analysis metric:
        config = LineageInsightsConfig(
            llm_wrapper=llm,
            metric_key="fitness",  # Metric to analyze for insights
            parent_selection_strategy="best_fitness",
            fitness_selector_metric="ast_entropy",  # Use entropy to select best parent
            fitness_selector_higher_is_better=False  # Lower entropy is better
        )
    
    3. Analyze negative fitness but select parents with highest fitness:
        config = LineageInsightsConfig(
            llm_wrapper=llm,
            metric_key="fitness",
            higher_is_better=True,  # For insight analysis: higher fitness is better
            parent_selection_strategy="best_fitness",
            fitness_selector_higher_is_better=True  # Select parent with highest fitness
        )
    """
    llm_wrapper: LLMInterface
    metric_key: str  # Primary metric to analyze for insights
    max_ancestors: int = 5
    metadata_key: str = "lineage_insights"
    output_format: Literal["text"] = "text"
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None
    metric_description: Optional[str] = None
    higher_is_better: bool = True  # Whether higher values of metric_key represent improvement
    parent_selection_strategy: Literal["first", "random", "best_fitness"] = "first"
    
    # Fitness selector configuration (only used when parent_selection_strategy="best_fitness")
    fitness_selector_metric: Optional[str] = None  # Metric to use for parent selection (defaults to metric_key)
    fitness_selector_higher_is_better: Optional[bool] = None  # Direction for parent selection (defaults to higher_is_better)

    class Config:
        arbitrary_types_allowed = True


class GenerateLineageInsightsStage(Stage):

    def __init__(
        self,
        config: LineageInsightsConfig,
        storage: ProgramStorage,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.storage = storage

        self.system_prompt = (
            config.system_prompt_template or DEFAULT_SYSTEM_PROMPT_LINEAGE_TEXT
        )
        self.user_prompt_template = (
            config.user_prompt_template or DEFAULT_USER_PROMPT_LINEAGE_TEXT
        )

    async def _select_parent_id(self, current: Program) -> Optional[str]:
        """Select which parent to follow based on configuration strategy."""
        if not current.lineage or not current.lineage.parents:
            return None
            
        parents = current.lineage.parents
        
        if self.config.parent_selection_strategy == "first":
            return parents[0]
        elif self.config.parent_selection_strategy == "random":
            return random.choice(parents)
        elif self.config.parent_selection_strategy == "best_fitness":
            return await self._select_best_fitness_parent(parents)
        else:
            return parents[0]
    
    async def _select_best_fitness_parent(self, parent_ids: List[str]) -> Optional[str]:
        """Select the parent with the best fitness value among multiple parents."""
        if not parent_ids:
            return None
        
        if len(parent_ids) == 1:
            return parent_ids[0]
        
        # Use fitness selector configuration if provided, otherwise fallback to main config
        fitness_metric = self.config.fitness_selector_metric or self.config.metric_key
        fitness_higher_is_better = (
            self.config.fitness_selector_higher_is_better 
            if self.config.fitness_selector_higher_is_better is not None 
            else self.config.higher_is_better
        )
        
        best_parent_id = None
        best_fitness = None
        valid_parents = 0
        
        for parent_id in parent_ids:
            try:
                parent = await self.storage.get(parent_id)
                if parent is None:
                    logger.debug(f"Parent {parent_id} not found in storage, skipping")
                    continue
                
                if not hasattr(parent, 'metrics') or parent.metrics is None:
                    logger.debug(f"Parent {parent_id} has no metrics, skipping")
                    continue
                
                fitness_value = parent.metrics.get(fitness_metric)
                if fitness_value is None:
                    logger.debug(f"Parent {parent_id} missing metric '{fitness_metric}', skipping")
                    continue
                
                try:
                    fitness_value = float(fitness_value)
                    valid_parents += 1
                    
                    # Compare based on whether higher or lower is better
                    if best_fitness is None:
                        best_fitness = fitness_value
                        best_parent_id = parent_id
                    else:
                        is_better = (
                            fitness_value > best_fitness if fitness_higher_is_better
                            else fitness_value < best_fitness
                        )
                        if is_better:
                            best_fitness = fitness_value
                            best_parent_id = parent_id
                            
                except (ValueError, TypeError):
                    logger.debug(f"Parent {parent_id} has invalid fitness value: {fitness_value}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error retrieving parent {parent_id} for fitness comparison: {e}")
                continue
        
        if best_parent_id is None:
            logger.warning(f"No valid parents found for fitness comparison, falling back to first parent")
            return parent_ids[0]
        
        selection_info = (
            f"best {'highest' if fitness_higher_is_better else 'lowest'} {fitness_metric}"
            if fitness_metric != self.config.metric_key
            else f"best {fitness_metric}"
        )
        
        logger.debug(f"Selected parent {best_parent_id} with {selection_info}={best_fitness:.4f} "
                    f"from {valid_parents} valid parents out of {len(parent_ids)} total")
        
        return best_parent_id

    async def build_lineage(self, program: Program) -> List[Program]:
        """Build lineage chain from current program back to root, with robust error handling."""
        if program.is_root:
            return []
        
        lineage = [program]
        current = program
        visited_ids = {program.id}  # Prevent infinite loops
        failed_retrievals = 0
        
        while (not current.is_root and 
               len(lineage) < self.config.max_ancestors and 
               current.lineage is not None):
            
            # Validate lineage structure
            if not hasattr(current.lineage, 'parents') or not current.lineage.parents:
                logger.debug(f"Program {current.id} has no parents in lineage, stopping")
                break
            
            # Select parent using configured strategy
            parent_id = await self._select_parent_id(current)
            if parent_id is None:
                logger.debug(f"No valid parent selected for {current.id}, stopping")
                break
            
            # Prevent cycles (shouldn't happen but be defensive)
            if parent_id in visited_ids:
                logger.warning(f"Detected lineage cycle at program {parent_id}, stopping")
                break
            
            try:
                # Attempt to retrieve parent from storage
                parent = await self.storage.get(parent_id)
                if parent is None:
                    failed_retrievals += 1
                    logger.warning(f"Parent program {parent_id} not found in storage, stopping lineage")
                    break
                
                # Validate parent has required attributes
                if not hasattr(parent, 'metrics') or parent.metrics is None:
                    logger.warning(f"Parent program {parent_id} has no metrics, stopping lineage")
                    break
                
                # Validate parent has the required metric
                if self.config.metric_key not in parent.metrics:
                    logger.warning(f"Parent program {parent_id} missing metric '{self.config.metric_key}', stopping lineage")
                    break
                
                lineage.append(parent)
                visited_ids.add(parent_id)
                current = parent
                
            except Exception as e:
                failed_retrievals += 1
                logger.warning(f"Failed to retrieve parent {parent_id}: {e}, stopping lineage")
                break
        
        if len(lineage) <= 1:
            logger.debug(f"Program {program.id} has no valid lineage (only self)")
            return []
        
        # Log lineage statistics for debugging
        logger.info(f"Built lineage for {program.id}: {len(lineage)} programs, "
                   f"{failed_retrievals} failed retrievals, "
                   f"path: {' → '.join([p.id[:8] for p in reversed(lineage)])}")
        
        # Return from oldest to newest (reverse order)
        return list(reversed(lineage))

    def get_lineage_stats(self, lineage: List[Program]) -> Dict[str, Any]:
        """Get statistics about the lineage for debugging and validation."""
        if not lineage:
            return {"length": 0, "valid": False}
        
        stats = {
            "length": len(lineage),
            "valid": True,
            "program_ids": [p.id for p in lineage],
            "has_metrics": all(hasattr(p, 'metrics') and p.metrics for p in lineage),
            "has_target_metric": all(self.config.metric_key in (p.metrics or {}) for p in lineage),
        }
        
        # Check metric progression
        if stats["has_target_metric"]:
            values = [p.metrics[self.config.metric_key] for p in lineage]
            stats["metric_values"] = values
            stats["metric_changes"] = [values[i+1] - values[i] for i in range(len(values)-1)]
            stats["total_change"] = values[-1] - values[0] if len(values) > 1 else 0
        
        return stats

    async def _execute_stage(self, program: Program, started_at: datetime):
        try:
            lineage = await self.build_lineage(program)
            if not lineage:
                program.set_metadata(self.config.metadata_key, "<Program has no lineage>")
                return build_stage_result(
                    status=StageState.COMPLETED,
                    started_at=started_at,
                    output="<Program has no lineage>",
                    stage_name=self.stage_name,
                    metadata={self.config.metadata_key: "<Program has no lineage>"},
                )

            # Get lineage statistics for debugging
            lineage_stats = self.get_lineage_stats(lineage)
            logger.debug(f"Lineage stats for {program.id}: {lineage_stats}")
            
            # Validate lineage quality
            if not lineage_stats["has_target_metric"]:
                error_msg = f"Lineage incomplete: missing '{self.config.metric_key}' metric in some programs"
                logger.warning(error_msg)
                program.set_metadata(self.config.metadata_key, f"<{error_msg}>")
                return build_stage_result(
                    status=StageState.COMPLETED,
                    started_at=started_at,
                    output=error_msg,
                    stage_name=self.stage_name,
                    metadata={
                        self.config.metadata_key: f"<{error_msg}>",
                        "lineage_stats": lineage_stats
                    },
                )

            transitions = []
            metric_key = self.config.metric_key

            for parent, child in zip(lineage[:-1], lineage[1:]):
                try:
                    parent_metric = parent.metrics.get(metric_key)
                    child_metric = child.metrics.get(metric_key)
                    
                    if parent_metric is None or child_metric is None:
                        logger.warning(f"Missing {metric_key} metric for parent {parent.id} or child {child.id}")
                        continue
                    
                    parent_metric = float(parent_metric)
                    child_metric = float(child_metric)
                    
                    delta = child_metric - parent_metric
                    
                    if self.config.higher_is_better:
                        improvement = delta > 0
                        interpretation = "improved" if improvement else "degraded" 
                    else:
                        improvement = delta < 0
                        interpretation = "improved" if improvement else "degraded"
                    
                    transitions.append({
                        "from_id": parent.id,
                        "to_id": child.id,
                        "delta": delta,
                        "improvement": improvement,
                        "interpretation": interpretation,
                        "parent_value": parent_metric,
                        "child_value": child_metric,
                        "parent_code": parent.code,
                        "child_code": child.code,
                        "parent_error_summary": parent.get_all_errors_summary(),
                        "child_error_summary": child.get_all_errors_summary(),
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid metric values for {metric_key}: {e}")
                    continue

            if not transitions:
                raise StageError("No valid metric transitions for lineage analysis.")

            logger.info(f"Generated {len(transitions)} transitions for lineage analysis of {program.id}")

            user_prompt = self._render_user_prompt(transitions)
            response = await self.config.llm_wrapper.generate_async(
                user_prompt, system_prompt=self.system_prompt
            )
            parsed = response.strip()
            if not parsed:
                raise StageError("LLM returned empty response")

            program.set_metadata(self.config.metadata_key, parsed)

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=parsed,
                stage_name=self.stage_name,
                metadata={
                    self.config.metadata_key: parsed,
                    f"{self.config.metadata_key}_parents_used": [p.id for p in lineage],
                    "lineage_stats": lineage_stats,
                    "transitions_analyzed": len(transitions),
                },
            )

        except Exception as e:
            raise StageError(f"[{self.stage_name}] Lineage insight generation failed: {e}") from e

    def _render_user_prompt(self, transitions: List[Dict[str, Any]]) -> str:
        metric_key = self.config.metric_key
        metric_description = self.config.metric_description or f"the {metric_key} metric"
        higher_better = "higher values are better" if self.config.higher_is_better else "lower values are better"
        
        blocks = []
        for t in transitions:
            delta_str = f"{t['delta']:+.4f}"
            improvement_str = "🟢 IMPROVEMENT" if t['improvement'] else "🔴 DEGRADATION"
            
            blocks.append(f"""--- Transition ---
From program: {t['from_id']} 
To program: {t['to_id']}
{metric_key}: {t['parent_value']:.4f} → {t['child_value']:.4f} (change: {delta_str})
Result: {improvement_str} - Performance {t['interpretation']}

[Parent Code]
```python
{t['parent_code']}
```

[Child Code]
```python
{t['child_code']}
```

[Parent Error Summary]
{t['parent_error_summary']}

[Child Error Summary]
{t['child_error_summary']}
""")
        
        context_info = f"""
METRIC CONTEXT:
- {metric_key}: {metric_description}
- Direction: {higher_better}
- Analyze how code changes led to performance improvements or degradations
"""
        
        return context_info + self.user_prompt_template.format(
            examples_block="\n".join(blocks), metric_key=metric_key
        )
