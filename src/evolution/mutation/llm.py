"""LLM-driven mutation operator used by the evolutionary synthesis engine.

This module contains a concrete implementation of a `MutationOperator` that leverages a
large-language-model (LLM) to propose new candidate programs.  It supports both
rewrite-based and unified-diff mutation strategies and records detailed interaction
logs for debugging and research reproducibility purposes.
"""
from typing import Callable, List, Optional
from src.evolution.mutation.base import MutationOperator, MutationSpec
from src.programs.program import Program
from src.llm.wrapper import LLMInterface
from src.exceptions import MutationError
from loguru import logger
import re
import textwrap
import diffpatch
import os
import json
from datetime import datetime

def format_metrics(metrics: dict[str, float], metric_descriptions: dict[str, str]) -> str:
    """Convert a metrics dictionary into a human-readable bulleted string.

    If *metric_descriptions* is provided, each metric key will be annotated with a
    short textual description.  Missing descriptions are replaced with the
    placeholder "No description available".
    """
    assert set(metric_descriptions.keys()).issubset(set(metrics.keys())), "metric_descriptions is not a subset of metrics"
    return "\n".join(
        f"- {k}: {metrics[k]} ({v})"
        for k, v in metric_descriptions.items()
    )

# Create logs directory if it doesn't exist
LOGS_DIR = "llm_mutation_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def log_mutation_summary(mutation_id: str, prompt_length: int, response_length: int, code_length: int, success: bool):
    """Log a summary of each mutation to a master log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_file = f"{LOGS_DIR}/mutation_summary.log"
    
    try:
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | {mutation_id} | Prompt:{prompt_length}ch | Response:{response_length}ch | Code:{code_length}ch | Success:{success}\n")
    except Exception as e:
        logger.error(f"[LLMMutationOperator] Failed to write summary log: {e}")

def dump_llm_interaction(prompt: str, system_prompt: str, response: str, final_code: str, mutation_id: str):
    """Dump LLM interaction to a file for debugging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{LOGS_DIR}/mutation_{mutation_id}_{timestamp}.json"
    
    interaction_data = {
        "timestamp": timestamp,
        "mutation_id": mutation_id,
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "llm_response": response,
        "final_code": final_code,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "code_length": len(final_code),
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(interaction_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[LLMMutationOperator] Dumped interaction to {filename}")
    except Exception as e:
        logger.error(f"[LLMMutationOperator] Failed to dump interaction: {e}")

def dump_prompt_and_response_txt(prompt: str, system_prompt: str, response: str, final_code: str, mutation_id: str):
    """Also dump in a more readable text format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{LOGS_DIR}/mutation_{mutation_id}_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"LLM MUTATION OPERATOR LOG - {timestamp}\n")
            f.write(f"Mutation ID: {mutation_id}\n")
            f.write("="*80 + "\n\n")
            
            f.write("SYSTEM PROMPT:\n")
            f.write("-"*40 + "\n")
            f.write(system_prompt)
            f.write("\n\n")
            
            f.write("USER PROMPT:\n")
            f.write("-"*40 + "\n")
            f.write(prompt)
            f.write("\n\n")
            
            f.write("LLM RESPONSE:\n")
            f.write("-"*40 + "\n")
            f.write(response)
            f.write("\n\n")
            
            f.write("EXTRACTED FINAL CODE:\n")
            f.write("-"*40 + "\n")
            f.write(final_code)
            f.write("\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"System prompt length: {len(system_prompt)} chars\n")
            f.write(f"User prompt length: {len(prompt)} chars\n")
            f.write(f"LLM response length: {len(response)} chars\n")
            f.write(f"Final code length: {len(final_code)} chars\n")
            
        logger.info(f"[LLMMutationOperator] Dumped readable log to {filename}")
    except Exception as e:
        logger.error(f"[LLMMutationOperator] Failed to dump readable log: {e}")

SYSTEM_PROMPT_TEMPLATE = """
You are an AI research assistant embedded in an evolutionary code synthesis system.
Your role is to improve Python programs that solve complex mathematical or scientific problems.

TASK DEFINITION:
{task_definition}

DOMAIN HINTS:
{task_hints}

Your responsibilities:
- Propose meaningful, non-trivial changes to improve performance on defined metrics
- Generate only valid Python code — no comments or explanations in the output
- You may refactor, restructure, or replace key logic if beneficial
- Prefer innovative or creative strategies to minor edits or cleanups
- All randomness must be seeded
- DO NOT change the function signature of `run_code`, which is used for evaluation

Mutation pressure:
- Mutations that are too similar to the parent will likely be discarded
- Novel, high-impact structural ideas are favored
- You are competing with past and future mutations. To succeed, your code must innovate
""".strip()

REWRITE_PROMPT_TEMPLATE = """
Task:
Given one or more parent programs with their metrics and insights, generate a new Python program that may improve the metric of interest.

Requirements:
1. Return exactly one fenced Python code block: ```python ... ```
2. Do not include any text, comments, or explanations outside the code block
3. Output must be valid Python and directly executable

Exploration Guidelines:
- Make bold or creative changes when appropriate
- You may rewrite large parts of the logic if that improves performance
- Avoid trivial edits such as renaming variables or changing constants without reason
- Use insights and metrics to guide your changes
- Use lineage insights to guide your changes based on the program's evolutionary history and changes in the metric of interest

{parent_blocks}
""".strip()

DIFF_PROMPT_TEMPLATE = """
Task:
Given a parent Python program with associated metrics and insights, generate a non-trivial improvement in the form of a valid unified diff.

Requirements:
1. Return exactly one fenced code block labeled `diff`: ```diff
2. The diff must apply cleanly with `git apply --unidiff-zero`
3. Include 3 lines of context for each hunk; omit file headers or explanations

Exploration Guidelines:
- Substantial changes are encouraged — restructure logic if needed
- Use the insights and metrics to propose changes that are more than superficial
- Avoid minimal edits unless they're critical to performance
- You're part of an evolutionary process: originality is favored, redundancy is penalized

==== PARENT PROGRAM ====
```python
{parent_code}
==== METRICS ====
{parent_metrics}

==== INSIGHTS ====
{parent_insights}

==== LINEAGE INSIGHTS ====
{parent_lineage_insights}
""".strip()

class LLMMutationOperator(MutationOperator):
    def __init__(
        self,
        llm_wrapper: LLMInterface,
        metric_descriptions: dict[str, str],
        mutation_mode: str = "rewrite",
        fallback_to_rewrite: bool = True,
        fetch_insights_fn: Callable[[Program], str] = lambda x: x.metadata.get("insights", "No insights available."),
        fetch_lineage_insights_fn: Callable[[Program], str] = lambda x: x.metadata.get("lineage_insights", "No lineage insights available."),
        max_parents: int = 2,
        user_prompt_template_rewrite: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        task_definition: str = "The goal is to numerically approximate solutions to complex mathematical problems.",
        task_hints: str = "Prioritize numerical stability, convergence speed, and algorithmic originality.",
    ):
        self.llm_wrapper = llm_wrapper
        self.mutation_mode = mutation_mode
        self.fallback_to_rewrite = fallback_to_rewrite
        self.max_parents = 1 if mutation_mode == "diff" else max_parents
        self.fetch_insights_fn = fetch_insights_fn
        self.fetch_lineage_insights_fn = fetch_lineage_insights_fn
        self.user_prompt_template_rewrite = user_prompt_template_rewrite
        self.metric_descriptions = metric_descriptions

        if system_prompt_template is not None:
            self.system_prompt = system_prompt_template.format(
                task_definition=task_definition,
                task_hints=task_hints,
            )
        else:
            self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_definition=task_definition,
            task_hints=task_hints,
        )

    async def mutate_batch(
        self, parents: List[Program]
    ) -> List[MutationSpec]:
        mutations: List[MutationSpec] = []

        for i in range(0, len(parents), self.max_parents):
            selected_parents = parents[i : i + self.max_parents]

            try:
                if self.mutation_mode == "diff" and len(selected_parents) != 1:
                    raise MutationError("Diff-based mutation requires exactly 1 parent program")

                prompt = self._build_prompt(selected_parents)
                logger.debug(f"[LLMMutationOperator] Sending prompt (length: {len(prompt)} chars)")

                llm_response = await self.llm_wrapper.generate_async(prompt, system_prompt=self.system_prompt)

                if self.mutation_mode == "diff":
                    parent_code = selected_parents[0].code
                    try:
                        patched_code = self._apply_diff_and_extract(parent_code, llm_response)
                        final_code = patched_code
                    except Exception as diff_error:
                        logger.warning(f"[LLMMutationOperator] Failed to apply diff: {diff_error}")
                        if self.fallback_to_rewrite:
                            logger.warning("Diff failed, falling back to rewrite.")
                            fallback = LLMMutationOperator(self.llm_wrapper, mutation_mode="rewrite")
                            mutations += await fallback.mutate_batch(selected_parents)
                            continue
                        else:
                            raise
                else:
                    final_code = self._extract_code_block(llm_response)

                model_tag = self.llm_wrapper.model.replace("/", "_")
                label = f"llm_{self.mutation_mode}_{model_tag}"
                
                # DUMP LLM INTERACTION FOR DEBUGGING
                dump_llm_interaction(prompt, self.system_prompt, llm_response, final_code, label)
                dump_prompt_and_response_txt(prompt, self.system_prompt, llm_response, final_code, label)
                log_mutation_summary(label, len(prompt), len(llm_response), len(final_code), True)
                mutations.append(
                    MutationSpec(code=final_code.strip(), parents=selected_parents, name=label)
                )

            except Exception as e:
                logger.error(f"[LLMMutationOperator] Mutation failed: {e}")
                # Log failed mutation attempt
                failure_label = f"failed_mutation_{i}"
                log_mutation_summary(failure_label, 0, 0, 0, False)
                raise MutationError(f"LLM-based mutation failed: {e}") from e

        return mutations

    def _build_prompt(self, parents: List[Program]) -> str:
        if self.mutation_mode == "diff":
            logger.warning("Diff mode does not use user prompt template")
            p = parents[0]
            return DIFF_PROMPT_TEMPLATE.format(
                parent_code=p.code,
                parent_metrics=format_metrics(p.metrics, self.metric_descriptions),
                parent_insights=self.fetch_insights_fn(p),
                parent_lineage_insights=self.fetch_lineage_insights_fn(p),
            )

        parent_blocks = []
        for i, p in enumerate(parents):
            block = f"""=== Parent {i+1} ===
```python
{p.code}
```

=== Metrics ===
{format_metrics(p.metrics, self.metric_descriptions)}

=== Insights ===
{self.fetch_insights_fn(p)}

=== Lineage Insights ===
{self.fetch_lineage_insights_fn(p)}"""
            parent_blocks.append(block)

        if self.user_prompt_template_rewrite is not None:
            return self.user_prompt_template_rewrite.format(count=len(parents), parent_blocks="\n\n".join(parent_blocks))
        else:
            return REWRITE_PROMPT_TEMPLATE.format(count=len(parents), parent_blocks="\n\n".join(parent_blocks))

    def _extract_code_block(self, text: str) -> str:
        # Match ```python ... ``` or just ``` ... ```
        pattern = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else text.strip()

    def _apply_diff_and_extract(self, original_code: str, response_text: str) -> str:
        diff_text = self._extract_code_block(response_text)
        if not diff_text.strip():
            raise MutationError("Empty diff returned by LLM")

        try:
            return diffpatch.apply_patch(original_code, diff_text)
        except Exception as e:
            raise MutationError(f"Failed to apply patch: {e}") from e
