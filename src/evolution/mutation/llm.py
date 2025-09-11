"""LLM-driven mutation operator used by the evolutionary synthesis engine.

This module contains a concrete implementation of a `MutationOperator` that leverages a
large-language-model (LLM) to propose new candidate programs.  It supports both
rewrite-based and unified-diff mutation strategies and records detailed interaction
logs for debugging and research reproducibility purposes.
"""

from datetime import datetime
import json
import os
import re
import random
from typing import Callable, List, Optional

import diffpatch
from loguru import logger

from src.evolution.mutation.base import MutationOperator, MutationSpec
from src.exceptions import MutationError
from src.llm.wrapper import LLMInterface
from src.programs.program import Program


def format_metrics(
    metrics: dict[str, float], metric_descriptions: dict[str, str]
) -> str:
    """Convert a metrics dictionary into a human-readable bulleted string.

    If *metric_descriptions* is provided, each metric key will be annotated with a
    short textual description.  Missing descriptions are replaced with the
    placeholder "No description available".
    """
    assert set(metric_descriptions.keys()).issubset(
        set(metrics.keys())
    ), "metric_descriptions is not a subset of metrics"
    return "\n".join(
        f"- {k}: {metrics[k]:.5f} ({v})" for k, v in metric_descriptions.items()
    )


# Create logs directory if it doesn't exist
LOGS_DIR = "llm_mutation_logs"
os.makedirs(LOGS_DIR, exist_ok=True)


def log_mutation_summary(
    mutation_id: str,
    prompt_length: int,
    response_length: int,
    code_length: int,
    success: bool,
):
    """Log a summary of each mutation to a master log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_file = f"{LOGS_DIR}/mutation_summary.log"

    try:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(
                f"{timestamp} | {mutation_id} | Prompt:{prompt_length}ch | Response:{response_length}ch | Code:{code_length}ch | Success:{success}\n"
            )
    except Exception as e:
        logger.error(f"[LLMMutationOperator] Failed to write summary log: {e}")


def dump_llm_interaction(
    prompt: str,
    system_prompt: str,
    response: str,
    final_code: str,
    mutation_id: str,
):
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
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(interaction_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[LLMMutationOperator] Dumped interaction to {filename}")
    except Exception as e:
        logger.error(f"[LLMMutationOperator] Failed to dump interaction: {e}")


def dump_prompt_and_response_txt(
    prompt: str,
    system_prompt: str,
    response: str,
    final_code: str,
    mutation_id: str,
):
    """Also dump in a more readable text format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{LOGS_DIR}/mutation_{mutation_id}_{timestamp}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"LLM MUTATION OPERATOR LOG - {timestamp}\n")
            f.write(f"Mutation ID: {mutation_id}\n")
            f.write("=" * 80 + "\n\n")

            f.write("SYSTEM PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(system_prompt)
            f.write("\n\n")

            f.write("USER PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
            f.write("\n\n")

            f.write("LLM RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(response)
            f.write("\n\n")

            f.write("EXTRACTED FINAL CODE:\n")
            f.write("-" * 40 + "\n")
            f.write(final_code)
            f.write("\n\n")

            f.write("STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"System prompt length: {len(system_prompt)} chars\n")
            f.write(f"User prompt length: {len(prompt)} chars\n")
            f.write(f"LLM response length: {len(response)} chars\n")
            f.write(f"Final code length: {len(final_code)} chars\n")

        logger.info(f"[LLMMutationOperator] Dumped readable log to {filename}")
    except Exception as e:
        logger.error(f"[LLMMutationOperator] Failed to dump readable log: {e}")


class LLMMutationOperator(MutationOperator):
    def __init__(
        self,
        *,
        llm_wrapper: LLMInterface,
        metric_descriptions: dict[str, str],
        mutation_mode: str = "rewrite",
        fallback_to_rewrite: bool = True,
        fetch_insights_fn: Callable[[Program], str] = lambda x: x.metadata.get(
            "insights", "No insights available."
        ),
        fetch_lineage_insights_fn: Callable[
            [Program], str
        ] = lambda x: x.metadata.get(
            "lineage_insights", "No lineage insights available."
        ),
        user_prompt_templates: list[str],
        user_prompt_template_weights_factory: Callable[[List[Program]], list[float]],
        system_prompt_template: str,
        task_definition: str = "The goal is to numerically approximate solutions to complex mathematical problems.",
        task_hints: str = "Prioritize numerical stability, convergence speed, and algorithmic originality.",
    ):
        self.llm_wrapper = llm_wrapper
        self.mutation_mode = mutation_mode
        self.fallback_to_rewrite = fallback_to_rewrite
        self.fetch_insights_fn = fetch_insights_fn
        self.fetch_lineage_insights_fn = fetch_lineage_insights_fn
        self.metric_descriptions = metric_descriptions
        self.system_prompt = system_prompt_template.format(
            task_definition=task_definition,
            task_hints=task_hints,
        )
        self.user_prompt_templates = user_prompt_templates
        self.user_prompt_template_weights_factory = user_prompt_template_weights_factory

    async def mutate_single(
        self, selected_parents: List[Program]
    ) -> Optional[MutationSpec]:
        """Generate a single mutation from the selected parents.

        Args:
            selected_parents: List of parent programs to mutate

        Returns:
            MutationSpec if successful, None if no mutation could be generated
        """
        if not selected_parents:
            logger.warning(
                f"[LLMMutationOperator] No parents provided for mutation"
            )
            return None

        try:
            if self.mutation_mode == "diff" and len(selected_parents) != 1:
                raise MutationError(
                    "Diff-based mutation requires exactly 1 parent program"
                )

            prompt = self._build_prompt(selected_parents)
            logger.debug(
                f"[LLMMutationOperator] Sending prompt (length: {len(prompt)} chars)"
            )

            llm_response = await self.llm_wrapper.generate_async(
                prompt, system_prompt=self.system_prompt
            )

            if self.mutation_mode == "diff":
                parent_code = selected_parents[0].code
                try:
                    patched_code = self._apply_diff_and_extract(
                        parent_code, llm_response
                    )
                    final_code = patched_code
                except Exception as diff_error:
                    logger.warning(
                        f"[LLMMutationOperator] Failed to apply diff: {diff_error}"
                    )
                    raise MutationError(
                        f"Diff application failed: {diff_error}"
                    ) from diff_error
            else:
                final_code = self._extract_code_block(llm_response)

            model_tag = self.llm_wrapper.model.replace("/", "_")
            parent_ids = "_".join([p.id[:8] for p in selected_parents])
            label = f"llm_{self.mutation_mode}_{model_tag}_{parent_ids}"

            # DUMP LLM INTERACTION FOR DEBUGGING
            dump_llm_interaction(
                prompt, self.system_prompt, llm_response, final_code, label
            )
            dump_prompt_and_response_txt(
                prompt, self.system_prompt, llm_response, final_code, label
            )
            log_mutation_summary(
                label, len(prompt), len(llm_response), len(final_code), True
            )

            mutation_spec = MutationSpec(
                code=final_code.strip(), parents=selected_parents, name=label
            )
            logger.debug(
                f"[LLMMutationOperator] Generated mutation from {len(selected_parents)} parents"
            )

            return mutation_spec

        except Exception as e:
            logger.error(f"[LLMMutationOperator] Mutation failed: {e}")
            # Log failed mutation attempt
            failure_label = f"failed_mutation_{parent_ids if 'parent_ids' in locals() else 'unknown'}"
            log_mutation_summary(failure_label, 0, 0, 0, False)
            raise MutationError(f"LLM-based mutation failed: {e}") from e

    def _build_prompt(self, parents: List[Program]) -> str:
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
{self.fetch_lineage_insights_fn(p)}
"""
            parent_blocks.append(block)
        weights = self.user_prompt_template_weights_factory(parents)
        assert len(weights) == len(self.user_prompt_templates), "Number of weights must match number of templates"
        template = random.choices(self.user_prompt_templates, weights=weights, k=1)[0]
        return template.format(
            count=len(parents), parent_blocks="\n\n".join(parent_blocks)
        )

    def _extract_code_block(self, text: str) -> str:
        """Extract the outer fenced code block, ignoring inner backticks.

        This method treats only fences that start at the beginning of a line as
        valid open/close markers. This avoids prematurely closing when triple
        backticks appear inside the code (e.g., within Python docstrings).
        """
        # Find the first opening fence at start-of-line: ` or ```python
        open_match = re.search(r"(?m)^```(?:[a-zA-Z0-9_+\-]+)?\s*$", text)
        if not open_match:
            return text.strip()

        # Search for the closing fence at start-of-line after the opener
        after_open = text[open_match.end():]
        close_match = re.search(r"(?m)^```\s*$", after_open)
        if not close_match:
            # If no proper closing fence, fall back to returning the whole text
            return text.strip()

        code_block = after_open[: close_match.start()]
        # Trim a single leading newline if present and trailing whitespace
        if code_block.startswith("\n"):
            code_block = code_block[1:]
        return code_block.rstrip()

    def _apply_diff_and_extract(
        self, original_code: str, response_text: str
    ) -> str:
        diff_text = self._extract_code_block(response_text)
        if not diff_text.strip():
            raise MutationError("Empty diff returned by LLM")

        try:
            return diffpatch.apply_patch(original_code, diff_text)
        except Exception as e:
            raise MutationError(f"Failed to apply patch: {e}") from e
