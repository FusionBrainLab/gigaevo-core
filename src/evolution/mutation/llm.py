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
from typing import Callable, Optional

import diffpatch
from loguru import logger

from src.evolution.mutation.base import MutationOperator, MutationSpec
from src.evolution.mutation.context import MutationContext
from src.exceptions import MutationError
from src.llm.agents.mutation import MutationAgent
from src.llm.models import MultiModelRouter
from src.llm.wrapper import LLMInterface  # Keep for type compatibility
from src.programs.program import Program
from src.programs.metrics.context import MetricsContext
from src.programs.metrics.formatter import MetricsFormatter



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
    """Mutation operator using LangGraph-based MutationAgent.
    
    This class maintains backward compatibility while using the new agent architecture.
    All existing interfaces and logging are preserved.
    """
    
    def __init__(
        self,
        *,
        llm_wrapper: LLMInterface | MultiModelRouter,
        metrics_context: MetricsContext,
        mutation_mode: str = "rewrite",
        fallback_to_rewrite: bool = True,
        context_key: str = "mutation_context",
        user_prompt_template: str,
        system_prompt_template: str,
        task_definition: str = "The goal is to numerically approximate solutions to complex mathematical problems.",
        task_hints: str = "Prioritize numerical stability, convergence speed, and algorithmic originality.",
    ):
        # Store configuration for compatibility and logging
        self.llm_wrapper = llm_wrapper
        self.mutation_mode = mutation_mode
        self.fallback_to_rewrite = fallback_to_rewrite
        self.context_key = context_key
        self.metrics_context = metrics_context
        
        # Format system prompt
        metrics_formatter = MetricsFormatter(metrics_context)
        metrics_description = metrics_formatter.format_metrics_description()
        self.system_prompt = system_prompt_template.format(
            task_definition=task_definition,
            task_hints=task_hints,
            metrics_description=metrics_description,
        )
        self.user_prompt_template = user_prompt_template
        
        # Create internal MutationAgent
        self.agent = MutationAgent(
            llm=llm_wrapper,
            mutation_mode=mutation_mode,
            system_prompt=self.system_prompt,
            user_prompt_template=user_prompt_template,
        )
        
        logger.info(
            f"[LLMMutationOperator] Initialized with mode: {mutation_mode} "
            "(using LangGraph agent)"
        )

    async def mutate_single(
        self, selected_parents: list[Program]
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

            logger.debug(
                f"[LLMMutationOperator] Running mutation agent for {len(selected_parents)} parents"
            )
            
            # Run agent graph
            result = await self.agent.arun(
                input=selected_parents,
                metadata={"mutation_mode": self.mutation_mode}
            )
            
            # Extract results from agent output
            if "code" not in result:
                raise ValueError(f"Missing 'code' key in agent result. Available keys: {list(result.keys())}")
            if "response" not in result:
                raise ValueError(f"Missing 'response' key in agent result. Available keys: {list(result.keys())}")
            
            final_code: str = result["code"]
            llm_response_text: str = result["response"]
            
            # Check if code extraction failed
            if not final_code or not final_code.strip():
                error_msg = result.get("error", "Unknown parsing error")
                raise ValueError(f"Failed to extract code from LLM response: {error_msg}")
            
            # Get model name and create label
            model_tag = getattr(self.llm_wrapper, "model", "unknown").replace("/", "_")
            parent_ids = "_".join([p.id[:8] for p in selected_parents])
            label = f"llm_{self.mutation_mode}_{model_tag}_{parent_ids}"

            # DUMP LLM INTERACTION FOR DEBUGGING (preserve existing logging)
            # We need to get the prompts from the agent's last state
            # For now, reconstruct the prompt for logging (backward compatibility)
            prompt = self._build_prompt(selected_parents)
            
            dump_llm_interaction(
                prompt, self.system_prompt, llm_response_text, final_code, label
            )
            dump_prompt_and_response_txt(
                prompt, self.system_prompt, llm_response_text, final_code, label
            )
            log_mutation_summary(
                label, len(prompt), len(llm_response_text), len(final_code), True
            )

            mutation_spec = MutationSpec(
                code=final_code.strip(), parents=selected_parents, name=label
            )
            logger.debug(
                f"[LLMMutationOperator] Generated mutation from {len(selected_parents)} parents "
                "(via LangGraph agent)"
            )

            return mutation_spec

        except Exception as e:
            logger.error(f"[LLMMutationOperator] Mutation failed: {e}")
            # Log failed mutation attempt
            parent_ids = "_".join([p.id[:8] for p in selected_parents]) if selected_parents else "unknown"
            failure_label = f"failed_mutation_{parent_ids}"
            log_mutation_summary(failure_label, 0, 0, 0, False)
            raise MutationError(f"LLM-based mutation failed: {e}") from e

    def _build_prompt(self, parents: list[Program]) -> str:
        """Build prompt for logging (uses MutationContext if available)."""
        parent_blocks = []
        for i, p in enumerate(parents):
            context: str = p.get_metadata(self.context_key)
            logger.debug("[LLMMutationOperator] _build_prompt() Context: {}", context)
            if context:
                block = f"""=== Parent {i+1} ===
```python
{p.code}
```

{context}
"""
            else:
                block = f"""=== Parent {i+1} ===
```python
{p.code}
```
"""

            parent_blocks.append(block)
        
        # Use the single user prompt template
        return self.user_prompt_template.format(
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
