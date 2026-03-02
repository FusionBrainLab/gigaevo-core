"""CARL-aligned chain execution engine.

Step-batched execution with history-based context and tool dispatch via
$-reference resolution. All samples process each step together, yielding
homogeneous LLM request batches for efficient vLLM batching.
"""

import asyncio
import re
from collections.abc import Callable

from problems.chains.types import ChainSpec, ChainResult, LLMStep, ToolStep


def _resolve_reference(
    ref: str,
    outer_context: str,
    step_outputs: list[str],
) -> str:
    """Resolve a $-reference to a concrete value.

    Supported syntax:
        $outer_context  — the original sample context string
        $history[-1]    — last completed step's output
        $history[N]     — step output at history index N (0-based)

    Args:
        ref: The $-reference string
        outer_context: Sample context string
        step_outputs: List of step outputs so far (0-indexed)

    Returns:
        Resolved string value
    """
    if ref == "$outer_context":
        return outer_context

    if ref == "$history[-1]":
        if not step_outputs:
            return ""
        return step_outputs[-1]

    match = re.match(r"\$history\[(\d+)\]", ref)
    if match:
        idx = int(match.group(1))
        if idx < len(step_outputs):
            return step_outputs[idx]
        return ""

    raise ValueError(f"Unknown reference syntax: {ref}")


def _resolve_dependencies(
    step_deps: list[int],
    history: list[str],
    step_outputs: list[str],
) -> tuple[list[str], dict[int, str]]:
    """Resolve dependency-filtered history and outputs for a step.

    Args:
        step_deps: List of dependency step numbers (1-based). Empty = all prior.
        history: Full accumulated history entries.
        step_outputs: Full accumulated step outputs.

    Returns:
        (visible_history, visible_outputs) filtered by dependencies
    """
    n_completed = len(step_outputs)

    if not step_deps:
        # Empty deps = see all prior steps
        visible_history = history[:n_completed]
        visible_outputs = {i + 1: step_outputs[i] for i in range(n_completed)}
    else:
        visible_history = []
        visible_outputs = {}
        for dep in step_deps:
            idx = dep - 1  # Convert 1-based to 0-based
            if 0 <= idx < n_completed:
                visible_history.append(history[idx])
                visible_outputs[dep] = step_outputs[idx]

    return visible_history, visible_outputs


# --- Stepwise helpers ---


async def _call_llm_with_semaphore(
    client,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call LLM with concurrency control."""
    async with semaphore:
        return await client(prompt)


def _execute_tool_step_batch(
    step: ToolStep,
    n: int,
    outer_contexts: list[str],
    all_step_outputs: list[list[str]],
    tool_registry: dict[str, Callable],
) -> list[str]:
    """Execute a tool step for all samples in one batch call.

    Resolves $-references for every sample, then calls the tool function
    once with the full list of resolved kwargs dicts.

    Always called via asyncio.to_thread() — runs in a thread pool, so tool
    functions that need async (e.g. external_llm using asyncio.run()) work
    cleanly.
    """
    tool_name = step.step_config.tool_name
    if tool_name not in tool_registry:
        raise ValueError(
            f"Tool '{tool_name}' not found in registry. "
            f"Available: {list(tool_registry.keys())}"
        )

    all_resolved = []
    for i in range(n):
        resolved = {
            param: _resolve_reference(ref, outer_contexts[i], all_step_outputs[i])
            for param, ref in step.step_config.input_mapping.items()
        }
        all_resolved.append(resolved)

    return tool_registry[tool_name](all_resolved)


async def _execute_llm_step_batch(
    step: LLMStep,
    chain: ChainSpec,
    n: int,
    outer_contexts: list[str],
    all_step_outputs: list[list[str]],
    all_histories: list[list[str]],
    client,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Execute an LLM step for all samples concurrently."""
    prompts = []
    for i in range(n):
        visible_history, _ = _resolve_dependencies(
            step.dependencies, all_histories[i], all_step_outputs[i]
        )
        prompts.append(
            chain.prompt_builder.build_prompt(
                step=step,
                visible_history=visible_history,
                outer_context=outer_contexts[i],
                system_prompt=chain.system_prompt,
            )
        )

    return list(
        await asyncio.gather(
            *(_call_llm_with_semaphore(client.copy(), p, semaphore) for p in prompts)
        )
    )


# --- Main execution ---


async def _run_chain_on_dataset_async(
    chain: ChainSpec,
    client,
    dataset: list[dict],
    outer_context_builder: Callable[[dict], str],
    tool_registry: dict[str, Callable] | None = None,
    max_concurrent: int = 32,
) -> list[ChainResult]:
    """Step-batched execution: all samples process each step together.

    Processes ALL samples through step 1, then ALL through step 2, etc.
    This yields homogeneous LLM request batches (same prompt structure,
    similar length) which vLLM can batch far more efficiently.

    Tool functions receive a list of resolved kwargs dicts and return a list
    of result strings (one per sample). Tools are called via asyncio.to_thread()
    so they can use asyncio.run() internally if needed.

    Args:
        chain: Validated ChainSpec with resolved steps
        client: LLMClient instance
        dataset: List of sample dicts
        outer_context_builder: Builds data context string from sample
        tool_registry: Dict mapping tool_name -> callable(list[dict]) -> list[str]
        max_concurrent: Max parallel LLM calls per step

    Returns:
        Ordered list of ChainResult (one per sample)
    """
    n = len(dataset)
    outer_contexts = [outer_context_builder(s) for s in dataset]
    all_step_outputs: list[list[str]] = [[] for _ in range(n)]
    all_histories: list[list[str]] = [[] for _ in range(n)]
    semaphore = asyncio.Semaphore(max_concurrent)

    for step in chain.steps:
        if isinstance(step, ToolStep):
            if tool_registry is None:
                raise ValueError(
                    f"Tool step {step.number} encountered but no tool_registry "
                    f"provided"
                )
            results = await asyncio.to_thread(
                _execute_tool_step_batch,
                step,
                n,
                outer_contexts,
                all_step_outputs,
                tool_registry,
            )
        elif isinstance(step, LLMStep):
            results = await _execute_llm_step_batch(
                step,
                chain,
                n,
                outer_contexts,
                all_step_outputs,
                all_histories,
                client,
                semaphore,
            )
        else:
            raise ValueError(f"Unknown step type: {type(step).__name__}")

        for i in range(n):
            all_step_outputs[i].append(results[i])
            all_histories[i].append(
                chain.prompt_builder.format_history_entry(
                    number=step.number,
                    title=step.title,
                    result=results[i],
                )
            )

    return [
        ChainResult(
            history=all_histories[i],
            final_output=all_step_outputs[i][-1] if all_step_outputs[i] else "",
            step_outputs=all_step_outputs[i],
        )
        for i in range(n)
    ]


def run_chain_on_dataset(
    chain: ChainSpec,
    client,
    dataset: list[dict],
    outer_context_builder: Callable[[dict], str],
    tool_registry: dict[str, Callable] | None = None,
    max_concurrent: int = 32,
) -> list[ChainResult]:
    """Run chain on dataset using step-batched execution (sync wrapper).

    Args:
        chain: Validated ChainSpec
        client: LLMClient instance
        dataset: List of sample dicts
        outer_context_builder: Builds data context string from sample
        tool_registry: Dict mapping tool_name -> batched tool function.
            Each function must accept a list of resolved-kwargs dicts
            (one per sample) and return a list of result strings (one per
            sample).  Signature: ``(items: list[dict]) -> list[str]``.
        max_concurrent: Max parallel LLM calls per step

    Returns:
        Ordered list of ChainResult (one per sample)
    """
    return asyncio.run(
        _run_chain_on_dataset_async(
            chain,
            client,
            dataset,
            outer_context_builder,
            tool_registry,
            max_concurrent,
        )
    )
