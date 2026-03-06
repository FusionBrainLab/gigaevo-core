"""Validate PAPILLON chain specification and compute fitness metrics.

Fitness = (avg_quality + (1 - avg_leakage)) / 2

Quality is judged by GPT-4o-mini (bidirectional comparison with target).
Leakage is judged by GPT-4o-mini (PII count in sanitized query).
"""

import asyncio
from statistics import mean

from problems.chains.chain_validation import validate_chain_spec
from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.client import LLMClient
from problems.chains.papillon.shared_config import (
    LLM_CONFIG,
    JUDGE_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.papillon.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.papillon.utils.external_llm import make_external_llm_fn
from problems.chains.papillon.utils.pipeline import judge_quality, judge_leakage


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics.

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps.

    Returns:
        Dict with fitness, avg_quality, avg_leakage, is_valid.
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load context (dataset)
    context = load_context(n_samples=150)
    dataset = context["train_dataset"]

    # 3. Execute chain (redact → external_llm → aggregate)
    client = LLMClient(**LLM_CONFIG)
    tool_registry = {"external_llm": make_external_llm_fn(LLM_CONFIG)}
    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )

    # 4. Evaluate quality and leakage via judge LLM
    #    step_outputs[0] = Step 1 output (sanitized query)
    #    final_output    = Step 3 output (final response)
    async def judge_all():
        judge_client = LLMClient(**JUDGE_CONFIG)
        sem = asyncio.Semaphore(32)

        async def judge_one(sample, result):
            async with sem:
                quality = await judge_quality(
                    judge_client.copy(),
                    sample["user_query"],
                    result.final_output,
                    sample[context["target_field"]],
                )
                leakage = await judge_leakage(
                    judge_client.copy(),
                    result.step_outputs[0],
                    sample.get(context["pii_field"], ""),
                )
                return quality, leakage

        return await asyncio.gather(
            *(judge_one(s, r) for s, r in zip(dataset, results))
        )

    judge_results = asyncio.run(judge_all())

    avg_quality = mean(q for q, _ in judge_results)
    avg_leakage = mean(lk for _, lk in judge_results)
    fitness = (avg_quality + (1 - avg_leakage)) / 2

    return {
        "fitness": fitness,
        "avg_quality": avg_quality,
        "avg_leakage": avg_leakage,
        "is_valid": 1,
    }
