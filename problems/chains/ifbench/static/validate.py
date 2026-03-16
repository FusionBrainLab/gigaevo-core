"""Validate IFBench chain specification and compute fitness metrics."""

from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.ifbench.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.ifbench.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.ifbench.utils.evaluation import test_instruction_following


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics.

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps.

    Returns:
        Dict with fitness, is_valid.
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
    context = load_context(n_samples=300)
    dataset = context["train_dataset"]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Run chain on dataset (no tools)
    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    # 5. Evaluate constraint satisfaction
    scores = []
    for sample, result in zip(dataset, results):
        response = result.final_output
        if response and response.strip():
            score = test_instruction_following(sample, response)
        else:
            score = 0.0
        scores.append(score)

    fitness = mean(scores) if scores else 0.0

    return {
        "fitness": fitness,
        "is_valid": 1,
    }
