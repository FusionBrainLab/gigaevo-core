"""Validate neurips_test/ifbench chain and compute constraint satisfaction fitness."""

from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.ifbench.utils.evaluation import test_instruction_following
from problems.chains.neurips_test.ifbench.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.neurips_test.ifbench.static.config import (
    STATIC_CHAIN_TOPOLOGY,
    load_baseline,
)
from problems.chains.runner_config import RunnerConfig


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute constraint satisfaction fitness.

    Returns:
        Dict with fitness (constraint satisfaction rate) and is_valid.
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load dataset
    context = load_context(n_samples=300)
    dataset = context["train_dataset"]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Run chain (no tools — pure LLM task)
    results = run_chain_on_dataset(
        chain,
        client,
        dataset,
        outer_context_builder,
        tool_registry=None,
        runner_config=RunnerConfig.from_env(),
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
