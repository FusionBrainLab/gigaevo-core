"""Validate nlp/gsm8k chain and compute exact-match accuracy fitness."""

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.nlp.gsm8k.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.nlp.gsm8k.static.config import (
    STATIC_CHAIN_TOPOLOGY,
    load_baseline,
)
from problems.chains.nlp.gsm8k.utils.evaluation import (
    calculate_accuracy,
    extract_gsm8k_answer,
)
from problems.chains.runner_config import RunnerConfig


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute exact-match accuracy on GSM8K.

    The chain must output ``Answer: <number>`` in its final step.

    Returns:
        Dict with fitness (exact-match accuracy) and is_valid.
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
    context = load_context(n_samples=200)
    dataset = context["train_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Run chain (no tools — pure math reasoning)
    results = run_chain_on_dataset(
        chain,
        client,
        dataset,
        outer_context_builder,
        tool_registry=None,
        runner_config=RunnerConfig.from_env(),
    )

    # 5. Extract answers from final step outputs and compute accuracy
    predictions = [extract_gsm8k_answer(r.final_output) for r in results]

    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    fitness = calculate_accuracy(targets, predictions)

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
