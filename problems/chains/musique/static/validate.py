"""Validate MuSiQue chain specification and compute fitness metrics (static mode)."""

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.musique.full.validate import extract_answer, calculate_exact_match
from problems.chains.musique.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.musique.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics."""
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    context = load_context(n_samples=300)
    dataset = context["train_dataset"]
    targets = [
        sample.get("answer_aliases", [sample.get("answer", "")])
        for sample in dataset
    ]

    client = LLMClient(**LLM_CONFIG)
    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    predictions = [extract_answer(r.final_output) for r in results]
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    fitness = calculate_exact_match(targets, predictions)
    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
