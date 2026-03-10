"""Validate MuSiQue chain specification and compute fitness metrics."""

import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.musique.full.config import FULL_CHAIN_CONFIG
from problems.chains.musique.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.musique.utils.utils import normalize_text


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def calculate_exact_match(
    targets: list[list[str]],
    predictions: list[str | None],
) -> float:
    """Calculate alias-aware Exact Match (EM) after text normalization."""
    matches = []

    for pred, target_aliases in zip(predictions, targets):
        if pred is None:
            matches.append(0)
            continue

        norm_pred = normalize_text(pred)
        norm_targets = {
            normalize_text(str(alias))
            for alias in target_aliases
            if str(alias).strip()
        }
        matches.append(int(norm_pred in norm_targets))

    return mean(matches) if matches else 0.0


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics."""
    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
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
