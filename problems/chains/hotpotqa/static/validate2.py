"""Fast validation for Optuna: random sampling + early stopping at baseline."""

import random
import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset_stepwise
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.hotpotqa.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hotpotqa.utils.retrieval import batch_retrieve
from problems.chains.hotpotqa.utils.utils import normalize_text

_ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern."""
    match = _ANSWER_RE.search(response)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def calculate_exact_match(
    targets: list[str],
    predictions: list[str | None],
) -> float:
    """Calculate Exact Match (EM) after text normalization."""
    matches = []
    for pred, target in zip(predictions, targets):
        if pred is None:
            matches.append(0)
            continue
        norm_pred = normalize_text(pred)
        norm_target = normalize_text(str(target))
        matches.append(int(norm_pred == norm_target))
    return mean(matches) if matches else 0.0


def validate(
    chain_spec: dict,
    n_samples: int = 100,
    early_stop_after: int = 30,
    baseline_fitness: float = 0.44,
    seed: int | None = 42,
) -> dict:
    """Fast validate for Optuna with random sampling and early stopping.

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps
        n_samples: Number of samples to evaluate (default 100 vs 300 in full)
        early_stop_after: Check EM after this many samples
        baseline_fitness: Early-stop if EM is below this (set to baseline EM)
        seed: Random seed for reproducible sampling (None for no seed)

    Returns:
        Dict with fitness, avg_extraction_failures, is_valid
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load full context, then randomly sample
    context = load_context(n_samples=None)
    full_dataset = context["train_dataset"]

    rng = random.Random(seed)
    dataset = rng.sample(full_dataset, min(n_samples, len(full_dataset)))
    targets = [s[context["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build batch tool registry for step-batched execution
    bm25_dir = context["bm25s_index_dir"]
    corpus_path = context["corpus_path"]

    def _batch_retrieve(kwargs_list: list[dict]) -> list[str]:
        queries = [kw["query"] for kw in kwargs_list]
        return batch_retrieve(queries, bm25_dir, k=7, corpus_path=corpus_path)

    batch_tool_registry = {"retrieve": _batch_retrieve}

    step_max_tokens = {
        2: 1024,   # summarize retrieved facts
        3: 1024,   # generate search query
        5: 1024,   # combine evidence
        6: 1024,   # final answer
    }

    # 5. Early stopping: run first batch, abort if below baseline
    early_dataset = dataset[:early_stop_after]
    early_targets = targets[:early_stop_after]

    early_results = run_chain_on_dataset_stepwise(
        chain, client, early_dataset, outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
    )
    early_preds = [extract_answer(r.final_output) for r in early_results]
    early_em = calculate_exact_match(early_targets, early_preds)

    if early_em < baseline_fitness:
        early_failures = (
            sum(1 for p in early_preds if p is None) / len(early_preds)
            if early_preds
            else 0.0
        )
        return {
            "fitness": early_em,
            "avg_extraction_failures": early_failures,
            "is_valid": 1,
        }

    # 6. Run remaining samples
    remaining_dataset = dataset[early_stop_after:]
    remaining_targets = targets[early_stop_after:]

    remaining_results = run_chain_on_dataset_stepwise(
        chain, client, remaining_dataset, outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
    )
    remaining_preds = [extract_answer(r.final_output) for r in remaining_results]

    # 7. Combine all predictions
    all_preds = early_preds + remaining_preds
    all_targets = early_targets + remaining_targets

    extraction_failures = (
        sum(1 for p in all_preds if p is None) / len(all_preds)
        if all_preds
        else 0.0
    )
    fitness = calculate_exact_match(all_targets, all_preds)

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
