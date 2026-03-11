import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.full.config import FULL_CHAIN_CONFIG
from problems.chains.hotpotqa.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.hotpotqa.utils.retrieval import make_retrieve_fn
from problems.chains.hotpotqa.utils.utils import normalize_text


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def calculate_exact_match(
    targets: list[str],
    predictions: list[str | None],
) -> float:
    """Calculate Exact Match (EM) after text normalization.

    Args:
        targets: List of gold answer strings
        predictions: List of predicted answer strings (None for extraction failures)

    Returns:
        EM score as a float in [0, 1]
    """
    matches = []

    for pred, target in zip(predictions, targets):
        if pred is None:
            matches.append(0)
            continue

        norm_pred = normalize_text(pred)
        norm_target = normalize_text(str(target))
        matches.append(int(norm_pred == norm_target))

    return mean(matches) if matches else 0.0


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics.

    Uses full_chain mode: no frozen steps, no topology matching.
    The LLM can freely rearrange step types, dependencies, and content
    within the constraints defined by FULL_CHAIN_CONFIG.

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps

    Returns:
        Dict with fitness, avg_extraction_failures, is_valid
    """
    # 1. Structural validation (full_chain mode — no topology, no frozen baseline)
    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    # 2. Load context (dataset + retrieval paths)
    context = load_context(n_samples=100)
    dataset = context["train_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build tool registry (index lazy-loaded from disk)
    tool_registry = {
        "retrieve": make_retrieve_fn(
            context["bm25s_index_dir"], k=7, corpus_path=context["corpus_path"]
        )
    }

    # 5. Run chain on dataset
    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )

    # 6. Extract answers from final step outputs
    predictions = [extract_answer(r.final_output) for r in results]

    # 7. Compute metrics
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
