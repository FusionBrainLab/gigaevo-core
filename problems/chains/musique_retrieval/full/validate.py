import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.musique_retrieval.full.config import FULL_CHAIN_CONFIG
from problems.chains.musique_retrieval.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.musique_retrieval.utils.failure_artifact import (
    build_failed_examples_artifact,
    is_alias_exact_match,
)
from problems.chains.musique_retrieval.utils.retrieval import make_retrieve_fn


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
        matches.append(int(is_alias_exact_match(pred, target_aliases)))

    return mean(matches) if matches else 0.0


def validate(chain_spec: dict) -> tuple[dict[str, float], str]:
    """Validate chain specification and return metrics plus failure artifact."""
    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    context = load_context(n_samples=100)
    dataset = context["train_dataset"]
    passages_by_task = {sample["task_id"]: sample["passages"] for sample in dataset}
    targets = [
        sample.get("answer_aliases", [sample.get("answer", "")]) for sample in dataset
    ]

    client = LLMClient(**LLM_CONFIG)
    tool_registry = {
        "retrieve": make_retrieve_fn(
            context["task_index_dir"],
            passages_by_task=passages_by_task,
            k=7,
        )
    }

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )
    predictions = [extract_answer(r.final_output) for r in results]

    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )
    fitness = calculate_exact_match(targets, predictions)
    artifact = build_failed_examples_artifact(
        dataset,
        targets,
        predictions,
        fitness=fitness,
        extraction_failures=extraction_failures,
    )

    return (
        {
            "fitness": fitness,
            "avg_extraction_failures": extraction_failures,
            "is_valid": 1,
        },
        artifact,
    )
