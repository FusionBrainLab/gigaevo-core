from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.musique_retrieval.full.validate import (
    calculate_exact_match,
    extract_answer,
)
from problems.chains.musique_retrieval.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.musique_retrieval.static.config import (
    STATIC_CHAIN_TOPOLOGY,
    load_baseline,
)
from problems.chains.musique_retrieval.utils.failure_artifact import (
    build_failed_examples_artifact,
)
from problems.chains.musique_retrieval.utils.retrieval import make_retrieve_fn


def validate(chain_spec: dict) -> tuple[dict[str, float], str]:
    """Validate chain specification and return metrics plus failure artifact."""
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    context = load_context(n_samples=300)
    dataset = context["train_dataset"]
    passages_by_task = {sample["task_id"]: sample["passages"] for sample in dataset}
    targets = [
        sample.get("answer_aliases", [sample.get("answer", "")])
        for sample in dataset
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
