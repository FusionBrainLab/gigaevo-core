"""Validate HoVer chain specification and compute retrieval coverage fitness."""

from statistics import mean

from problems.chains.chain_validation import validate_chain_spec
from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.client import LLMClient
from problems.chains.hover.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.hover.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hover.utils.retrieval import make_retrieve_fn
from problems.chains.hover.utils.utils import (
    extract_titles_from_passages,
    discrete_retrieval_eval,
)


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics.

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps

    Returns:
        Dict with fitness (retrieval coverage) and is_valid
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load context (dataset + retrieval paths)
    context = load_context(n_samples=300)
    dataset = context["train_dataset"]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build tool registry: two retrieve tools with different k
    tool_registry = {
        "retrieve": make_retrieve_fn(
            context["bm25s_index_dir"], k=7, corpus_path=context["corpus_path"]
        ),
        "retrieve_deep": make_retrieve_fn(
            context["bm25s_index_dir"], k=10, corpus_path=context["corpus_path"]
        ),
    }

    # 5. Run chain on dataset
    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )

    # 6. Evaluate retrieval coverage
    #    Collect passages from all 3 tool step outputs (steps 1, 4, 7 = indices 0, 3, 6).
    #    Indices are safe: static mode enforces exactly 7 steps with this topology.
    scores = []
    for sample, result in zip(dataset, results):
        all_passages = "\n".join(
            [
                result.step_outputs[0],  # Step 1 output
                result.step_outputs[3],  # Step 4 output
                result.step_outputs[6],  # Step 7 output
            ]
        )
        found_titles = extract_titles_from_passages(all_passages)
        gold_titles = set(sample["supporting_facts"])
        scores.append(discrete_retrieval_eval(gold_titles, found_titles))

    fitness = mean(scores) if scores else 0.0

    return {
        "fitness": fitness,
        "is_valid": 1,
    }
