"""Validate HoVer chain specification and compute retrieval coverage fitness."""

from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hover.shared_config import (
    get_llm_config,
    load_context,
    outer_context_builder,
)
from problems.chains.hover.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hover.utils.retrieval import make_retrieve_fn
from problems.chains.hover.utils.utils import (
    discrete_retrieval_eval,
    extract_titles_from_passages,
    normalize_text,
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
    client = LLMClient(**get_llm_config())

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
    failures = []
    for sample, result in zip(dataset, results):
        hop_passages = [
            result.step_outputs[0],  # Step 1 (hop 1)
            result.step_outputs[3],  # Step 4 (hop 2)
            result.step_outputs[6],  # Step 7 (hop 3)
        ]
        all_passages = "\n".join(hop_passages)
        found_titles = extract_titles_from_passages(all_passages)
        gold_titles = set(sample["supporting_facts"])
        normalized_gold = {normalize_text(t) for t in gold_titles}
        score = discrete_retrieval_eval(gold_titles, found_titles)
        scores.append(score)

        if score == 0:
            hop_titles = [extract_titles_from_passages(p) for p in hop_passages]
            hop_queries = [
                result.step_outputs[2] if len(result.step_outputs) > 2 else "",
                result.step_outputs[5] if len(result.step_outputs) > 5 else "",
            ]
            failures.append(
                {
                    "claim": sample.get("claim", ""),
                    "gold_titles": sorted(normalized_gold),
                    "n_gold": len(normalized_gold),
                    "hop1_found": sorted(normalized_gold & hop_titles[0]),
                    "hop1_missing": sorted(normalized_gold - hop_titles[0]),
                    "hop2_found": sorted(normalized_gold & hop_titles[1]),
                    "hop2_missing": sorted(normalized_gold - hop_titles[1]),
                    "hop3_found": sorted(normalized_gold & hop_titles[2]),
                    "hop3_missing": sorted(normalized_gold - hop_titles[2]),
                    "all_found": sorted(normalized_gold & found_titles),
                    "all_missing": sorted(normalized_gold - found_titles),
                    "hop2_query": hop_queries[0][:200],
                    "hop3_query": hop_queries[1][:200],
                }
            )

    fitness = mean(scores) if scores else 0.0
    metrics = {
        "fitness": fitness,
        "is_valid": 1,
    }

    return (metrics, failures)
