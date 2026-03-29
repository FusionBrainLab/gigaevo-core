"""Validate HoVer chain specification and compute soft (fractional) retrieval coverage fitness."""

from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hover.shared_config import (
    get_llm_config,
    load_context,
    outer_context_builder,
    release_chain_endpoint,
)
from problems.chains.hover.static_soft.config import (
    STATIC_CHAIN_TOPOLOGY,
    load_baseline,
)
from problems.chains.hover.utils.retrieval import make_retrieve_fn
from problems.chains.hover.utils.utils import (
    extract_titles_from_passages,
    normalize_text,
)


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute soft fitness metrics.

    Soft (fractional) fitness: gold_found / n_gold per sample.
    Values: 0.0, 0.333, 0.667, or 1.0 for 3-hop HoVer (0/3, 1/3, 2/3, 3/3).

    Returns a plain dict (no failure artifact) — uses pipeline=standard.
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

    # 3. Create LLM client (occupancy-based load balancing across chain servers)
    llm_config = get_llm_config()
    endpoint = llm_config["client_kwargs"]["base_url"]
    success = True
    try:
        client = LLMClient(**llm_config)

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
    except Exception:
        success = False
        raise
    finally:
        release_chain_endpoint(endpoint, success=success)

    # 6. Evaluate soft (fractional) retrieval coverage
    #    gold_found / n_gold per sample: 0/3, 1/3, 2/3, or 3/3.
    scores = []
    for sample, result in zip(dataset, results):
        all_passages = "\n".join(
            [
                result.step_outputs[0],  # Step 1 (hop 1)
                result.step_outputs[3],  # Step 4 (hop 2)
                result.step_outputs[6],  # Step 7 (hop 3)
            ]
        )
        found_titles = extract_titles_from_passages(all_passages)
        gold_titles = set(sample["supporting_facts"])
        normalized_gold = {normalize_text(t) for t in gold_titles}
        if normalized_gold:
            scores.append(len(normalized_gold & found_titles) / len(normalized_gold))
        else:
            scores.append(1.0)

    fitness = mean(scores) if scores else 0.0

    return {
        "fitness": fitness,
        "is_valid": 1,
    }
