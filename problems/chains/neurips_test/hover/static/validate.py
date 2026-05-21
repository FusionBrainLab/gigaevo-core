"""Validate neurips_test/hover chain and compute soft retrieval coverage fitness."""

from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hover.utils.retrieval import retrieve as bm25_retrieve
from problems.chains.hover.utils.utils import (
    extract_titles_from_passages,
    normalize_text,
)
from problems.chains.neurips_test.hover.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.neurips_test.hover.static.config import (
    STATIC_CHAIN_TOPOLOGY,
    load_baseline,
)
from problems.chains.runner_config import RunnerConfig


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute soft retrieval coverage fitness.

    Soft (fractional) fitness: gold_found / n_gold per sample.
    Values: 0.0, 0.333, 0.667, or 1.0 for 3-hop HoVer (0/3, 1/3, 2/3, 3/3).

    Returns:
        Dict with fitness and is_valid.
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load dataset + retrieval context
    context = load_context(n_samples=300)
    dataset = context["train_dataset"]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build per-sample tool registry — each call: fn(query: str) -> str
    index_dir = context["bm25s_index_dir"]
    corpus_path = context["corpus_path"]
    tool_registry = {
        "retrieve": lambda query: bm25_retrieve(
            query, index_dir, k=7, corpus_path=corpus_path
        ),
        "retrieve_deep": lambda query: bm25_retrieve(
            query, index_dir, k=10, corpus_path=corpus_path
        ),
    }

    # 5. Run chain on dataset (RunnerConfig read from GIGAEVO_CHAIN_RUNNER_CONFIG env var)
    results = run_chain_on_dataset(
        chain,
        client,
        dataset,
        outer_context_builder,
        tool_registry=tool_registry,
        runner_config=RunnerConfig.from_env(),
    )

    # 6. Soft retrieval coverage: gold_found / n_gold per sample
    #    Tool step outputs are at indices 0 (step 1), 3 (step 4), 6 (step 7).
    scores = []
    for sample, result in zip(dataset, results):
        all_passages = "\n".join(
            [
                result.step_outputs[0],  # hop 1
                result.step_outputs[3],  # hop 2
                result.step_outputs[6],  # hop 3
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
