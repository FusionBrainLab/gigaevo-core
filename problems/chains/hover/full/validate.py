"""Validate HoVer chain specification and compute soft (fractional) retrieval coverage fitness.

Full-chain mode: no frozen steps, no topology matching. The LLM can freely
rearrange step count, types, dependencies, and content within FULL_CHAIN_CONFIG
constraints. Retrieval scoring is adaptive — scans ALL tool-step outputs for
gold article matches, regardless of their position in the chain.
"""

from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hover.full.config import FULL_CHAIN_CONFIG
from problems.chains.hover.shared_config import (
    get_llm_config,
    load_context,
    outer_context_builder,
)
from problems.chains.hover.utils.retrieval import make_retrieve_fn
from problems.chains.hover.utils.utils import (
    extract_titles_from_passages,
    normalize_text,
)


def evaluate_soft_coverage_adaptive(dataset, results, chain):
    """Compute soft retrieval coverage by scanning ALL tool-step outputs.

    Unlike the static evaluator which hardcodes step indices [0, 3, 6],
    this scans every tool-type step in the chain and collects all retrieved
    passages. This allows dynamic chains with variable topology to be scored.

    Args:
        dataset: List of preprocessed samples with "supporting_facts".
        results: List of ChainResult from run_chain_on_dataset.
        chain: Validated chain object (to identify tool-step indices).

    Returns:
        List of per-sample soft coverage scores (0.0 to 1.0).
    """
    tool_indices = [i for i, step in enumerate(chain.steps) if step.step_type == "tool"]

    scores = []
    for sample, result in zip(dataset, results):
        # Collect passages from ALL tool steps
        all_passages = "\n".join(
            result.step_outputs[i] for i in tool_indices if i < len(result.step_outputs)
        )
        found_titles = extract_titles_from_passages(all_passages)
        gold_titles = set(sample["supporting_facts"])
        normalized_gold = {normalize_text(t) for t in gold_titles}
        if normalized_gold:
            scores.append(len(normalized_gold & found_titles) / len(normalized_gold))
        else:
            scores.append(1.0)
    return scores


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute soft fitness metrics.

    Uses full_chain mode: no frozen steps, no topology matching.
    Soft (fractional) fitness: gold_found / n_gold per sample.
    Adaptive scoring: scans all tool-step outputs for gold articles.

    Returns a plain dict (no failure artifact) -- uses pipeline=standard.
    """
    # 0. Early step-count check (reject before expensive chain eval)
    steps = chain_spec.get("steps", [])
    max_steps = FULL_CHAIN_CONFIG["max_steps"]
    if len(steps) > max_steps:
        return {"fitness": 0.0, "is_valid": 0, "n_steps": len(steps), "n_tool_steps": 0}

    # 1. Structural validation (full_chain mode)
    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
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

    # 6. Adaptive soft coverage scoring
    scores = evaluate_soft_coverage_adaptive(dataset, results, chain)
    fitness = mean(scores) if scores else 0.0

    # 7. Structure metrics for observability
    n_steps = len(chain.steps)
    n_tool_steps = len([s for s in chain.steps if s.step_type == "tool"])

    return {
        "fitness": fitness,
        "is_valid": 1,
        "n_steps": n_steps,
        "n_tool_steps": n_tool_steps,
    }
