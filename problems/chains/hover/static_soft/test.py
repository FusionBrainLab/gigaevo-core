"""Test the best evolved chain on the test dataset (static mode)."""

import argparse
from statistics import mean, stdev

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hover.shared_config import (
    BM25S_INDEX_DIR,
    CORPUS_PATH,
    DATASET_CONFIG,
    get_llm_config,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.hover.static_soft.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hover.utils.retrieval import make_retrieve_fn
from problems.chains.hover.utils.utils import (
    discrete_retrieval_eval,
    extract_titles_from_passages,
)
from problems.chains.utils import get_best_program


def load_test_context(n_samples: int | None = None) -> dict:
    """Load test dataset context."""
    raw_samples = load_jsonl(DATASET_CONFIG["test_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    processed = [preprocess_sample(s) for s in raw_samples]

    return {
        "test_dataset": processed,
        "bm25s_index_dir": BM25S_INDEX_DIR,
        "corpus_path": CORPUS_PATH,
    }


def evaluate_retrieval_coverage(dataset, results):
    """Compute retrieval coverage from chain results."""
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
    return mean(scores) if scores else 0.0


def test_baseline(n_samples: int = 3):
    """Quick baseline test: validate and run on a few samples."""
    baseline = load_baseline()
    chain = validate_chain_spec(
        baseline,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    print(f"Baseline validated: {len(chain.steps)} steps")
    print(
        f"System prompt: {chain.system_prompt[:80] if chain.system_prompt else '(empty)'}..."
    )

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]

    client = LLMClient(**get_llm_config())
    tool_registry = {
        "retrieve": make_retrieve_fn(
            context["bm25s_index_dir"], k=7, corpus_path=context["corpus_path"]
        ),
        "retrieve_deep": make_retrieve_fn(
            context["bm25s_index_dir"], k=10, corpus_path=context["corpus_path"]
        ),
    }

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )

    coverage = evaluate_retrieval_coverage(dataset, results)

    print(f"\n=== Baseline Results ({n_samples} samples) ===")
    for i, (sample, result) in enumerate(zip(dataset, results)):
        found = extract_titles_from_passages(
            "\n".join(
                [
                    result.step_outputs[0],
                    result.step_outputs[3],
                    result.step_outputs[6],
                ]
            )
        )
        gold = set(sample["supporting_facts"])
        hit = discrete_retrieval_eval(gold, found)
        print(
            f"  Sample {i + 1}: coverage={hit}, gold={gold}, found_count={len(found)}"
        )

    print(f"\nRetrieval Coverage: {coverage:.4f}")

    return {"retrieval_coverage": coverage}


def test_best_chain(
    redis_db: int,
    redis_prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    n_samples: int | None = None,
    n_repeats: int = 1,
):
    """Extract best chain and evaluate on test dataset.

    When n_repeats > 1, runs the full evaluation multiple times and
    reports mean, std, and per-repeat scores.
    """
    from tools.utils import RedisRunConfig

    config = RedisRunConfig(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_prefix=redis_prefix,
    )
    best = get_best_program(config, fitness_col="metric_fitness", minimize=False)

    if best is None:
        print("No programs found in Redis")
        return

    print(f"Best program ID: {best['id']}")
    print(f"Training fitness (coverage): {best['fitness']:.4f}")
    print(f"Code:\n{best['code']}\n")

    exec_globals = {}
    exec(best["code"], exec_globals)
    chain_spec = exec_globals["entrypoint"]()

    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]

    client = LLMClient(**get_llm_config())
    tool_registry = {
        "retrieve": make_retrieve_fn(
            context["bm25s_index_dir"], k=7, corpus_path=context["corpus_path"]
        ),
        "retrieve_deep": make_retrieve_fn(
            context["bm25s_index_dir"], k=10, corpus_path=context["corpus_path"]
        ),
    }

    coverages = []
    for rep in range(1, n_repeats + 1):
        results = run_chain_on_dataset(
            chain, client, dataset, outer_context_builder, tool_registry
        )
        coverage = evaluate_retrieval_coverage(dataset, results)
        coverages.append(coverage)
        print(f"  Repeat {rep}/{n_repeats}: coverage = {coverage:.4f}")

    print("\n=== Test Results ===")
    mean_cov = mean(coverages)
    print(f"Retrieval Coverage (mean): {mean_cov:.4f}")
    if n_repeats > 1:
        std_cov = stdev(coverages)
        print(f"Retrieval Coverage (std):  {std_cov:.4f}")
        print(f"Retrieval Coverage (all):  {[f'{c:.4f}' for c in coverages]}")
        print(f"n_repeats: {n_repeats}")

    return {"retrieval_coverage_mean": mean_cov, "retrieval_coverage_all": coverages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test HoVer chain on test dataset (static)"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "redis"],
        default="baseline",
        help="Test mode: 'baseline' runs baseline on a few samples, "
        "'redis' tests best evolved chain from Redis",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of test samples (default: 3 for baseline, all for redis)",
    )
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-prefix", type=str, default="")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of times to repeat the full test evaluation (default: 1)",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        test_baseline(n_samples=args.n_samples or 3)
    elif args.mode == "redis":
        test_best_chain(
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            n_samples=args.n_samples,
            n_repeats=args.n_repeats,
        )
