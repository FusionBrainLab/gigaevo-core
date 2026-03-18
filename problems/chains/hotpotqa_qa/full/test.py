"""Test the best evolved chain on the test dataset (HotpotQA-QA full_chain mode)."""

import argparse
import random

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa_qa.full.config import FULL_CHAIN_CONFIG, load_baseline
from problems.chains.hotpotqa_qa.full.validate import (
    calculate_exact_match,
    extract_answer,
)
from problems.chains.hotpotqa_qa.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.utils import get_best_program


def load_test_context(n_samples: int | None = None, seed: int = 42) -> dict:
    """Load test dataset context with passage selection."""
    raw_samples = load_jsonl(DATASET_CONFIG["test_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    rng = random.Random(seed)

    processed = [
        preprocess_sample(s, k=DATASET_CONFIG["k_passages"], rng=rng)
        for s in raw_samples
    ]

    return {
        "test_dataset": processed,
        "target_field": DATASET_CONFIG["target_field"],
    }


def test_baseline(n_samples: int = 3):
    """Quick baseline test: validate and run on a few samples."""
    baseline = load_baseline()
    chain = validate_chain_spec(
        baseline,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    print(f"Baseline validated: {len(chain.steps)} steps")
    print(
        f"System prompt: {chain.system_prompt[:80] if chain.system_prompt else '(empty)'}..."
    )

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    client = LLMClient(**LLM_CONFIG)

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    predictions = [extract_answer(r.final_output) for r in results]

    print(f"\n=== Baseline Results ({n_samples} samples) ===")
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"  Sample {i + 1}: pred={pred!r}, target={target!r}")

    exact_match = calculate_exact_match(targets, predictions)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    print(f"\nExact Match: {exact_match:.4f}")
    print(f"Extraction failures: {extraction_failures:.4f}")

    return {"exact_match": exact_match, "extraction_failures": extraction_failures}


def test_best_chain(
    redis_db: int,
    redis_prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    n_samples: int | None = None,
):
    """Extract best chain and evaluate on test dataset."""
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
    print(f"Training fitness (EM): {best['fitness']:.4f}")
    print(f"Code:\n{best['code']}\n")

    exec_globals = {}
    exec(best["code"], exec_globals)
    chain_spec = exec_globals["entrypoint"]()

    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    client = LLMClient(**LLM_CONFIG)

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    predictions = [extract_answer(r.final_output) for r in results]

    exact_match = calculate_exact_match(targets, predictions)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    print("\n=== Test Results ===")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Extraction failures: {extraction_failures:.4f}")

    return {"exact_match": exact_match, "extraction_failures": extraction_failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chain on test dataset (HotpotQA-QA full_chain)"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "redis"],
        default="baseline",
        help="Test mode: 'baseline' runs baseline on a few samples, "
        "'redis' tests best evolved chain from Redis",
    )
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-prefix", type=str, default="")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.mode == "baseline":
        test_baseline(n_samples=args.n_samples)
    elif args.mode == "redis":
        test_best_chain(
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            n_samples=args.n_samples,
        )
