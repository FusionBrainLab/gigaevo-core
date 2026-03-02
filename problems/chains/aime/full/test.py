"""Test the best evolved chain on the test dataset (AIME full_chain mode)."""

import argparse

import pandas as pd

from problems.chains.utils import get_best_program
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.client import LLMClient
from problems.chains.aime.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    outer_context_builder,
)
from problems.chains.aime.full.config import FULL_CHAIN_CONFIG, load_baseline
from problems.chains.aime.full.validate import extract_answer, calculate_fitness


def load_test_context(year: int = 2025, n_trials: int = 4) -> dict:
    """Load test dataset context."""
    df = pd.read_csv(DATASET_CONFIG["path"])

    df = df[df["Year"] == year].reset_index(drop=True)
    df = pd.concat([df] * n_trials, ignore_index=True)

    return {
        "test_dataset": df.to_dict("records"),
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
        f"System prompt: "
        f"{chain.system_prompt[:80] if chain.system_prompt else '(empty)'}..."
    )

    # Use year=2025 with n_trials=1 for quick check
    context = load_test_context(year=2025, n_trials=1)
    dataset = context["test_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset[:n_samples]
        targets = targets[:n_samples]

    client = LLMClient(**LLM_CONFIG)

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    predictions = [extract_answer(r.final_output) for r in results]

    print(f"\n=== Baseline Results ({len(dataset)} samples) ===")
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"  Sample {i + 1}: pred={pred!r}, target={target!r}")

    accuracy = calculate_fitness(targets, predictions)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Extraction failures: {extraction_failures:.4f}")

    return {"accuracy": accuracy, "extraction_failures": extraction_failures}


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
    print(f"Training fitness: {best['fitness']:.4f}")
    print(f"Code:\n{best['code']}\n")

    exec_globals = {}
    exec(best["code"], exec_globals)
    chain_spec = exec_globals["entrypoint"]()

    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    context = load_test_context(year=2025, n_trials=4)
    dataset = context["test_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset[:n_samples]
        targets = targets[:n_samples]

    client = LLMClient(**LLM_CONFIG)

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    predictions = [extract_answer(r.final_output) for r in results]

    accuracy = calculate_fitness(targets, predictions)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    print("\n=== Test Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Extraction failures: {extraction_failures:.4f}")

    return {"accuracy": accuracy, "extraction_failures": extraction_failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chain on test dataset (AIME full_chain)"
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
