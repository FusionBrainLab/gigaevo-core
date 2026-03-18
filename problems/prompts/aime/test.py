"""Test the best evolved prompt on the test dataset."""

import argparse

import pandas as pd

from problems.prompts.aime.config import DATASET_CONFIG, LLM_CONFIG, load_baseline
from problems.prompts.aime.validate import calculate_fitness, extract_answer
from problems.prompts.client import LLMClient
from problems.prompts.utils import RedisRunConfig, get_best_program, run_prompts


def load_test_context(year: int = 2025, n_trials: int = 4) -> dict:
    """Load test dataset context."""
    test_dataset = pd.read_csv(DATASET_CONFIG["path"])

    test_dataset = test_dataset[test_dataset["Year"] == year].reset_index(drop=True)
    test_dataset = pd.concat([test_dataset] * n_trials, ignore_index=True)

    return {
        "test_dataset": test_dataset,
        "target_field": DATASET_CONFIG["target_field"],
    }


def test_baseline(n_samples: int = 3):
    """Quick baseline test: validate and run on a few samples."""
    prompt_template = load_baseline()

    print(f"Baseline prompt:\n{prompt_template}\n")

    # For baseline, use year=2025 with n_trials=1 for quick check
    context = load_test_context(year=2025, n_trials=1)
    dataset = context["test_dataset"]

    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.head(n_samples).reset_index(drop=True)
        context["test_dataset"] = dataset

    client = LLMClient(**LLM_CONFIG)
    results = run_prompts(prompt_template, client, context, dataset_key="test_dataset")

    raw_responses = results["predictions"]
    predictions = [extract_answer(r) for r in raw_responses]
    targets = dataset[context["target_field"]].tolist()

    print(f"\n=== Baseline Results ({len(dataset)} samples) ===")
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"  Sample {i + 1}: pred={pred!r}, target={target!r}")

    accuracy = calculate_fitness(dataset, predictions, context["target_field"])
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Extraction failures: {extraction_failures:.4f}")

    return {"accuracy": accuracy, "extraction_failures": extraction_failures}


def test_best_prompt(
    redis_db: int,
    redis_prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    n_samples: int | None = None,
):
    """Extract best prompt and evaluate on test dataset."""
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
    print(f"Training fitness: {best['fitness']}")
    print(f"Code:\n{best['code']}\n")

    exec_globals = {}
    exec(best["code"], exec_globals)
    prompt_template = exec_globals["entrypoint"]()

    context = load_test_context(year=2025, n_trials=4)

    client = LLMClient(**LLM_CONFIG)
    results = run_prompts(prompt_template, client, context, dataset_key="test_dataset")

    raw_responses = results["predictions"]
    predictions = [extract_answer(r) for r in raw_responses]

    dataset = context["test_dataset"]
    target_field = context["target_field"]

    accuracy = calculate_fitness(dataset, predictions, target_field)

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
    parser = argparse.ArgumentParser(description="Test prompt on test dataset")
    parser.add_argument(
        "--mode",
        choices=["baseline", "redis"],
        default="baseline",
        help="Test mode: 'baseline' runs baseline on a few samples, "
        "'redis' tests best evolved prompt from Redis",
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
        test_best_prompt(
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            n_samples=args.n_samples,
        )
