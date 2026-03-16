"""Test the best evolved prompt on the test dataset."""

import argparse

import pandas as pd

from problems.prompts.pupa.config import DATASET_CONFIG, load_baseline
from problems.prompts.pupa.utils.pipeline import run_pipeline
from problems.prompts.pupa.validate import calculate_fitness
from problems.prompts.utils import RedisRunConfig, get_best_program


def load_test_context(n_samples: int | None = None) -> dict:
    """Load test dataset context."""
    test_dataset = pd.read_csv(DATASET_CONFIG["test_path"])

    if n_samples is not None and n_samples < len(test_dataset):
        test_dataset = test_dataset.head(n_samples).reset_index(drop=True)

    return {
        "test_dataset": test_dataset,
        "target_field": DATASET_CONFIG["target_field"],
        "pii_field": DATASET_CONFIG["pii_field"],
    }


def test_baseline(n_samples: int = 3):
    """Quick baseline test: validate and run on a few samples."""
    prompt_template = load_baseline()

    print(f"Baseline prompt:\n{prompt_template}\n")

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]

    results = run_pipeline(prompt_template, context, dataset_key="test_dataset")
    metrics = calculate_fitness(results)

    print(f"\n=== Baseline Results ({len(dataset)} samples) ===")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Quality: {metrics['avg_quality']:.4f}")
    print(f"Leakage: {metrics['avg_leakage']:.4f}")

    return metrics


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

    context = load_test_context(n_samples=n_samples)

    results = run_pipeline(prompt_template, context, dataset_key="test_dataset")
    metrics = calculate_fitness(results)

    print("\n=== Test Results ===")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Quality: {metrics['avg_quality']:.4f}")
    print(f"Leakage: {metrics['avg_leakage']:.4f}")

    return metrics


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
