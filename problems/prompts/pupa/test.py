"""Test the best evolved prompt on the test dataset."""

import argparse

import pandas as pd

from problems.prompts.utils import RedisRunConfig, get_best_program
from problems.prompts.pupa.config import DATASET_CONFIG
from problems.prompts.pupa.utils.pipeline import run_pipeline
from problems.prompts.pupa.validate import calculate_fitness


def load_test_context() -> dict:
    """Load test dataset context."""
    test_dataset = pd.read_csv(DATASET_CONFIG["test_path"])

    return {
        "test_dataset": test_dataset,
        "target_field": DATASET_CONFIG["target_field"],
        "pii_field": DATASET_CONFIG["pii_field"],
    }


def test_best_prompt(
    redis_db: int,
    redis_prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
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

    context = load_test_context()

    results = run_pipeline(prompt_template, context, dataset_key="test_dataset")
    metrics = calculate_fitness(results)

    print("\n=== Test Results ===")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Quality: {metrics['avg_quality']:.4f}")
    print(f"Leakage: {metrics['avg_leakage']:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test best prompt on test dataset")
    parser.add_argument("--redis-db", type=int, required=True)
    parser.add_argument("--redis-prefix", type=str, required=True)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    test_best_prompt(
        redis_db=args.redis_db,
        redis_prefix=args.redis_prefix,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )
