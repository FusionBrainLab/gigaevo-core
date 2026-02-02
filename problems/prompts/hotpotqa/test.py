"""Test the best evolved prompt on the test dataset."""

import argparse
import random

import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.utils import run_prompts, RedisRunConfig, get_best_program
from problems.prompts.hotpotqa.config import (
    LLM_CONFIG,
    DATASET_CONFIG,
    load_jsonl,
    preprocess_sample,
)
from problems.prompts.hotpotqa.validate import extract_answer, calculate_fitness


def load_test_context(n_samples: int | None = None, seed: int = 42) -> dict:
    """Load test dataset context."""
    raw_samples = load_jsonl(DATASET_CONFIG["test_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    rng = random.Random(seed)

    processed = [
        preprocess_sample(s, k=DATASET_CONFIG["k_passages"], rng=rng)
        for s in raw_samples
    ]

    return {
        "test_dataset": pd.DataFrame(processed),
        "target_field": DATASET_CONFIG["target_field"],
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
    print(f"Training fitness (EM): {best['fitness']:.4f}")
    print(f"Code:\n{best['code']}\n")

    exec_globals = {}
    exec(best["code"], exec_globals)
    prompt_template = exec_globals["entrypoint"]()
    prompt_template = """
Answer the following question based on the provided passages.

Question: {question}

Passages:
{passages}

Provide your answer in the exact format: Answer: <your answer>

Answer: <answer>
""".strip()

    context = load_test_context()

    client = LLMClient(**LLM_CONFIG)
    results = run_prompts(prompt_template, client, context, dataset_key="test_dataset")

    raw_responses = results["predictions"]
    predictions = [extract_answer(r) for r in raw_responses]

    exact_match = calculate_fitness(
        context["test_dataset"], predictions, context["target_field"]
    )

    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    print("\n=== Test Results ===")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Extraction failures: {extraction_failures:.4f}")

    return {"exact_match": exact_match}


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
