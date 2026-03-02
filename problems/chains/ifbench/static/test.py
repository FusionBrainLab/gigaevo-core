"""Test the best evolved chain on the test dataset (static mode)."""

import argparse
from statistics import mean

from problems.chains.utils import get_best_program
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.client import LLMClient
from problems.chains.ifbench.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    load_jsonl,
    preprocess_sample,
    outer_context_builder,
)
from problems.chains.ifbench.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.ifbench.utils.evaluation import test_instruction_following


def load_test_context(n_samples: int | None = None) -> dict:
    """Load test dataset context."""
    raw_samples = load_jsonl(DATASET_CONFIG["test_path"])

    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    processed = [preprocess_sample(s) for s in raw_samples]

    return {
        "test_dataset": processed,
    }


def evaluate_results(dataset: list[dict], results) -> dict:
    """Evaluate chain results against constraints."""
    scores = []
    for sample, result in zip(dataset, results):
        response = result.final_output
        if response and response.strip():
            score = test_instruction_following(sample, response)
        else:
            score = 0.0
        scores.append(score)

    fitness = mean(scores) if scores else 0.0
    return {"fitness": fitness}


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
    print(f"System prompt: {chain.system_prompt[:80] if chain.system_prompt else '(empty)'}...")

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]

    client = LLMClient(**LLM_CONFIG)

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    print(f"\n=== Baseline Results ({n_samples} samples) ===")
    for i, result in enumerate(results):
        response = result.final_output
        preview = response[:100] if response else "(empty)"
        print(f"  Sample {i+1}: {preview!r}...")

    metrics = evaluate_results(dataset, results)
    print(f"\nConstraint satisfaction: {metrics['fitness']:.4f}")

    return metrics


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

    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]

    client = LLMClient(**LLM_CONFIG)

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry=None
    )

    metrics = evaluate_results(dataset, results)

    print("\n=== Test Results ===")
    print(f"Constraint satisfaction: {metrics['fitness']:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test chain on test dataset (static)")
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
