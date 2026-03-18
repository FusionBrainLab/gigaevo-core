"""Test the best evolved chain on the test dataset (PAPILLON static mode)."""

import argparse
import asyncio
from statistics import mean

import pandas as pd

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.papillon.shared_config import (
    DATASET_CONFIG,
    JUDGE_CONFIG,
    LLM_CONFIG,
    outer_context_builder,
)
from problems.chains.papillon.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.papillon.utils.external_llm import make_external_llm_fn
from problems.chains.papillon.utils.pipeline import judge_leakage, judge_quality
from problems.chains.utils import get_best_program


def load_test_context(n_samples: int | None = None) -> dict:
    """Load test dataset context."""
    test_df = pd.read_csv(DATASET_CONFIG["test_path"])

    if n_samples is not None and n_samples < len(test_df):
        test_df = test_df.head(n_samples).reset_index(drop=True)

    return {
        "test_dataset": test_df.to_dict("records"),
        "target_field": DATASET_CONFIG["target_field"],
        "pii_field": DATASET_CONFIG["pii_field"],
    }


def _evaluate(dataset, results, context):
    """Run quality and leakage judges on chain results."""

    async def judge_all():
        judge_client = LLMClient(**JUDGE_CONFIG)
        sem = asyncio.Semaphore(32)

        async def judge_one(sample, result):
            async with sem:
                quality = await judge_quality(
                    judge_client.copy(),
                    sample["user_query"],
                    result.final_output,
                    sample[context["target_field"]],
                )
                leakage = await judge_leakage(
                    judge_client.copy(),
                    result.step_outputs[0],
                    sample.get(context["pii_field"], ""),
                )
                return quality, leakage

        return await asyncio.gather(
            *(judge_one(s, r) for s, r in zip(dataset, results))
        )

    judge_results = asyncio.run(judge_all())

    avg_quality = mean(q for q, _ in judge_results)
    avg_leakage = mean(l for _, l in judge_results)
    fitness = (avg_quality + (1 - avg_leakage)) / 2

    return {
        "fitness": fitness,
        "avg_quality": avg_quality,
        "avg_leakage": avg_leakage,
    }


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
    print(f"System prompt: {chain.system_prompt[:80]}...")

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]

    client = LLMClient(**LLM_CONFIG)
    tool_registry = {"external_llm": make_external_llm_fn(LLM_CONFIG)}

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )

    # Show per-sample outputs
    for i, result in enumerate(results):
        print(f"\n--- Sample {i + 1} ---")
        print(f"  Original query: {dataset[i]['user_query'][:100]}...")
        print(f"  Sanitized query: {result.step_outputs[0][:100]}...")
        print(f"  External response: {result.step_outputs[1][:100]}...")
        print(f"  Final response: {result.final_output[:100]}...")

    metrics = _evaluate(dataset, results, context)

    print(f"\n=== Baseline Results ({n_samples} samples) ===")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Quality: {metrics['avg_quality']:.4f}")
    print(f"Leakage: {metrics['avg_leakage']:.4f}")

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
    tool_registry = {"external_llm": make_external_llm_fn(LLM_CONFIG)}

    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )

    metrics = _evaluate(dataset, results, context)

    print("\n=== Test Results ===")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Quality: {metrics['avg_quality']:.4f}")
    print(f"Leakage: {metrics['avg_leakage']:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chain on test dataset (PAPILLON static)"
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
