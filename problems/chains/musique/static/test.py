"""Test evolved chains on the MuSiQue test dataset (static mode)."""

import argparse
import random

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.musique.full.validate import calculate_exact_match, extract_answer
from problems.chains.musique.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.musique.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.utils import get_best_program


def load_test_context(n_samples: int | None = None, seed: int = 42) -> dict:
    """Load test dataset context with deterministic passage selection."""
    raw_samples = load_jsonl(DATASET_CONFIG["test_path"])
    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    rng = random.Random(seed)
    processed = [
        preprocess_sample(s, k=DATASET_CONFIG["k_passages"], rng=rng)
        for s in raw_samples
    ]
    return {"test_dataset": processed}


def _evaluate_chain(chain, dataset: list[dict]) -> dict:
    targets = [
        sample.get("answer_aliases", [sample.get("answer", "")]) for sample in dataset
    ]

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
    return {
        "predictions": predictions,
        "targets": targets,
        "exact_match": exact_match,
        "extraction_failures": extraction_failures,
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

    context = load_test_context(n_samples=n_samples)
    dataset = context["test_dataset"]
    metrics = _evaluate_chain(chain, dataset)

    print(f"\n=== Baseline Results ({n_samples} samples) ===")
    for i, (pred, target_aliases) in enumerate(
        zip(metrics["predictions"], metrics["targets"]), start=1
    ):
        print(f"  Sample {i}: pred={pred!r}, targets={target_aliases!r}")

    print(f"\nExact Match: {metrics['exact_match']:.4f}")
    print(f"Extraction failures: {metrics['extraction_failures']:.4f}")
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
    print(f"Training fitness (EM): {best['fitness']:.4f}")

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
    metrics = _evaluate_chain(chain, dataset)

    print("\n=== Test Results ===")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Extraction failures: {metrics['extraction_failures']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chain on MuSiQue test dataset (static)"
    )
    parser.add_argument("--mode", choices=["baseline", "redis"], default="baseline")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-prefix", type=str, default="")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.mode == "baseline":
        test_baseline(n_samples=args.n_samples)
    else:
        test_best_chain(
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            n_samples=args.n_samples,
        )
