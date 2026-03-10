"""Test the best evolved retrieval chain on MuSiQue test dataset (full mode)."""

import argparse

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.musique_retrieval.full.config import (
    FULL_CHAIN_CONFIG,
    load_baseline,
)
from problems.chains.musique_retrieval.full.validate import (
    calculate_exact_match,
    extract_answer,
)
from problems.chains.musique_retrieval.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    TASK_BM25S_INDEX_DIR,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.musique_retrieval.utils.retrieval import make_retrieve_fn
from problems.chains.utils import get_best_program


def load_test_context(n_samples: int | None = None) -> dict:
    """Load test dataset context."""
    raw_samples = load_jsonl(DATASET_CONFIG["test_path"])
    if n_samples is not None and n_samples < len(raw_samples):
        raw_samples = raw_samples[:n_samples]

    processed = [preprocess_sample(s, sample_idx=i) for i, s in enumerate(raw_samples)]
    return {
        "test_dataset": processed,
        "task_index_dir": TASK_BM25S_INDEX_DIR,
    }


def _evaluate_chain(chain, context: dict) -> dict:
    dataset = context["test_dataset"]
    targets = [
        sample.get("answer_aliases", [sample.get("answer", "")])
        for sample in dataset
    ]
    passages_by_task = {sample["task_id"]: sample["passages"] for sample in dataset}

    client = LLMClient(**LLM_CONFIG)
    tool_registry = {
        "retrieve": make_retrieve_fn(
            context["task_index_dir"],
            passages_by_task=passages_by_task,
            k=7,
        )
    }
    results = run_chain_on_dataset(
        chain, client, dataset, outer_context_builder, tool_registry
    )
    predictions = [extract_answer(r.final_output) for r in results]

    exact_match = calculate_exact_match(targets, predictions)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )
    return {
        "exact_match": exact_match,
        "extraction_failures": extraction_failures,
        "predictions": predictions,
        "targets": targets,
    }


def test_baseline(n_samples: int = 3):
    """Quick baseline test: validate and run on a few samples."""
    baseline = load_baseline()
    chain = validate_chain_spec(
        baseline,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    context = load_test_context(n_samples=n_samples)
    metrics = _evaluate_chain(chain, context)

    print(f"\n=== Baseline Results ({n_samples} samples) ===")
    for i, (pred, targets) in enumerate(
        zip(metrics["predictions"], metrics["targets"]), start=1
    ):
        print(f"  Sample {i}: pred={pred!r}, targets={targets!r}")
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
    metrics = _evaluate_chain(chain, context)

    print("\n=== Test Results ===")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Extraction failures: {metrics['extraction_failures']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chain on MuSiQue test dataset (full_chain retrieval)"
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
