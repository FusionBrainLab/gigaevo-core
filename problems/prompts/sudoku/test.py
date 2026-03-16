"""Test the baseline or best evolved Sudoku system prompt."""

from __future__ import annotations

import argparse
import pprint
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from problems.prompts.sudoku.config import load_baseline
from problems.prompts.sudoku.runtime import evaluate_prompt
from problems.prompts.utils import RedisRunConfig, get_best_program


def _load_prompt_from_program(program_code: str) -> str:
    exec_globals: dict[str, object] = {}
    exec(program_code, exec_globals)
    return exec_globals["entrypoint"]()


def test_baseline(
    *,
    split: str,
    n_samples: int | None,
    max_steps: int | None,
) -> dict[str, float]:
    prompt_template = load_baseline()
    metrics, artifact = evaluate_prompt(
        prompt_template,
        split=split,
        max_examples=n_samples,
        max_steps=max_steps,
    )

    print("=== Baseline Sudoku Prompt ===")
    print(prompt_template)
    print("\n=== Metrics ===")
    pprint.pp(metrics)

    failures = [
        item
        for item in artifact["results"]
        if not item["success"]
    ]
    if failures:
        print("\n=== Example Failures ===")
        for failure in failures[: min(3, len(failures))]:
            print(
                f"example_id={failure['example_id']} "
                f"reason={failure['failure_reason']} "
                f"raw_last_output={failure['raw_last_output']!r}"
            )

    return metrics


def test_best_prompt(
    *,
    redis_db: int,
    redis_prefix: str,
    redis_host: str,
    redis_port: int,
    split: str,
    n_samples: int | None,
    max_steps: int | None,
) -> dict[str, float] | None:
    config = RedisRunConfig(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_prefix=redis_prefix,
    )

    best = get_best_program(config, fitness_col="metric_fitness", minimize=False)
    if best is None:
        print("No programs found in Redis")
        return None

    prompt_template = _load_prompt_from_program(best["code"])
    metrics, artifact = evaluate_prompt(
        prompt_template,
        split=split,
        max_examples=n_samples,
        max_steps=max_steps,
    )

    print(f"Best program ID: {best['id']}")
    print(f"Training fitness: {best['fitness']}")
    print("\n=== Evolved Sudoku Prompt ===")
    print(prompt_template)
    print("\n=== Metrics ===")
    pprint.pp(metrics)

    failures = [
        item
        for item in artifact["results"]
        if not item["success"]
    ]
    if failures:
        print("\n=== Example Failures ===")
        for failure in failures[: min(3, len(failures))]:
            print(
                f"example_id={failure['example_id']} "
                f"reason={failure['failure_reason']} "
                f"raw_last_output={failure['raw_last_output']!r}"
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sudoku prompt on local vLLM")
    parser.add_argument("--mode", choices=["baseline", "redis"], default="baseline")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-prefix", type=str, default="")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.mode == "baseline":
        test_baseline(
            split=args.split,
            n_samples=args.n_samples,
            max_steps=args.max_steps,
        )
    else:
        test_best_prompt(
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            split=args.split,
            n_samples=args.n_samples,
            max_steps=args.max_steps,
        )
