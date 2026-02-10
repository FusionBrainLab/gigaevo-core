import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.utils import run_prompts, RedisRunConfig, get_best_program
from problems.prompts.gsm8k.config import LLM_CONFIG, DATASET_CONFIG
from problems.prompts.gsm8k.validate import extract_answer, calculate_fitness


def load_test_context(n_samples: int = 300) -> dict:
    """Load test dataset context."""
    test_dataset = pd.read_csv(DATASET_CONFIG["test_path"])

    test_dataset = test_dataset[:n_samples]

    return {
        "test_dataset": test_dataset,
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
    print(f"Training fitness: {best['fitness']}")
    print(f"Code:\n{best['code']}\n")

    exec_globals = {}
    exec(best["code"], exec_globals)
    prompt_template = exec_globals["entrypoint"]()

    context = load_test_context()

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

    return {
        "accuracy": accuracy,
        "extraction_failures": extraction_failures,
    }


if __name__ == "__main__":
    import argparse

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
