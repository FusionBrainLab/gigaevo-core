import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.utils import run_prompts, RedisRunConfig, get_best_program
from problems.prompts.ifbench.config import LLM_CONFIG, DATASET_CONFIG
from problems.prompts.ifbench.validate import calculate_fitness


def load_test_context() -> dict:
    """Load test dataset context."""
    test_dataset = pd.read_json(DATASET_CONFIG["test_path"], lines=True)

    return {
        "test_dataset": test_dataset,
    }


def test_best_prompt(
    redis_db: int,
    redis_prefix: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
):
    """Extract best prompt and evaluate on test dataset."""
    # 1. Get best program from Redis
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

    # 2. Extract prompt template by executing the code
    exec_globals = {}
    exec(best["code"], exec_globals)
    prompt_template = exec_globals["entrypoint"]()

    # 3. Load test context
    context = load_test_context()

    # 4. Create LLM client and run
    client = LLMClient(**LLM_CONFIG)
    results = run_prompts(prompt_template, client, context, dataset_key="test_dataset")

    # 5. Extract predictions and compute fitness
    raw_responses = results["predictions"]
    dataset = context["test_dataset"]

    fitness = calculate_fitness(dataset, raw_responses)

    print("\n=== Test Results ===")
    print(f"Constraint Satisfaction Rate: {fitness:.4f}")

    return {
        "fitness": fitness,
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
