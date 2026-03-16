from statistics import mean

from problems.prompts.pupa.config import load_context
from problems.prompts.pupa.utils.pipeline import run_pipeline
from problems.prompts.utils import validate_prompt_template


def calculate_fitness(results: list[dict]) -> dict:
    """Calculate fitness and aggregate metrics from results."""
    if not results:
        return {"fitness": 0.0, "avg_quality": 0.0, "avg_leakage": 1.0}

    qualities = [r["quality"] for r in results]
    leakages = [r["leakage"] for r in results]

    avg_quality = mean(qualities)
    avg_leakage = mean(leakages)

    # PUPA fitness: (quality + (1 - leakage)) / 2
    fitness = (avg_quality + (1 - avg_leakage)) / 2

    return {
        "fitness": fitness,
        "avg_quality": avg_quality,
        "avg_leakage": avg_leakage,
    }


def validate(prompt_template: str) -> dict:
    """Validate prompt template and compute fitness metrics.

    Args:
        prompt_template: The evolved query rewriter prompt template

    Returns:
        dict: Metrics including fitness, avg_quality, avg_leakage, is_valid
    """
    context = load_context(n_samples=150)

    validate_prompt_template(
        prompt_template,
        required_placeholders=context.get("required_placeholders", []),
        available_placeholders=context.get("available_placeholders", []),
    )

    results = run_pipeline(prompt_template, context, dataset_key="train_dataset")
    metrics = calculate_fitness(results)

    return {
        "fitness": metrics["fitness"],
        "avg_quality": metrics["avg_quality"],
        "avg_leakage": metrics["avg_leakage"],
        "is_valid": 1,
    }
