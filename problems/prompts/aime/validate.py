from statistics import mean

import pandas as pd

from problems.prompts.aime.config import LLM_CONFIG, load_context
from problems.prompts.aime.utils import (
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)
from problems.prompts.client import LLMClient
from problems.prompts.utils import run_prompts, validate_prompt_template


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response.

    Args:
        response: Raw LLM response string

    Returns:
        Extracted answer str, or None if extraction failed
    """
    answer = remove_boxed(last_boxed_only_string(response))
    if answer is not None:
        answer = answer.lstrip("0")
        if answer == "":
            answer = None
    return answer


def calculate_fitness(
    data: pd.DataFrame,
    preds: list[str | None],
    target_field: str = "answer",
):
    """Calculate Accuracy."""
    accuracy = []
    for pred, target in zip(preds, data[target_field].tolist()):
        target = str(target)
        if pred is None:
            accuracy.append(0)
            continue
        try:
            pred = strip_string(pred)
            target = strip_string(target)
            accuracy.append(pred == target)
        except:
            accuracy.append(pred == target)

    return mean(accuracy)


def validate(prompt_template: str):
    """Validate prompt template and compute fitness metrics.

    Args:
        prompt_template: The evolved prompt template with {field} placeholders

    Returns:
        dict: Metrics including fitness, avg_extraction_failures, avg_cost_utilization, is_valid
    """
    # 1. Load dataset context
    context = load_context(years=(2023, 2024), n_trials=3)

    # 2. Validate template structure
    validate_prompt_template(
        prompt_template,
        required_placeholders=context.get("required_placeholders", []),
        available_placeholders=context.get("available_placeholders", []),
    )

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Run on dataset
    results = run_prompts(
        prompt_template,
        client,
        context,
        dataset_key="train_dataset",
    )

    # 5. Extract predictions from raw responses
    raw_responses = results["predictions"]  # Raw LLM response strings
    predictions = [extract_answer(r) for r in raw_responses]

    dataset = context["train_dataset"]

    # 6. Extraction failures: predictions that failed to parse (None)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    # 7. Main objective
    fitness = calculate_fitness(dataset, predictions, context["target_field"])

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
