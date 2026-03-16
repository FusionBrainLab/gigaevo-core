from statistics import mean
from typing import List, Union

import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.gsm8k.config import LLM_CONFIG, load_context
from problems.prompts.gsm8k.utils import (
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)
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


def normalize_number(value: str) -> str:
    """Normalize a number string for comparison."""
    try:
        # Remove commas and convert to float, then back to string
        return str(float(value.replace(",", "")))
    except (ValueError, AttributeError):
        return value


def calculate_fitness(
    data: pd.DataFrame,
    preds: List[Union[str, None]],
    target_field: str = "answer",
):
    """Calculate accuracy."""
    accuracy = []
    for pred, target in zip(preds, data[target_field].tolist()):
        target = str(target)
        if pred is None:
            accuracy.append(0)
            continue
        try:
            pred = normalize_number(strip_string(pred))
            target = normalize_number(target)
            accuracy.append(pred == target)
        except:
            accuracy.append(pred == target)

    return mean(accuracy)


def validate(prompt_template: str):
    """Validate prompt template and compute fitness metrics."""
    context = load_context(n_samples=300)

    validate_prompt_template(
        prompt_template,
        required_placeholders=context.get("required_placeholders", []),
        available_placeholders=context.get("available_placeholders", []),
    )

    client = LLMClient(**LLM_CONFIG)

    results = run_prompts(
        prompt_template,
        client,
        context,
        dataset_key="train_dataset",
    )

    raw_responses = results["predictions"]
    predictions = [extract_answer(r) for r in raw_responses]

    dataset = context["train_dataset"]

    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    fitness = calculate_fitness(dataset, predictions, context["target_field"])

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
