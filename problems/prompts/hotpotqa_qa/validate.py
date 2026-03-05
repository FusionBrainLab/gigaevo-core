"""Validation for HotpotQA multi-hop QA prompt evolution."""

import re
from statistics import mean

import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.utils import validate_prompt_template, run_prompts
from problems.prompts.hotpotqa_qa.config import LLM_CONFIG, load_context
from problems.prompts.hotpotqa_qa.utils import normalize_text


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def calculate_fitness(
    data: pd.DataFrame,
    predictions: list[str | None],
    target_field: str = "answer",
) -> float:
    """Calculate Exact Match (EM) after text normalization."""
    matches = []
    targets = data[target_field].tolist()

    for pred, target in zip(predictions, targets):
        if pred is None:
            matches.append(0)
            continue

        norm_pred = normalize_text(pred)
        norm_target = normalize_text(str(target))
        matches.append(int(norm_pred == norm_target))

    return mean(matches) if matches else 0.0


def validate(prompt_template: str) -> dict:
    """Validate prompt template and compute fitness metrics."""
    context = load_context(n_samples=300)

    validate_prompt_template(
        prompt_template,
        required_placeholders=context.get("required_placeholders", []),
        available_placeholders=context.get("available_placeholders", []),
    )

    client = LLMClient(**LLM_CONFIG)
    results = run_prompts(prompt_template, client, context, dataset_key="train_dataset")

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
