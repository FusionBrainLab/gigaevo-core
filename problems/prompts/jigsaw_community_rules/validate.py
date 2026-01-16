import re
from typing import List, Union
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from problems.prompts.client import LLMClient
from problems.prompts.utils import validate_prompt_template, run_prompts
from problems.prompts.jigsaw_community_rules.config import LLM_CONFIG, load_context


def extract_answer(response: str) -> float | None:
    """Extract answer from LLM response.

    Args:
        response: Raw LLM response string

    Returns:
        Extracted answer float, or None if extraction failed
    """
    # Match "Answer:" followed by optional whitespace and a number
    match = re.search(r'Answer:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
    if match:
        try:
            value = float(match.group(1))
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, value))
        except ValueError:
            return None
    return None


def calculate_fitness(
    data: pd.DataFrame,
    preds: List[Union[float, int, None]],
    target_field: str = "rule_violation",
):
    rule_metrics = []
    for _, group in data.groupby("rule"):
        if len(group[target_field].unique()) > 1:
            group_preds = [
                preds[i] if preds[i] is not None else 0.5 for i in group.index
            ]
            metric = roc_auc_score(group[target_field], group_preds)
            rule_metrics.append(metric)
    return np.mean(rule_metrics) if rule_metrics else None


def validate(prompt_template: str):
    # 1. Load dataset context
    context = load_context()

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

    call_logs = results["call_logs"]
    dataset = context["train_dataset"]

    # 6. Extraction failures: predictions that failed to parse (None)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    # 7. Cost utilization: average across samples
    avg_cost_utilization = (
        mean(log.cost_utilization for log in call_logs) if call_logs else 0.0
    )

    # 8. Main objective: average ROC-AUC on the dataset (higher is better)
    fitness = calculate_fitness(dataset, predictions, context["target_field"])

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "avg_cost_utilization": avg_cost_utilization,
        "is_valid": 1,
    }
