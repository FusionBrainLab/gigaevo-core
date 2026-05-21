"""Validate nlp/aime chain and compute accuracy fitness on math olympiad problems."""

from statistics import mean

from problems.chains.aime.utils.utils import (
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)
from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.nlp.aime.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.nlp.aime.static.config import FULL_CHAIN_CONFIG
from problems.chains.runner_config import RunnerConfig


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response using \\boxed{...} pattern."""
    answer = remove_boxed(last_boxed_only_string(response))
    if answer is not None:
        answer = answer.lstrip("0")
        if answer == "":
            answer = None
    return answer


def calculate_fitness(
    targets: list,
    predictions: list[str | None],
) -> float:
    """Calculate accuracy after answer normalization."""
    accuracy = []
    for pred, target in zip(predictions, targets):
        target = str(target)
        if pred is None:
            accuracy.append(0)
            continue
        try:
            pred = strip_string(pred)
            target = strip_string(target)
            accuracy.append(pred == target)
        except Exception:
            accuracy.append(pred == target)
    return mean(accuracy) if accuracy else 0.0


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute accuracy on AIME problems.

    Returns:
        Dict with fitness (accuracy), avg_extraction_failures, and is_valid.
    """
    # 1. Structural validation (full_chain mode)
    chain = validate_chain_spec(
        chain_spec,
        mode="full_chain",
        full_chain_config=FULL_CHAIN_CONFIG,
    )

    # 2. Load context (2023-2024 problems, 3 trials each)
    context = load_context(years=(2023, 2024), n_trials=3)
    dataset = context["train_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    # 3. Execute chain (no tools — all LLM steps)
    client = LLMClient(**LLM_CONFIG)
    results = run_chain_on_dataset(
        chain,
        client,
        dataset,
        outer_context_builder,
        tool_registry=None,
        runner_config=RunnerConfig.from_env(),
    )

    # 4. Extract answers from final step outputs
    predictions = [extract_answer(r.final_output) for r in results]

    # 5. Compute metrics
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    fitness = calculate_fitness(targets, predictions)

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
