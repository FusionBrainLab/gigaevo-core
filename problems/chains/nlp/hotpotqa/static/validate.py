"""Validate nlp/hotpotqa chain and compute exact-match fitness."""

import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset_stepwise
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.utils.retrieval import make_batch_tool_fn
from problems.chains.hotpotqa.utils.utils import normalize_text
from problems.chains.nlp.hotpotqa.shared_config import (
    DATASET_CONFIG,
    LLM_CONFIG,
    build_retriever,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.nlp.hotpotqa.static.config import (
    STATIC_CHAIN_TOPOLOGY,
    load_baseline,
)
from problems.chains.runner_config import RunnerConfig


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from vLLM thinking-mode output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern."""
    cleaned = strip_thinking(response)
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def calculate_exact_match(
    targets: list[str],
    predictions: list[str | None],
) -> float:
    """Calculate Exact Match (EM) after text normalization."""
    matches = []
    for pred, target in zip(predictions, targets):
        if pred is None:
            matches.append(0)
            continue
        norm_pred = normalize_text(pred)
        norm_target = normalize_text(str(target))
        matches.append(int(norm_pred == norm_target))
    return mean(matches) if matches else 0.0


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute exact-match fitness on HotpotQA.

    Returns:
        Dict with fitness (EM score), avg_extraction_failures, and is_valid.
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load first 300 samples
    raw_300 = load_jsonl(DATASET_CONFIG["train_path"])[:300]
    dataset = [preprocess_sample(s) for s in raw_300]
    targets = [s[DATASET_CONFIG["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build batch tool registry
    retriever = build_retriever(k=7)
    batch_tool_registry = {"retrieve": make_batch_tool_fn(retriever)}

    # 5. Run chain step-batched (optimal vLLM batching)
    step_max_tokens = {
        2: 8192,
        3: 8192,
        5: 8192,
        6: 8192,
    }
    results = run_chain_on_dataset_stepwise(
        chain,
        client,
        dataset,
        outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
        runner_config=RunnerConfig.from_env(),
    )

    # 6. Extract answers and compute metrics
    predictions = [extract_answer(r.final_output) for r in results]

    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    fitness = calculate_exact_match(targets, predictions)

    return {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }
