import re
from statistics import mean

from problems.chains.chain_validation import validate_chain_spec
from problems.chains.chain_runner import run_chain_on_dataset_stepwise
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.shared_config import (
    LLM_CONFIG,
    load_context,
    outer_context_builder,
)
from problems.chains.hotpotqa.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hotpotqa.utils.retrieval import batch_retrieve
from problems.chains.hotpotqa.utils.utils import normalize_text


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def calculate_exact_match(
    targets: list[str],
    predictions: list[str | None],
) -> float:
    """Calculate Exact Match (EM) after text normalization.

    Args:
        targets: List of gold answer strings
        predictions: List of predicted answer strings (None for extraction failures)

    Returns:
        EM score as a float in [0, 1]
    """
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
    """Validate chain specification and compute fitness metrics.

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps

    Returns:
        Dict with fitness, avg_extraction_failures, is_valid
    """
    # 1. Structural validation (catch ValueError → return sentinels)
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load context (dataset + retrieval paths)
    context = load_context(n_samples=300)
    dataset = context["train_dataset"]
    targets = [s[context["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build batch tool registry for step-batched execution
    bm25_dir = context["bm25s_index_dir"]
    corpus_path = context["corpus_path"]

    def _batch_retrieve(kwargs_list: list[dict]) -> list[str]:
        queries = [kw["query"] for kw in kwargs_list]
        return batch_retrieve(queries, bm25_dir, k=7, corpus_path=corpus_path)

    batch_tool_registry = {"retrieve": _batch_retrieve}

    # 5. Run chain on dataset (step-batched for optimal vLLM batching)
    #    Per-step max_tokens: short limits for query/answer steps,
    #    generous limits for reasoning steps.
    step_max_tokens = {
        2: 4096,   # summarize retrieved facts (thinking + substantial text)
        3: 2048,   # generate search query (thinking + short query)
        5: 4096,   # combine evidence from both retrievals (thinking + substantial text)
        6: 2048,   # final answer (thinking + "Answer: X")
    }
    results = run_chain_on_dataset_stepwise(
        chain, client, dataset, outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
    )

    # 6. Extract answers from final step outputs
    predictions = [extract_answer(r.final_output) for r in results]

    # 7. Compute metrics
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    fitness = calculate_exact_match(targets, predictions)

    metrics = {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }

    # 8. Collect failure cases for reflective mutation context
    failures = []
    for sample, pred, target in zip(dataset, predictions, targets):
        if pred is None or normalize_text(pred) != normalize_text(str(target)):
            failures.append({
                "question": sample["question"],
                "gold": target,
                "predicted": pred,
            })
    # Return (metrics, artifact) tuple — FetchArtifact extracts artifact for FormatterStage
    return (metrics, failures[:10])
