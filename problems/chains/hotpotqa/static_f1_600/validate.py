import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset_stepwise
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.shared_config import (
    BM25S_INDEX_DIR,
    CORPUS_PATH,
    DATASET_CONFIG,
    LLM_CONFIG,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.hotpotqa.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hotpotqa.utils.retrieval import batch_retrieve
from problems.chains.hotpotqa.utils.utils import normalize_text


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from vLLM thinking-mode output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_answer(response: str) -> str | None:
    """Extract answer from LLM response looking for 'Answer:' pattern.

    Strips <think> blocks first so re.search does not match 'Answer:' occurrences
    inside the model's internal reasoning trace.
    """
    cleaned = strip_thinking(response)
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return answer if answer else None
    return None


def get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return normalize_text(text).split()


def _f1_single(prediction: str, reference: str) -> float:
    """Compute SQuAD-style token-level F1 between a single prediction and reference."""
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    n_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_f1(
    targets: list[str],
    predictions: list[str | None],
) -> float:
    """Calculate mean token-level F1 (SQuAD-style) after text normalization.

    Args:
        targets: List of gold answer strings
        predictions: List of predicted answer strings (None for extraction failures)

    Returns:
        Mean F1 score as a float in [0, 1]
    """
    scores = []
    for pred, target in zip(predictions, targets):
        if pred is None:
            scores.append(0.0)
        else:
            scores.append(_f1_single(pred, str(target)))
    return mean(scores) if scores else 0.0


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
        else:
            norm_pred = normalize_text(pred)
            norm_target = normalize_text(str(target))
            matches.append(int(norm_pred == norm_target))
    return mean(matches) if matches else 0.0


def parse_retrieved_titles(step_output: str) -> list[str]:
    """Parse document titles from BM25 retrieval output.

    Input format: "[1] Title | passage text\n[2] Title | passage text..."
    Returns list of title strings.
    """
    return re.findall(r"\[(?:\d+)\]\s+(.+?)\s+\|", step_output)


def validate(chain_spec: dict) -> tuple[dict, list[dict]]:
    """Validate chain specification and compute fitness metrics.

    F1-600 variant: evolves on token-level F1 (SQuAD-style) on the fixed first-600
    training samples. Combines the F1 fitness signal (smoother, less overfit to
    surface forms than EM) with the larger validation set (lower selection noise
    floor than 300-sample val).

    EM is still computed and returned as a secondary metric for monitoring and
    final test comparison (GEPA benchmark uses EM). The valid_frontier_em Redis
    key is populated from the "em" field for within-metric gap analysis.

    step_outputs[0] = hop-1 BM25 retrieved passages
    step_outputs[3] = hop-2 BM25 retrieved passages

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps

    Returns:
        (metrics, failures) where metrics["fitness"] = F1, metrics["em"] = EM.
        failures = ASI-enhanced failure cases (all failures, random-sampled by formatter).
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load fixed first-600 samples
    raw_600 = load_jsonl(DATASET_CONFIG["train_path"])[:600]
    dataset = [preprocess_sample(s) for s in raw_600]
    targets = [s[DATASET_CONFIG["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build batch tool registry
    def _batch_retrieve(kwargs_list: list[dict]) -> list[str]:
        queries = [kw["query"] for kw in kwargs_list]
        return batch_retrieve(queries, BM25S_INDEX_DIR, k=7, corpus_path=CORPUS_PATH)

    batch_tool_registry = {"retrieve": _batch_retrieve}

    # 5. Run chain on dataset
    step_max_tokens = {
        2: 8192,  # summarize retrieved facts
        3: 8192,  # generate search query
        5: 8192,  # combine evidence
        6: 8192,  # final answer
    }
    results = run_chain_on_dataset_stepwise(
        chain,
        client,
        dataset,
        outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
    )

    # 6. Extract answers
    predictions = [extract_answer(r.final_output) for r in results]

    # 7. Compute metrics — F1 is primary (fitness), EM is secondary (diagnostic)
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )

    f1_score = calculate_f1(targets, predictions)
    em_score = calculate_exact_match(targets, predictions)

    metrics = {
        "fitness": f1_score,  # primary — drives MAP-Elites selection
        "em": em_score,  # diagnostic — not used for selection
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }

    # 8. Collect ASI-enhanced failure cases
    # Failure criterion: EM=0 (same as static_f1/validate.py), NOT F1 < 1.0.
    # Using EM=0 isolates the fitness metric effect from the mutation feedback signal.
    failures = []
    for raw_s, sample, result, pred, target in zip(
        raw_600, dataset, results, predictions, targets
    ):
        if pred is None or normalize_text(pred) != normalize_text(str(target)):
            gold_titles = set(raw_s.get("supporting_facts", {}).get("title", []))
            hop1_out = result.step_outputs[0] if len(result.step_outputs) > 0 else ""
            hop2_out = result.step_outputs[3] if len(result.step_outputs) > 3 else ""
            hop1_titles = set(parse_retrieved_titles(hop1_out))
            hop2_titles = set(parse_retrieved_titles(hop2_out))
            hop1_missing = sorted(gold_titles - hop1_titles)
            hop2_missing = sorted(gold_titles - hop2_titles)
            failures.append(
                {
                    "question": sample["question"],
                    "gold": target,
                    "predicted": pred,
                    "hop1_retrieved": len(hop1_titles & gold_titles),
                    "hop2_retrieved": len(hop2_titles & gold_titles),
                    "n_gold": len(gold_titles),
                    "hop1_missing": hop1_missing,
                    "hop2_missing": hop2_missing,
                }
            )

    return (metrics, failures)
