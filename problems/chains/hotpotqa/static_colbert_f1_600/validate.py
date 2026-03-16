"""HotpotQA validation — ColBERTv2 retriever + richer failure feedback.

F1-600 variant with two enhancements over static_f1_600:
  1. ColBERTv2 retriever instead of BM25 (higher recall on multi-hop queries)
  2. Richer ASI feedback: missing gold docs shown with full passage text,
     not just titles — gives the mutation LLM actionable content to improve queries.
"""

from pathlib import Path
import pickle
import re
from statistics import mean
import threading

from problems.chains.chain_runner import run_chain_on_dataset_stepwise
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.shared_config import (
    CORPUS_PATH,
    DATASET_CONFIG,
    LLM_CONFIG,
    load_jsonl,
    outer_context_builder,
    preprocess_sample,
)
from problems.chains.hotpotqa.static.config import STATIC_CHAIN_TOPOLOGY, load_baseline
from problems.chains.hotpotqa.utils.retrieval import make_batch_tool_fn
from problems.chains.hotpotqa.utils.utils import normalize_text

# ---------------------------------------------------------------------------
# Lazy corpus title → passage lookup (singleton per process)
# ---------------------------------------------------------------------------

_title_to_passage: dict[str, str] | None = None
_title_lookup_lock = threading.Lock()


def _get_title_to_passage() -> dict[str, str]:
    """Build {title: 'Title | text'} dict from corpus PKL (lazy, cached)."""
    global _title_to_passage
    if _title_to_passage is not None:
        return _title_to_passage
    with _title_lookup_lock:
        if _title_to_passage is not None:
            return _title_to_passage
        corpus_path = Path(CORPUS_PATH)
        if corpus_path.suffix == ".pkl":
            with open(corpus_path, "rb") as f:
                passages: list[str] = pickle.load(f)
        else:
            import gzip
            import json

            passages = []
            opener = gzip.open if corpus_path.suffix == ".gz" else open
            with opener(corpus_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        passages.append(
                            f"{doc.get('title', '')} | {doc.get('text', '')}"
                        )
        result: dict[str, str] = {}
        for passage in passages:
            sep = passage.find(" | ")
            if sep != -1:
                result[passage[:sep]] = passage
        _title_to_passage = result
    return _title_to_passage


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


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


def _f1_single(prediction: str, reference: str) -> float:
    """Compute SQuAD-style token-level F1 between a single prediction and reference."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
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


def calculate_f1(targets: list[str], predictions: list[str | None]) -> float:
    scores = [
        0.0 if pred is None else _f1_single(pred, str(target))
        for pred, target in zip(predictions, targets)
    ]
    return mean(scores) if scores else 0.0


def calculate_exact_match(targets: list[str], predictions: list[str | None]) -> float:
    matches = [
        0 if pred is None else int(normalize_text(pred) == normalize_text(str(target)))
        for pred, target in zip(predictions, targets)
    ]
    return mean(matches) if matches else 0.0


def parse_retrieved_titles(step_output: str) -> list[str]:
    """Parse document titles from retrieval output.

    Input format: "[1] Title | passage text\\n[2] Title | passage text..."
    Returns list of title strings.
    """
    return re.findall(r"\[(?:\d+)\]\s+(.+?)\s+\|", step_output)


# ---------------------------------------------------------------------------
# Main validate function
# ---------------------------------------------------------------------------


def validate(chain_spec: dict) -> tuple[dict, list[dict]]:
    """Validate chain specification and compute fitness metrics.

    ColBERT-F1-600 variant:
      - Fitness: token-level F1 on first-600 train samples (primary)
      - EM: secondary metric for GEPA comparison
      - Retriever: ColBERTv2 (higher recall than BM25, especially on second-hop)
      - ASI feedback: missing gold docs shown with full passage text, not just titles

    step_outputs[0] = hop-1 ColBERT retrieved passages
    step_outputs[3] = hop-2 ColBERT retrieved passages

    Returns:
        (metrics, failures) where metrics["fitness"] = F1, metrics["em"] = EM.
        failures = richer ASI failure cases (all; formatter random-samples 10).
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

    # 4. Build retriever — uses ColBERTServerRetriever when
    # HOTPOTQA_COLBERT_SERVER_URL is set (preferred for exec_runner workers),
    # falling back to in-process ColBERTRetriever otherwise.
    from problems.chains.hotpotqa.shared_config import build_retriever

    retriever = build_retriever(k=7)
    batch_tool_registry = {"retrieve": make_batch_tool_fn(retriever)}

    # 5. Run chain on dataset (step-batched for optimal vLLM batching)
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
    )

    # 6. Extract answers
    predictions = [extract_answer(r.final_output) for r in results]

    # 7. Compute metrics
    extraction_failures = (
        sum(1 for p in predictions if p is None) / len(predictions)
        if predictions
        else 0.0
    )
    f1_score = calculate_f1(targets, predictions)
    em_score = calculate_exact_match(targets, predictions)
    metrics = {
        "fitness": f1_score,
        "em": em_score,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }

    # 8. Collect richer ASI failure cases
    # Load corpus title→passage mapping for full-text lookup of missing gold docs.
    title_to_passage = _get_title_to_passage()

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
            # Full passage text for missing docs (richer signal than titles alone)
            hop1_missing_passages = [
                title_to_passage.get(t, f"{t} | (not found in corpus)")
                for t in hop1_missing
            ]
            hop2_missing_passages = [
                title_to_passage.get(t, f"{t} | (not found in corpus)")
                for t in hop2_missing
            ]
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
                    "hop1_missing_passages": hop1_missing_passages,
                    "hop2_missing_passages": hop2_missing_passages,
                }
            )

    return (metrics, failures)
