import hashlib
import json
import random
import re
from statistics import mean

from problems.chains.chain_runner import run_chain_on_dataset_stepwise
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.client import LLMClient
from problems.chains.hotpotqa.shared_config import (
    CORPUS_PATH,
    BM25S_INDEX_DIR,
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


def parse_retrieved_titles(step_output: str) -> list[str]:
    """Parse document titles from BM25 retrieval output.

    Input format: "[1] Title | passage text\n[2] Title | passage text..."
    Returns list of title strings.
    """
    return re.findall(r"\[(?:\d+)\]\s+(.+?)\s+\|", step_output)


def validate(chain_spec: dict) -> dict:
    """Validate chain specification and compute fitness metrics.

    P1 (Rotation) + P2 (ASI) combined:
    - P1: chain_spec-hash-seeded random subset of 300 from 1000 training samples
    - P2: per-hop BM25 retrieval recall vs gold supporting docs in failure cases

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps

    Returns:
        (metrics, failures[:10]) tuple — metrics dict + ASI-enhanced failure cases
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load rotated 300-sample subset seeded by chain_spec hash (P1)
    #    Keep raw samples for supporting_facts access (P2)
    raw_all = load_jsonl(DATASET_CONFIG["train_path"])  # 1000 samples
    spec_seed = int(
        hashlib.sha256(
            json.dumps(chain_spec, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        16,
    ) % (2**32)
    rng = random.Random(spec_seed)
    raw_300 = rng.sample(raw_all, 300)
    dataset = [preprocess_sample(s) for s in raw_300]
    targets = [s[DATASET_CONFIG["target_field"]] for s in dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build batch tool registry
    def _batch_retrieve(kwargs_list: list[dict]) -> list[str]:
        queries = [kw["query"] for kw in kwargs_list]
        return batch_retrieve(queries, BM25S_INDEX_DIR, k=7, corpus_path=CORPUS_PATH)

    batch_tool_registry = {"retrieve": _batch_retrieve}

    # 5. Run chain
    # Per-step max_tokens: generous for all steps — thinking mode <think> blocks
    # can consume 1000-2000 tokens. Steps 3/6 had 2048 which was insufficient.
    step_max_tokens = {
        2: 8192,
        3: 8192,
        5: 8192,
        6: 8192,
    }
    results = run_chain_on_dataset_stepwise(
        chain, client, dataset, outer_context_builder,
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
    fitness = calculate_exact_match(targets, predictions)
    metrics = {
        "fitness": fitness,
        "avg_extraction_failures": extraction_failures,
        "is_valid": 1,
    }

    # 8. Collect ASI-enhanced failure cases with per-hop retrieval diagnostics (P2)
    failures = []
    for raw_s, sample, result, pred, target in zip(
        raw_300, dataset, results, predictions, targets
    ):
        if pred is None or normalize_text(pred) != normalize_text(str(target)):
            gold_titles = set(
                raw_s.get("supporting_facts", {}).get("title", [])
            )
            hop1_out = result.step_outputs[0] if len(result.step_outputs) > 0 else ""
            hop2_out = result.step_outputs[3] if len(result.step_outputs) > 3 else ""
            hop1_titles = set(parse_retrieved_titles(hop1_out))
            hop2_titles = set(parse_retrieved_titles(hop2_out))
            hop1_missing = sorted(gold_titles - hop1_titles)
            hop2_missing = sorted(gold_titles - hop2_titles)
            failures.append({
                "question": sample["question"],
                "gold": target,
                "predicted": pred,
                "hop1_retrieved": len(hop1_titles & gold_titles),
                "hop2_retrieved": len(hop2_titles & gold_titles),
                "n_gold": len(gold_titles),
                "hop1_missing": hop1_missing,
                "hop2_missing": hop2_missing,
            })

    return (metrics, failures[:10])
