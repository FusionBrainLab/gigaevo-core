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

# Held-out validation split indices
EVO_END = 700  # train[0:EVO_END]     — evo set (mutation feedback source)
HELD_START = 700  # train[HELD_START:]   — held-out val (fitness signal, no feedback)
VAL_600_END = (
    600  # train[0:VAL_600_END] — supplementary EM for cold_start gap comparison
)


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
    """Calculate mean token-level F1 (SQuAD-style) after text normalization."""
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
    """Calculate Exact Match (EM) after text normalization."""
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
    """Parse document titles from BM25 retrieval output."""
    return re.findall(r"\[(?:\d+)\]\s+(.+?)\s+\|", step_output)


def validate(chain_spec: dict) -> tuple[dict, list[dict]]:
    """Validate chain specification and compute fitness metrics.

    Held-out validation variant: split train[0:1000] into an evolution set
    (train[0:700]) and a held-out validation set (train[700:1000]).

    CRITICAL DESIGN INVARIANT:
    - The mutation LLM receives failure examples ONLY from the evo set.
    - The held-out set is evaluated for fitness (held_F1) but never exposed as
      failure cases. This provides a fully unbiased selection signal.
    - fitness = held_F1 only (NOT averaged with evo_F1).

    Metrics returned:
    - fitness:     held_F1 — drives MAP-Elites selection (unbiased signal)
    - evo_f1:      F1 on evo set — shown to mutation LLM as gap signal
    - em:          EM on full 1000 samples (evo + held) — for val-test gap analysis
    - val_em_600:  EM on train[0:600] — for comparable gap vs cold_start reference
    - avg_extraction_failures: fraction of evo-set samples with failed answer extraction
    - is_valid:    1 if chain is structurally valid

    Args:
        chain_spec: Dict from entrypoint() with system_prompt and steps

    Returns:
        (metrics, failures) where failures come ONLY from the evo set.
    """
    # 1. Structural validation
    baseline = load_baseline()
    chain = validate_chain_spec(
        chain_spec,
        mode="static",
        topology=STATIC_CHAIN_TOPOLOGY,
        frozen_baseline=baseline,
    )

    # 2. Load train data and split into evo set and held-out set
    all_train = load_jsonl(DATASET_CONFIG["train_path"])
    raw_evo = all_train[:EVO_END]  # train[0:700] — mutation feedback source
    raw_held = all_train[HELD_START:]  # train[700:1000] — held-out val (no feedback)

    evo_dataset = [preprocess_sample(s) for s in raw_evo]
    held_dataset = [preprocess_sample(s) for s in raw_held]
    evo_targets = [s[DATASET_CONFIG["target_field"]] for s in evo_dataset]
    held_targets = [s[DATASET_CONFIG["target_field"]] for s in held_dataset]

    # 3. Create LLM client
    client = LLMClient(**LLM_CONFIG)

    # 4. Build batch tool registry
    def _batch_retrieve(kwargs_list: list[dict]) -> list[str]:
        queries = [kw["query"] for kw in kwargs_list]
        return batch_retrieve(queries, BM25S_INDEX_DIR, k=7, corpus_path=CORPUS_PATH)

    batch_tool_registry = {"retrieve": _batch_retrieve}

    step_max_tokens = {
        2: 8192,
        3: 8192,
        5: 8192,
        6: 8192,
    }

    # 5. Run chain on evo set (train[0:700])
    evo_results = run_chain_on_dataset_stepwise(
        chain,
        client,
        evo_dataset,
        outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
    )

    # 6. Run chain on held-out set (train[700:1000])
    # This set is scored for fitness ONLY — its results are never used for failure feedback.
    held_results = run_chain_on_dataset_stepwise(
        chain,
        client,
        held_dataset,
        outer_context_builder,
        batch_tool_registry=batch_tool_registry,
        step_max_tokens=step_max_tokens,
    )

    # 7. Extract answers
    evo_predictions = [extract_answer(r.final_output) for r in evo_results]
    held_predictions = [extract_answer(r.final_output) for r in held_results]

    # 8. Compute metrics
    evo_f1 = calculate_f1(evo_targets, evo_predictions)
    held_f1 = calculate_f1(held_targets, held_predictions)

    # EM on full 1000 samples (evo + held concatenated) — for val-test gap analysis
    all_targets = evo_targets + held_targets
    all_predictions = evo_predictions + held_predictions
    all_em = calculate_exact_match(all_targets, all_predictions)

    # EM on train[0:600] only — for comparable val-test gap vs cold_start reference
    val_em_600 = calculate_exact_match(
        evo_targets[:VAL_600_END], evo_predictions[:VAL_600_END]
    )

    # Extraction failures on evo set (this is what the mutation LLM acts on)
    evo_extraction_failures = (
        sum(1 for p in evo_predictions if p is None) / len(evo_predictions)
        if evo_predictions
        else 0.0
    )

    metrics = {
        "fitness": held_f1,  # held-out F1 — pure unbiased selection signal
        "evo_f1": evo_f1,  # evo-set F1 — shown to mutation LLM as gap signal
        "em": all_em,  # full 1000-sample EM — for val-test gap analysis
        "val_em_600": val_em_600,  # EM on train[0:600] — comparable gap vs cold_start
        "avg_extraction_failures": evo_extraction_failures,
        "is_valid": 1,
    }

    # 9. Collect ASI-enhanced failure cases — EVO SET ONLY
    # CRITICAL: held-out set results are NEVER included here.
    # The mutation LLM may only see failure examples from the evo set (train[0:700]).
    # This is the invariant that makes held_F1 an unbiased selection signal.
    failures = []
    for raw_s, sample, result, pred, target in zip(
        raw_evo, evo_dataset, evo_results, evo_predictions, evo_targets
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
