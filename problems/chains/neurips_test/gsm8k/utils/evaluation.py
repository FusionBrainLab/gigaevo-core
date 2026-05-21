"""GSM8K evaluation utilities.

Extracts the final numerical answer from both the model's output and the
ground-truth answer string, then compares them for exact match.

Ground-truth format: ``"Step-by-step reasoning...\\n#### 42"``
Model output format: ``"...\\nAnswer: 42"`` (preferred) or any text containing
the number.
"""

import re
from statistics import mean


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numerical answer from a model output string.

    Tries patterns in order:
      1. ``Answer: <number>`` — preferred explicit format
      2. ``#### <number>`` — GSM8K ground-truth delimiter (if model copies it)
      3. Last standalone integer in the text

    Returns:
        Normalised integer string (e.g. ``"42"``), or ``None`` if not found.
    """
    if not text:
        return None

    # Pattern 1: explicit "Answer: <number>"
    m = re.search(r"[Aa]nswer\s*:\s*([+-]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return _normalise_number(m.group(1))

    # Pattern 2: GSM8K-style "#### <number>"
    m = re.search(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return _normalise_number(m.group(1))

    # Pattern 3: last integer in the text
    nums = re.findall(r"[+-]?\b\d[\d,]*(?:\.\d+)?\b", text)
    if nums:
        return _normalise_number(nums[-1])

    return None


def extract_gsm8k_gold(answer_str: str) -> str | None:
    """Extract the ground-truth integer from a GSM8K answer string.

    Args:
        answer_str: Full answer string ending with ``"\\n#### 42"``.

    Returns:
        Normalised integer string, or ``None`` if the delimiter is missing.
    """
    m = re.search(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)", answer_str)
    if m:
        return _normalise_number(m.group(1))
    return None


def _normalise_number(num_str: str) -> str:
    """Strip commas and normalise to a plain integer or float string."""
    cleaned = num_str.replace(",", "").strip()
    # Convert to float then back to int if it is whole
    try:
        val = float(cleaned)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return cleaned


def calculate_accuracy(
    gold_answers: list[str],
    predictions: list[str | None],
) -> float:
    """Compute exact-match accuracy between gold answers and model predictions.

    Args:
        gold_answers: Ground-truth answer strings (with ``####`` delimiter).
        predictions: Extracted model answers (``None`` for extraction failures).

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    matches = []
    for pred, gold_str in zip(predictions, gold_answers):
        if pred is None:
            matches.append(0)
            continue
        gold = extract_gsm8k_gold(gold_str)
        if gold is None:
            matches.append(0)
            continue
        matches.append(int(pred == gold))

    return mean(matches) if matches else 0.0
