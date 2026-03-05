"""Evaluation utilities for HoVer retrieval coverage."""

import re
import string
import unicodedata


def normalize_text(s: str) -> str:
    """Normalize text for matching.

    Matches dspy.evaluate.normalize_text logic:
        1) Unicode NFD normalization
        2) lowercasing
        3) punctuation removal
        4) whitespace collapse
    """
    s = unicodedata.normalize("NFD", s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def extract_titles_from_passages(passages_text: str) -> set[str]:
    """Extract document titles from BM25 retrieval output.

    Each passage is formatted as "[N] Title | Text...", so we strip the
    "[N] " prefix and split on " | " to get the title — matching GEPA's
    approach: ``c.split(" | ")[0]``.

    Returns a set of normalized title strings.
    """
    titles = set()
    for line in passages_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip "[N] " prefix
        line = re.sub(r"^\[\d+\]\s*", "", line)
        # Split on " | " and take the title part (GEPA pattern)
        parts = line.split(" | ", 1)
        if parts:
            title = parts[0].strip()
            if title:
                titles.add(normalize_text(title))
    return titles


def discrete_retrieval_eval(
    gold_titles: set[str], found_titles: set[str]
) -> int:
    """Returns 1 if all gold titles were found in retrieved passages, else 0.

    Both sets should contain normalized title strings.
    Matches GEPA's discrete_retrieval_eval metric.
    """
    normalized_gold = {normalize_text(t) for t in gold_titles}
    return int(normalized_gold.issubset(found_titles))
