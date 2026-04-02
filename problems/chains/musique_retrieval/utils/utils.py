"""Text normalization utilities for MuSiQue retrieval."""

import re
import string
import unicodedata


def normalize_text(s: str) -> str:
    """Normalize text for exact match comparison.

    Steps:
        1) Unicode NFD normalization
        2) lowercasing
        3) punctuation removal
        4) English article removal ("a", "an", "the")
        5) whitespace collapse
    """
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
