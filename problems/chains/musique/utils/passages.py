"""Passage selection and formatting for MuSiQue-like datasets."""

import random
from typing import Any


def format_passage(title: str, text: str) -> str:
    """Format a single passage as 'Title | paragraph text'."""
    return f"{title} | {text}"


def _clean_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip()).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_musique_paragraphs(sample: dict) -> list[tuple[str, bool]]:
    """Extract (formatted_passage, is_supporting) pairs from MuSiQue schema."""
    paragraphs = sample.get("paragraphs")
    if not isinstance(paragraphs, list):
        return []

    extracted: list[tuple[str, bool]] = []
    for i, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            continue
        title = str(paragraph.get("title") or f"Paragraph {i + 1}")
        text = _clean_text(
            paragraph.get("paragraph_text")
            or paragraph.get("paragraph")
            or paragraph.get("text")
        )
        if not text:
            continue
        extracted.append(
            (
                format_passage(title, text),
                bool(paragraph.get("is_supporting", False)),
            )
        )
    return extracted


def _extract_hotpot_style_paragraphs(sample: dict) -> list[tuple[str, bool]]:
    """Fallback extractor for context/title/sentences schema."""
    context = sample.get("context")
    if not isinstance(context, dict):
        return []

    titles = context.get("title")
    sentences = context.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return []

    supporting_titles: set[str] = set()
    supporting_facts = sample.get("supporting_facts")
    if isinstance(supporting_facts, dict):
        titles_from_sf = supporting_facts.get("title", [])
        if isinstance(titles_from_sf, list):
            supporting_titles = {str(t) for t in titles_from_sf}

    extracted: list[tuple[str, bool]] = []
    for title, sents in zip(titles, sentences):
        title_str = str(title)
        text = _clean_text(sents)
        if not text:
            continue
        extracted.append(
            (
                format_passage(title_str, text),
                title_str in supporting_titles,
            )
        )
    return extracted


def select_passages(
    sample: dict,
    k: int = 8,
    rng: random.Random | None = None,
) -> list[str]:
    """Select supporting passages and distractors, then shuffle."""
    if rng is None:
        rng = random.Random()

    extracted = _extract_musique_paragraphs(sample)
    if not extracted:
        extracted = _extract_hotpot_style_paragraphs(sample)

    if not extracted:
        return []

    supporting = [p for p, is_supporting in extracted if is_supporting]
    distractors = [p for p, is_supporting in extracted if not is_supporting]

    if len(supporting) >= k:
        rng.shuffle(supporting)
        selected = supporting[:k]
    else:
        selected = list(supporting)
        rng.shuffle(distractors)
        selected.extend(distractors[: max(0, k - len(selected))])

    rng.shuffle(selected)
    return selected
