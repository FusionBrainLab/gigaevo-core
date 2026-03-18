"""Passage selection and formatting for HotpotQA-QA.

Copied from prompts/hotpotqa_qa/config.py to keep the chain problem self-contained.
"""

import random


def format_passage(title: str, sentences: list[str]) -> str:
    """Format a single passage as 'Title | sentence1 sentence2 ...'"""
    return f"{title} | {' '.join(sentences)}"


def select_passages(
    sample: dict, k: int = 7, rng: random.Random | None = None
) -> list[str]:
    """Select golden passages plus distractors, shuffled.

    Always includes all golden (supporting_facts) passages, then fills
    remaining slots with randomly selected distractors.

    Args:
        sample: Raw HotpotQA sample with context and supporting_facts.
        k: Total number of passages to return.
        rng: Random number generator for reproducibility.

    Returns:
        List of formatted passage strings.
    """
    if rng is None:
        rng = random.Random()

    golden_titles = set(sample["supporting_facts"]["title"])

    context_titles = sample["context"]["title"]
    context_sentences = sample["context"]["sentences"]
    title_to_sentences = {
        title: sentences for title, sentences in zip(context_titles, context_sentences)
    }

    golden_passages = []
    distractor_passages = []

    for title in context_titles:
        formatted = format_passage(title, title_to_sentences[title])
        if title in golden_titles:
            golden_passages.append(formatted)
        else:
            distractor_passages.append(formatted)

    selected = golden_passages.copy()

    remaining_slots = k - len(selected)
    if remaining_slots > 0:
        rng.shuffle(distractor_passages)
        selected.extend(distractor_passages[:remaining_slots])

    rng.shuffle(selected)

    return selected
