from collections.abc import Sequence

from problems.chains.musique_retrieval.utils.utils import normalize_text

MAX_FAILED_EXAMPLES_IN_ARTIFACT = 12
MAX_QUESTION_CHARS_IN_ARTIFACT = 280
MAX_ALIAS_CHARS_IN_ARTIFACT = 80
MAX_ALIASES_PER_EXAMPLE = 5


def is_alias_exact_match(prediction: str | None, target_aliases: Sequence[str]) -> bool:
    """Check alias-aware exact match after normalization."""
    if prediction is None:
        return False

    norm_pred = normalize_text(prediction)
    norm_targets = {
        normalize_text(str(alias)) for alias in target_aliases if str(alias).strip()
    }
    return norm_pred in norm_targets


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def build_failed_examples_artifact(
    dataset: list[dict],
    targets: list[list[str]],
    predictions: list[str | None],
    *,
    fitness: float,
    extraction_failures: float,
    max_examples: int = MAX_FAILED_EXAMPLES_IN_ARTIFACT,
) -> str:
    """Create mutation-context artifact with evaluation failures."""
    failed_examples: list[tuple[int, dict, str | None, list[str], str]] = []

    for idx, (sample, prediction, target_aliases) in enumerate(
        zip(dataset, predictions, targets)
    ):
        if is_alias_exact_match(prediction, target_aliases):
            continue
        failure_type = "extraction_failure" if prediction is None else "mismatch"
        failed_examples.append((idx, sample, prediction, target_aliases, failure_type))

    lines = [
        "## Failed Evaluation Examples (MuSiQue Retrieval)",
        "",
        f"Overall score (fitness / EM): {fitness:.4f}",
        f"Extraction failure rate: {extraction_failures:.4f}",
        f"Evaluated samples: {len(predictions)}",
        f"Failed samples: {len(failed_examples)}",
    ]

    if not failed_examples:
        lines.extend(["", "No failed samples."])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            f"Showing up to {max_examples} failed examples for mutation guidance.",
        ]
    )

    for idx, sample, prediction, target_aliases, failure_type in failed_examples[
        :max_examples
    ]:
        shown_aliases = [
            _truncate_text(alias, MAX_ALIAS_CHARS_IN_ARTIFACT)
            for alias in target_aliases[:MAX_ALIASES_PER_EXAMPLE]
        ]
        aliases_text = ", ".join(shown_aliases) if shown_aliases else "<none>"
        if len(target_aliases) > MAX_ALIASES_PER_EXAMPLE:
            aliases_text = f"{aliases_text}, ... (+{len(target_aliases) - MAX_ALIASES_PER_EXAMPLE} more)"

        lines.extend(
            [
                "",
                f"{idx + 1}. task_id={sample.get('task_id', '<unknown>')} | failure={failure_type}",
                f"Question: {_truncate_text(sample.get('question', ''), MAX_QUESTION_CHARS_IN_ARTIFACT)}",
                f"Prediction: {prediction!r}",
                f"Gold aliases: {aliases_text}",
            ]
        )

    omitted = len(failed_examples) - min(len(failed_examples), max_examples)
    if omitted > 0:
        lines.extend(["", f"... {omitted} additional failed samples omitted."])

    return "\n".join(lines)
