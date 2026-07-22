from __future__ import annotations

from pathlib import Path
import re

from gigaevo.problems.context import ProblemContext

_SECTION_PATTERN = re.compile(
    r"^(TASK|DATASET|COLUMNS|CONTRACT|PROTOCOL|STRATEGY|CONSTRAINTS)\b.*$",
    re.MULTILINE,
)


def _extract_dataset_context(text: str) -> str:
    sections: dict[str, str] = {}
    matches = list(_SECTION_PATTERN.finditer(text))
    for index, match in enumerate(matches):
        start = match.start()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[start:stop].strip()

    selected = [
        sections[name] for name in ("TASK", "DATASET", "COLUMNS") if name in sections
    ]
    if not selected:
        raise ValueError(
            "tabular task description has no TASK, DATASET, or COLUMNS sections"
        )
    return "\n\n".join(selected)


class DagTabProblemContext(ProblemContext):
    """FeatureGraph ABI combined with the selected tabular dataset semantics."""

    def __init__(self, problem_dir: str | Path, dataset: str = "california"):
        super().__init__(problem_dir)
        self.dataset = dataset

    @property
    def task_description(self) -> str:
        abi = super().task_description
        dataset_path = (
            self.problem_dir.parent / "tabular" / self.dataset / "task_description.txt"
        )
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Missing tabular task description for dataset {self.dataset!r}: {dataset_path}"
            )
        dataset_context = _extract_dataset_context(dataset_path.read_text())
        return (
            f"{abi}\n\n"
            "SELECTED DATASET CONTEXT\n"
            f"dataset id: {self.dataset}\n"
            "Column indices [j] below map exactly to FeatureGraph names xj. "
            "Use the supplied semantics when present; do not invent semantics for "
            "anonymized columns.\n\n"
            f"{dataset_context}"
        )
