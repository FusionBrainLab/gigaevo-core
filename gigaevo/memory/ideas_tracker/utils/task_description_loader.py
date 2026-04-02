from __future__ import annotations

import os
from pathlib import Path


def load_task_description(redis_prefix: str, idea_tracker_location: Path) -> str:
    """
    Load human-readable task description for the current experiment.

    Searches problems/ directory tree for a leaf directory matching redis_prefix
    and loads task_description.txt from it. Returns placeholder if not found.

    Returns:
        Task description text from matching directory, or "No description available".
    """
    prefix_value = redis_prefix or ""
    if not prefix_value:
        return "No description available"
    prefix_value = prefix_value.replace("/", "_")

    project_root = idea_tracker_location.parents[3]
    problems_root = project_root / "problems"

    try:
        # Walk the problems tree and collect all leaf directories.
        leaf_dirs: list[Path] = []
        for root, dirs, _files in os.walk(problems_root):
            if "initial_programs" in dirs:
                leaf_dirs.append(Path(root))

        for leaf in leaf_dirs:
            split_index = leaf.parts.index("problems") + 1
            true_name = "_".join(leaf.parts[split_index:])
            if true_name == prefix_value:
                candidate_file = leaf / "task_description.txt"
                if candidate_file.is_file():
                    return candidate_file.read_text(encoding="utf-8").strip()
    except Exception:
        return "No description available"

    return "No description available"
