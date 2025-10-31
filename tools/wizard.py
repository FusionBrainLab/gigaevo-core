from __future__ import annotations

import argparse
from pathlib import Path

from gigaevo.problems.layout import ProblemLayout as PL


def main() -> None:
    parser = argparse.ArgumentParser(
        "Problem Wizard", description="Scaffold new problem directories"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Target problem directory to create (e.g., problems/new_problem)",
    )
    parser.add_argument(
        "--add-context", action="store_true", help="Include context.py scaffold"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files if present"
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Custom task description text",
    )
    parser.add_argument(
        "--task-hints", type=str, default=None, help="Custom task hints text"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom mutation system prompt template",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help="Custom mutation user prompt template",
    )
    args = parser.parse_args()

    target_dir = Path(args.target)
    PL.scaffold(
        target_dir,
        add_context=args.add_context,
        overwrite=args.overwrite,
        task_description=args.task_description,
        task_hints=args.task_hints,
        mutation_system_prompt=args.system_prompt,
        mutation_user_prompt=args.user_prompt,
    )
    print(f"Scaffolded problem at {target_dir}")


if __name__ == "__main__":
    main()
