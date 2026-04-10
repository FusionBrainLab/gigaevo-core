#!/usr/bin/env python3
"""Check that documentation tables match actual files on disk.

Catches stale docs before they reach the remote. Designed to run fast
(<1s) as a post-commit hook.

Checks:
  1. tools/README.md tool index vs tools/*.py

Agents and skills are auto-discovered by Claude Code from frontmatter
(see CLAUDE.md "Skills and Agents" section) — no manual tables to check.

Exit codes: 0 = clean, 1 = stale docs found
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

PROJ = Path(__file__).resolve().parent.parent
STALE: list[str] = []


def check_tools_readme():
    """tools/README.md tool index vs actual tool .py files."""
    readme = PROJ / "tools" / "README.md"
    tools_dir = PROJ / "tools"

    if not readme.exists():
        return

    # Actual .py tools (exclude __init__.py, __pycache__, test files)
    actual_general = {
        p.stem
        for p in tools_dir.glob("*.py")
        if p.stem not in ("__init__", "utils", "conftest")
        and not p.stem.startswith("test_")
    }
    actual_experiment = {
        p.stem
        for p in (tools_dir / "experiment").glob("*.py")
        if p.stem not in ("__init__", "conftest") and not p.stem.startswith("test_")
    }

    # Parse documented tools from README
    text = readme.read_text()
    documented: set[str] = set()
    for match in re.finditer(r"`(\w+)\.py`", text):
        documented.add(match.group(1))
    # Also catch shell scripts documented as tools
    for match in re.finditer(r"`(\w+)\.sh`", text):
        documented.add(match.group(1))

    all_actual = actual_general | actual_experiment
    missing = all_actual - documented

    for name in sorted(missing):
        STALE.append(f"Tool '{name}.py' exists but not documented in tools/README.md")


def main():
    check_tools_readme()

    if STALE:
        print(f"Documentation freshness check: {len(STALE)} issue(s) found\n")
        for issue in STALE:
            print(f"  - {issue}")
        print(
            "\nUpdate the relevant docs to match actual files, "
            "or add new entries for new files."
        )
        return 1
    else:
        print("Documentation freshness check: all clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
