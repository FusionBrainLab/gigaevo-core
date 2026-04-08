#!/usr/bin/env python3
"""Check that documentation tables match actual files on disk.

Catches stale docs before they reach the remote. Designed to run fast
(<1s) as a post-commit hook.

Checks:
  1. CLAUDE.md agents table vs .claude/agents/*.md
  2. CLAUDE.md skills table vs .claude/skills/*/SKILL.md
  3. tools/README.md tool index vs tools/*.py

Exit codes: 0 = clean, 1 = stale docs found
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

PROJ = Path(__file__).resolve().parent.parent
STALE: list[str] = []


def _extract_table_col(md_path: Path, col: int, header_pattern: str) -> set[str]:
    """Extract values from a markdown table column, starting after header_pattern."""
    text = md_path.read_text()
    in_table = False
    values: set[str] = set()
    for line in text.splitlines():
        if header_pattern in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("|") and "---" not in line:
                cells = [c.strip() for c in line.split("|")]
                # Filter empty cells from leading/trailing pipes
                cells = [c for c in cells if c]
                if len(cells) > col:
                    values.add(cells[col])
            elif not line.startswith("|") and line.strip():
                break  # End of table
    return values


def check_agents():
    """CLAUDE.md agents table vs actual agent files."""
    claude_md = PROJ / "CLAUDE.md"
    agents_dir = PROJ / ".claude" / "agents"

    if not claude_md.exists() or not agents_dir.exists():
        return

    # Actual agent files (exclude directories, only .md files at top level)
    actual = {p.stem for p in agents_dir.glob("*.md")}

    # Parse agents from CLAUDE.md table (column 0 after "## Agents" header)
    text = claude_md.read_text()
    documented: set[str] = set()
    in_agents = False
    for line in text.splitlines():
        if line.strip() == "## Agents":
            in_agents = True
            continue
        if in_agents:
            if line.startswith("| `") and "---" not in line:
                match = re.search(r"`([^`]+)`", line)
                if match:
                    documented.add(match.group(1))
            elif line.startswith("##") and "Agent" not in line:
                break

    missing_from_docs = actual - documented
    removed_but_documented = documented - actual

    for name in sorted(missing_from_docs):
        STALE.append(f"Agent '{name}' exists but not in CLAUDE.md agents table")
    for name in sorted(removed_but_documented):
        STALE.append(
            f"Agent '{name}' in CLAUDE.md but .claude/agents/{name}.md not found"
        )


def check_skills():
    """CLAUDE.md skills table vs actual skill directories."""
    claude_md = PROJ / "CLAUDE.md"
    skills_dir = PROJ / ".claude" / "skills"

    if not claude_md.exists() or not skills_dir.exists():
        return

    # Actual skill directories (must contain SKILL.md)
    actual = {p.parent.name for p in skills_dir.glob("*/SKILL.md")}

    # Parse skills from CLAUDE.md table
    text = claude_md.read_text()
    documented: set[str] = set()
    in_skills = False
    for line in text.splitlines():
        if "## Skills" in line:
            in_skills = True
            continue
        if in_skills:
            if line.startswith("| `/") and "---" not in line:
                match = re.search(r"`/([^`]+)`", line)
                if match:
                    documented.add(match.group(1))
            elif line.startswith("##"):
                break

    missing_from_docs = actual - documented
    removed_but_documented = documented - actual

    for name in sorted(missing_from_docs):
        STALE.append(f"Skill '{name}' exists but not in CLAUDE.md skills table")
    for name in sorted(removed_but_documented):
        STALE.append(
            f"Skill '/{name}' in CLAUDE.md but .claude/skills/{name}/SKILL.md not found"
        )


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
    check_agents()
    check_skills()
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
