"""CLI command: gigaevo run -- start an evolution run from any directory."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import click


def find_repo_root() -> Path:
    """Locate the source checkout that owns the installed gigaevo package.

    `run.py` and `config/` live at the repo root, not inside the package, so
    launching runs requires an editable/source install.
    """
    import gigaevo

    root = Path(gigaevo.__file__).resolve().parent.parent
    if not (root / "run.py").is_file():
        raise click.ClickException(
            f"run.py not found next to the gigaevo package (looked in {root}). "
            "`gigaevo run` needs a source checkout: install with "
            "`pip install -e .` from a git clone."
        )
    return root


def resolve_problem(target: str, repo_root: Path) -> tuple[str, Path]:
    """Resolve TARGET to (problem_name, absolute problem directory).

    TARGET is either a path to a problem directory (external problems) or the
    bare name of a problem bundled under <repo>/problems/. An existing local
    directory wins over a bundled problem of the same name.
    """
    from gigaevo.problems.layout import ProblemLayout

    candidate = Path(target).expanduser()
    if candidate.is_dir():
        problem_dir = candidate.resolve()
        name = problem_dir.name
    elif os.sep in target or target.startswith("~"):
        raise click.ClickException(f"Problem directory not found: {target}")
    else:
        name = target
        problem_dir = (repo_root / "problems" / target).resolve()
        if not problem_dir.is_dir():
            raise click.ClickException(
                f"'{target}' is neither a directory nor a bundled problem "
                f"under {repo_root / 'problems'}. Pass a path to a problem "
                "directory or a bundled problem name."
            )

    missing = [
        f for f in ProblemLayout.required_files() if not (problem_dir / f).is_file()
    ]
    if missing:
        raise click.ClickException(
            f"{problem_dir} is not a valid problem directory "
            f"(missing: {', '.join(missing)}). "
            f"Required: {', '.join(ProblemLayout.required_files())}."
        )
    return name, problem_dir


def build_run_argv(
    repo_root: Path, name: str, problem_dir: Path, overrides: tuple[str, ...]
) -> list[str]:
    """Build the run.py argv, injecting problem.name/dir unless user-supplied.

    Hydra errors on duplicate plain overrides, so each injection is skipped
    when the user passed their own.
    """
    argv = [sys.executable, str(repo_root / "run.py")]
    if not any(o.startswith("problem.name=") for o in overrides):
        argv.append(f"problem.name={name}")
    if not any(o.startswith("problem.dir=") for o in overrides):
        argv.append(f"problem.dir={problem_dir}")
    argv.extend(overrides)
    return argv


@click.command("run", context_settings={"ignore_unknown_options": True})
@click.argument("target")
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def run(target: str, overrides: tuple[str, ...]) -> None:
    """Start an evolution run on TARGET without cd-ing into the repo.

    TARGET is a path to a problem directory (for problems living outside the
    repo, e.g. when GigaEvo is used as a tool by another app) or the name of
    a problem bundled under the repo's problems/ tree. Everything after
    TARGET is passed to run.py verbatim: Hydra overrides and flags alike.

    \b
    Examples
    --------
      gigaevo run heilbron max_mutants=250 llm=codex
      gigaevo run /path/to/my_problem hydra.run.dir=outputs/my_run
      gigaevo run ./my_problem --cfg job          # preview composed config

    The process execs `python run.py problem.name=<name>
    problem.dir=<abs path> [overrides...]` from the source checkout. The
    working directory is preserved, so Hydra's default output dir
    (outputs/<date>/<time>) lands under the CALLER's cwd; pass
    hydra.run.dir=... to pin it.
    """
    repo_root = find_repo_root()
    name, problem_dir = resolve_problem(target, repo_root)
    argv = build_run_argv(repo_root, name, problem_dir, overrides)
    os.execv(sys.executable, argv)
