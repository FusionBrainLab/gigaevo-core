from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.problems.context import ProblemContext

_TABULAR = Path(__file__).resolve().parents[3] / "problems" / "tabular"
_ALL = [
    "california",
    "house",
    "diamond",
    "black-friday",
    "microsoft",
    "adult",
    "churn",
    "higgs-small",
    "otto",
    "covtype2",
]


@pytest.mark.parametrize("name", _ALL)
def test_problem_dir_structure(name):
    d = _TABULAR / name
    assert (d / "task_description.txt").is_file()
    assert (d / "validate.py").is_file()  # follows symlink
    assert (d / "metrics.yaml").is_file()
    assert (d / "initial_programs").is_dir()
    progs = list((d / "initial_programs").glob("*.py"))
    assert len(progs) == 5


@pytest.mark.parametrize("name", _ALL)
def test_problem_context_loads(name):
    ctx = ProblemContext(problem_dir=_TABULAR / name)
    ctx.validate()
    primary = ctx.metrics_context.get_primary_key()
    assert primary == "fitness"
