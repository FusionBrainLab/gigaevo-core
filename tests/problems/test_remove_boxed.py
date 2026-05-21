"""Tests for boxed-answer extraction helpers across AIME / GSM8K duplicates.

remove_boxed is duplicated under three problem trees with identical semantics;
each copy must return None for malformed input rather than raise AssertionError.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]

_REMOVE_BOXED_PATHS = [
    ("aime_chains_utils", "problems/chains/aime/utils/utils.py"),
    ("aime_prompts_utils", "problems/prompts/aime/utils.py"),
    ("gsm8k_prompts_utils", "problems/prompts/gsm8k/utils.py"),
]


def _load(module_name: str, relative_path: str) -> ModuleType:
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=_REMOVE_BOXED_PATHS, ids=[name for name, _ in _REMOVE_BOXED_PATHS])
def boxed_module(request: pytest.FixtureRequest) -> ModuleType:
    module_name, relative_path = request.param
    return _load(module_name, relative_path)


# ---------------------------------------------------------------------------
# Behavior for well-formed inputs must be preserved.
# ---------------------------------------------------------------------------


def test_remove_boxed_extracts_brace_form(boxed_module: ModuleType) -> None:
    assert boxed_module.remove_boxed("\\boxed{42}") == "42"


def test_remove_boxed_extracts_space_form(boxed_module: ModuleType) -> None:
    assert boxed_module.remove_boxed("\\boxed 42") == "42"


def test_remove_boxed_returns_none_when_no_boxed(boxed_module: ModuleType) -> None:
    assert boxed_module.remove_boxed("plain text") is None


def test_remove_boxed_returns_none_for_empty(boxed_module: ModuleType) -> None:
    assert boxed_module.remove_boxed("") is None


# ---------------------------------------------------------------------------
# Regression: malformed inputs must return None instead of raising.
# ---------------------------------------------------------------------------


def test_remove_boxed_truncated_brace_returns_none(boxed_module: ModuleType) -> None:
    """Mid-response truncation (no closing brace) must not crash extraction."""
    assert boxed_module.remove_boxed("\\boxed{42") is None


def test_remove_boxed_trailing_garbage_returns_none(boxed_module: ModuleType) -> None:
    """Trailing characters after the boxed expression must not crash extraction."""
    assert boxed_module.remove_boxed("\\boxed{42}xyz") is None


def test_remove_boxed_composed_with_last_boxed_only_string(
    boxed_module: ModuleType,
) -> None:
    """Standard call-site composition: feed last_boxed_only_string output through
    remove_boxed. Well-formed inputs extract; bare prose returns None."""
    extract = boxed_module.last_boxed_only_string
    parse = boxed_module.remove_boxed

    assert parse(extract("the answer is \\boxed{42} done")) == "42"
    assert parse(extract("nothing relevant")) is None
