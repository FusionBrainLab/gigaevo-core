"""Single-source harm predicate + threshold-free decision expressions.

Every eviction harm decision must resolve to the one
``BetaBinomialReputation.is_confidently_harmful``; a second concrete definition
anywhere under ``gigaevo/memory`` is copy drift. Decision comparisons in the
core gate/posterior modules must take thresholds from constructor fields —
inline numeric comparators (beyond 0/1 identity values) fail here so new
hardcoded thresholds are caught automatically.
"""

from __future__ import annotations

import ast
from pathlib import Path

import gigaevo.memory
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.reputation import BetaBinomialReputation

_MEMORY_ROOT = Path(gigaevo.memory.__file__).parent
_ALLOWED_PREDICATE_FILES = {
    _MEMORY_ROOT / "core" / "protocols.py",
    _MEMORY_ROOT / "core" / "reputation.py",
}
_DECISION_MODULES = (
    "core/auctioneer.py",
    "core/budgeter.py",
    "core/evictor.py",
    "core/reputation.py",
    "shared_memory/injection_posterior.py",
)
_IDENTITY_VALUES = {0, 1}


def _memory_sources() -> list[Path]:
    return [path for path in _MEMORY_ROOT.rglob("*.py") if "_vendor" not in path.parts]


def test_is_confidently_harmful_defined_only_in_core_reputation_and_protocol():
    definitions: dict[Path, list[int]] = {}
    for path in _memory_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "is_confidently_harmful"
            ):
                definitions.setdefault(path, []).append(node.lineno)
    extra = set(definitions) - _ALLOWED_PREDICATE_FILES
    assert not extra, f"copy-drift definitions of is_confidently_harmful: {extra}"
    assert _MEMORY_ROOT / "core" / "reputation.py" in definitions


def test_evictor_resolves_to_the_same_predicate_symbol():
    canonical = BetaBinomialReputation.is_confidently_harmful
    for site in (HarmEvictor(),):
        assert site.reputation.is_confidently_harmful.__func__ is canonical


def test_core_decision_comparisons_take_thresholds_from_fields():
    violations: list[str] = []
    for rel in _DECISION_MODULES:
        path = _MEMORY_ROOT / rel
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for operand in [node.left, *node.comparators]:
                if (
                    isinstance(operand, ast.Constant)
                    and isinstance(operand.value, (int, float))
                    and not isinstance(operand.value, bool)
                    and operand.value not in _IDENTITY_VALUES
                ):
                    violations.append(f"{rel}:{operand.lineno} -> {operand.value}")
    assert not violations, f"magic literals in decision comparisons: {violations}"
