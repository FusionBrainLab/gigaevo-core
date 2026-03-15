"""Regression tests for ValidityMetricAcceptor sentinel-bypass bug.

Bug: ValidityMetricAcceptor uses 'if not is_valid' to gate validity.  The
sentinel for is_valid (when higher_is_better=True) is MIN_VALUE_DEFAULT = -1e5.
bool(-1e5) is True, so 'not -1e5' is False — the check PASSES the sentinel and
programs with garbage metrics enter the archive.

Fix: check 'is_valid is None or is_valid <= 0' instead.
"""

from __future__ import annotations

from gigaevo.evolution.engine.acceptor import ValidityMetricAcceptor
from gigaevo.programs.metrics.context import MIN_VALUE_DEFAULT
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _done_program() -> Program:
    return Program(code="def solve(): pass", state=ProgramState.DONE)


def test_validity_acceptor_rejects_sentinel_value() -> None:
    """Sentinel is_valid (-1e5) must be rejected, not accepted.

    Before the fix: bool(-1e5) is True → 'not is_valid' is False →
    check PASSES → programs with sentinel metrics pollute the archive.
    """
    program = _done_program()
    program.add_metrics({"is_valid": MIN_VALUE_DEFAULT})  # -1e5 (sentinel)
    assert ValidityMetricAcceptor().is_accepted(program) is False, (
        "ValidityMetricAcceptor accepted a program with sentinel is_valid=-1e5.  "
        "Fix: replace 'if not is_valid' with 'if is_valid is None or is_valid <= 0'."
    )


def test_validity_acceptor_rejects_arbitrary_negative_value() -> None:
    """Any negative is_valid must be rejected."""
    program = _done_program()
    program.add_metrics({"is_valid": -1.0})
    assert ValidityMetricAcceptor().is_accepted(program) is False


def test_validity_acceptor_rejects_zero() -> None:
    """is_valid=0.0 must be rejected (explicitly invalid program)."""
    program = _done_program()
    program.add_metrics({"is_valid": 0.0})
    assert ValidityMetricAcceptor().is_accepted(program) is False


def test_validity_acceptor_accepts_positive_value() -> None:
    """Sanity: is_valid=1.0 must be accepted."""
    program = _done_program()
    program.add_metrics({"is_valid": 1.0})
    assert ValidityMetricAcceptor().is_accepted(program) is True


def test_validity_acceptor_rejects_missing_key() -> None:
    """Missing is_valid key must be rejected (unchanged behaviour)."""
    program = _done_program()
    program.add_metrics({"fitness": 0.5})
    assert ValidityMetricAcceptor().is_accepted(program) is False
