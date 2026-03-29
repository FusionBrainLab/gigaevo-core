"""Tests for program evolution acceptors.

Covers all acceptor classes: StateAcceptor, MetricsExistenceAcceptor,
ValidityMetricAcceptor, RequiredBehaviorKeysAcceptor, MutationContextAcceptor,
CompositeAcceptor, DefaultProgramEvolutionAcceptor, StandardEvolutionAcceptor.

Focuses on boundary conditions, sentinel values, and composition edge cases.
"""

from __future__ import annotations

import pytest

from gigaevo.evolution.engine.acceptor import (
    CompositeAcceptor,
    DefaultProgramEvolutionAcceptor,
    MetricsExistenceAcceptor,
    MutationContextAcceptor,
    RequiredBehaviorKeysAcceptor,
    StandardEvolutionAcceptor,
    StateAcceptor,
    ValidityMetricAcceptor,
)
from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(
    state: ProgramState = ProgramState.DONE,
    metrics: dict | None = None,
    metadata: dict | None = None,
) -> Program:
    p = Program(code="def solve(): return 1", state=state, atomic_counter=999)
    if metrics is not None:
        p.metrics = metrics
    if metadata is not None:
        for k, v in metadata.items():
            p.set_metadata(k, v)
    return p


# ===========================================================================
# StateAcceptor
# ===========================================================================


class TestStateAcceptor:
    def test_accepts_done_program(self):
        assert StateAcceptor().is_accepted(_make_program(ProgramState.DONE))

    def test_rejects_discarded(self):
        assert not StateAcceptor().is_accepted(_make_program(ProgramState.DISCARDED))

    def test_rejects_queued(self):
        assert not StateAcceptor().is_accepted(_make_program(ProgramState.QUEUED))

    def test_rejects_running(self):
        assert not StateAcceptor().is_accepted(_make_program(ProgramState.RUNNING))


# ===========================================================================
# MetricsExistenceAcceptor
# ===========================================================================


class TestMetricsExistenceAcceptor:
    def test_accepts_with_metrics(self):
        p = _make_program(metrics={"fitness": 1.0})
        assert MetricsExistenceAcceptor().is_accepted(p)

    def test_rejects_empty_metrics(self):
        p = _make_program(metrics={})
        assert not MetricsExistenceAcceptor().is_accepted(p)

    def test_rejects_none_metrics(self):
        # Program.metrics defaults to {} (empty dict), never None due to Pydantic validation
        # But an empty dict still means "no metrics"
        p = _make_program(metrics={})
        assert not MetricsExistenceAcceptor().is_accepted(p)


# ===========================================================================
# ValidityMetricAcceptor
# ===========================================================================


class TestValidityMetricAcceptor:
    def test_accepts_valid_positive(self):
        p = _make_program(metrics={VALIDITY_KEY: 1.0})
        assert ValidityMetricAcceptor().is_accepted(p)

    def test_rejects_zero(self):
        """Zero validity means invalid — must NOT pass."""
        p = _make_program(metrics={VALIDITY_KEY: 0})
        assert not ValidityMetricAcceptor().is_accepted(p)

    def test_rejects_negative(self):
        """Negative values (e.g., sentinel -1e5) must be rejected."""
        p = _make_program(metrics={VALIDITY_KEY: -1e5})
        assert not ValidityMetricAcceptor().is_accepted(p)

    def test_rejects_missing_key(self):
        p = _make_program(metrics={"some_other_metric": 1.0})
        assert not ValidityMetricAcceptor().is_accepted(p)

    def test_custom_validity_key(self):
        p = _make_program(metrics={"custom_valid": 1.0})
        assert ValidityMetricAcceptor(validity_key="custom_valid").is_accepted(p)

    def test_small_positive_accepted(self):
        """Even a tiny positive value like 1e-10 should be accepted."""
        p = _make_program(metrics={VALIDITY_KEY: 1e-10})
        assert ValidityMetricAcceptor().is_accepted(p)

    def test_negative_one_rejected(self):
        """Regression: -1 must be rejected (not just sentinel -1e5)."""
        p = _make_program(metrics={VALIDITY_KEY: -1})
        assert not ValidityMetricAcceptor().is_accepted(p)


# ===========================================================================
# RequiredBehaviorKeysAcceptor
# ===========================================================================


class TestRequiredBehaviorKeysAcceptor:
    def test_accepts_all_keys_present(self):
        p = _make_program(metrics={"x": 1.0, "y": 2.0, "fitness": 3.0})
        assert RequiredBehaviorKeysAcceptor({"x", "y"}).is_accepted(p)

    def test_rejects_missing_key(self):
        p = _make_program(metrics={"x": 1.0})
        assert not RequiredBehaviorKeysAcceptor({"x", "y"}).is_accepted(p)

    def test_empty_required_set_always_accepts(self):
        p = _make_program(metrics={"x": 1.0})
        assert RequiredBehaviorKeysAcceptor(set()).is_accepted(p)

    def test_rejects_when_all_keys_missing(self):
        p = _make_program(metrics={"z": 1.0})
        assert not RequiredBehaviorKeysAcceptor({"x", "y"}).is_accepted(p)

    def test_accepts_superset_of_required(self):
        """Extra keys beyond required should not cause rejection."""
        p = _make_program(metrics={"x": 1.0, "y": 2.0, "z": 3.0, "w": 4.0})
        assert RequiredBehaviorKeysAcceptor({"x", "y"}).is_accepted(p)


# ===========================================================================
# MutationContextAcceptor
# ===========================================================================


class TestMutationContextAcceptor:
    def test_accepts_with_mutation_context(self):
        p = _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: {"prompt": "test"}})
        assert MutationContextAcceptor().is_accepted(p)

    def test_rejects_without_mutation_context(self):
        p = _make_program()
        assert not MutationContextAcceptor().is_accepted(p)


# ===========================================================================
# CompositeAcceptor
# ===========================================================================


class TestCompositeAcceptor:
    def test_empty_composite_accepts_all(self):
        """No sub-acceptors means no rejection criteria."""
        composite = CompositeAcceptor([])
        assert composite.is_accepted(_make_program())

    def test_short_circuits_on_first_rejection(self):
        """Should stop checking after first failure (AND semantics)."""
        call_order = []

        class TrackingAcceptor(StateAcceptor):
            def __init__(self, name, result):
                self._name = name
                self._result = result

            def is_accepted(self, program):
                call_order.append(self._name)
                return self._result

        composite = CompositeAcceptor(
            [
                TrackingAcceptor("first", False),
                TrackingAcceptor("second", True),
            ]
        )
        p = _make_program()
        assert not composite.is_accepted(p)
        assert call_order == ["first"], "Should short-circuit after first rejection"


# ===========================================================================
# DefaultProgramEvolutionAcceptor
# ===========================================================================


class TestDefaultProgramEvolutionAcceptor:
    def test_accepts_done_with_metrics(self):
        p = _make_program(ProgramState.DONE, metrics={"fitness": 1.0})
        assert DefaultProgramEvolutionAcceptor().is_accepted(p)

    def test_rejects_done_without_metrics(self):
        p = _make_program(ProgramState.DONE, metrics={})
        assert not DefaultProgramEvolutionAcceptor().is_accepted(p)

    def test_rejects_queued_with_metrics(self):
        p = _make_program(ProgramState.QUEUED, metrics={"fitness": 1.0})
        assert not DefaultProgramEvolutionAcceptor().is_accepted(p)


# ===========================================================================
# StandardEvolutionAcceptor
# ===========================================================================


class TestStandardEvolutionAcceptor:
    def _make_fully_valid_program(self) -> Program:
        return _make_program(
            state=ProgramState.DONE,
            metrics={VALIDITY_KEY: 1.0, "x": 0.5, "y": 0.5},
            metadata={MUTATION_CONTEXT_METADATA_KEY: {"prompt": "test"}},
        )

    def test_accepts_fully_valid_program(self):
        p = self._make_fully_valid_program()
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})
        assert acceptor.is_accepted(p)

    def test_rejects_invalid_validity(self):
        p = _make_program(
            state=ProgramState.DONE,
            metrics={VALIDITY_KEY: 0, "x": 0.5, "y": 0.5},
            metadata={MUTATION_CONTEXT_METADATA_KEY: {"prompt": "test"}},
        )
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})
        assert not acceptor.is_accepted(p)

    def test_rejects_missing_behavior_key(self):
        p = _make_program(
            state=ProgramState.DONE,
            metrics={VALIDITY_KEY: 1.0, "x": 0.5},
            metadata={MUTATION_CONTEXT_METADATA_KEY: {"prompt": "test"}},
        )
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})
        assert not acceptor.is_accepted(p)

    def test_rejects_missing_mutation_context(self):
        p = _make_program(
            state=ProgramState.DONE,
            metrics={VALIDITY_KEY: 1.0, "x": 0.5, "y": 0.5},
        )
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})
        assert not acceptor.is_accepted(p)

    def test_rejects_wrong_state(self):
        p = _make_program(
            state=ProgramState.RUNNING,
            metrics={VALIDITY_KEY: 1.0, "x": 0.5, "y": 0.5},
            metadata={MUTATION_CONTEXT_METADATA_KEY: {"prompt": "test"}},
        )
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})
        assert not acceptor.is_accepted(p)
