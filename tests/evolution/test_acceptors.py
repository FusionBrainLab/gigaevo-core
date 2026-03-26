"""Tests for gigaevo/evolution/engine/acceptor.py"""

from gigaevo.evolution.engine.acceptor import (
    CompositeAcceptor,
    DefaultProgramEvolutionAcceptor,
    MetricsExistenceAcceptor,
    MutationContextAcceptor,
    MutationContextAndBehaviorKeysAcceptor,
    RequiredBehaviorKeysAcceptor,
    StandardEvolutionAcceptor,
    StateAcceptor,
    ValidityMetricAcceptor,
)
from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _prog(
    state=ProgramState.DONE,
    metrics=None,
    metadata=None,
):
    p = Program(code="def solve(): return 1", state=state)
    if metrics:
        p.add_metrics(metrics)
    if metadata:
        p.metadata = metadata
    return p


class TestStateAcceptor:
    def test_done_accepted(self):
        assert StateAcceptor().is_accepted(_prog(state=ProgramState.DONE))

    def test_discarded_rejected(self):
        assert not StateAcceptor().is_accepted(_prog(state=ProgramState.DISCARDED))

    def test_queued_rejected(self):
        assert not StateAcceptor().is_accepted(_prog(state=ProgramState.QUEUED))

    def test_running_rejected(self):
        assert not StateAcceptor().is_accepted(_prog(state=ProgramState.RUNNING))


class TestMetricsExistenceAcceptor:
    def test_with_metrics(self):
        p = _prog(metrics={"score": 1.0})
        assert MetricsExistenceAcceptor().is_accepted(p)

    def test_without_metrics(self):
        p = _prog()
        assert not MetricsExistenceAcceptor().is_accepted(p)

    def test_empty_metrics(self):
        p = _prog()
        p.metrics = {}
        assert not MetricsExistenceAcceptor().is_accepted(p)


class TestValidityMetricAcceptor:
    def test_valid(self):
        p = _prog(metrics={VALIDITY_KEY: 1.0})
        assert ValidityMetricAcceptor().is_accepted(p)

    def test_invalid(self):
        p = _prog(metrics={VALIDITY_KEY: 0.0})
        assert not ValidityMetricAcceptor().is_accepted(p)

    def test_missing_key(self):
        p = _prog(metrics={"other": 1.0})
        assert not ValidityMetricAcceptor().is_accepted(p)

    def test_custom_key(self):
        p = _prog(metrics={"custom_valid": 1.0})
        acc = ValidityMetricAcceptor(validity_key="custom_valid")
        assert acc.is_accepted(p)


class TestRequiredBehaviorKeysAcceptor:
    def test_all_present(self):
        p = _prog(metrics={"a": 1.0, "b": 2.0})
        acc = RequiredBehaviorKeysAcceptor({"a", "b"})
        assert acc.is_accepted(p)

    def test_missing_key(self):
        p = _prog(metrics={"a": 1.0})
        acc = RequiredBehaviorKeysAcceptor({"a", "b"})
        assert not acc.is_accepted(p)

    def test_empty_required_set(self):
        p = _prog(metrics={"a": 1.0})
        acc = RequiredBehaviorKeysAcceptor(set())
        assert acc.is_accepted(p)

    def test_extra_keys_ok(self):
        p = _prog(metrics={"a": 1.0, "b": 2.0, "c": 3.0})
        acc = RequiredBehaviorKeysAcceptor({"a"})
        assert acc.is_accepted(p)


class TestMutationContextAcceptor:
    def test_with_context(self):
        p = _prog(metadata={MUTATION_CONTEXT_METADATA_KEY: "some_context"})
        assert MutationContextAcceptor().is_accepted(p)

    def test_without_context(self):
        p = _prog()
        assert not MutationContextAcceptor().is_accepted(p)


class TestCompositeAcceptor:
    def test_all_pass(self):
        p = _prog(metrics={"score": 1.0})
        acc = CompositeAcceptor([StateAcceptor(), MetricsExistenceAcceptor()])
        assert acc.is_accepted(p)

    def test_first_fails(self):
        p = _prog(state=ProgramState.RUNNING, metrics={"score": 1.0})
        acc = CompositeAcceptor([StateAcceptor(), MetricsExistenceAcceptor()])
        assert not acc.is_accepted(p)

    def test_last_fails(self):
        p = _prog(state=ProgramState.DONE)
        acc = CompositeAcceptor([StateAcceptor(), MetricsExistenceAcceptor()])
        assert not acc.is_accepted(p)

    def test_empty_list_accepts(self):
        p = _prog()
        acc = CompositeAcceptor([])
        assert acc.is_accepted(p)


class TestDefaultProgramEvolutionAcceptor:
    def test_done_with_metrics(self):
        p = _prog(metrics={"score": 1.0})
        assert DefaultProgramEvolutionAcceptor().is_accepted(p)

    def test_done_no_metrics(self):
        p = _prog()
        assert not DefaultProgramEvolutionAcceptor().is_accepted(p)

    def test_running_with_metrics(self):
        p = _prog(state=ProgramState.RUNNING, metrics={"score": 1.0})
        assert not DefaultProgramEvolutionAcceptor().is_accepted(p)


class TestStandardEvolutionAcceptor:
    def _fully_valid_prog(self):
        return _prog(
            metrics={
                VALIDITY_KEY: 1.0,
                "behavior_a": 0.5,
            },
            metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"},
        )

    def test_fully_valid(self):
        p = self._fully_valid_prog()
        acc = StandardEvolutionAcceptor(
            required_behavior_keys={"behavior_a"},
        )
        assert acc.is_accepted(p)

    def test_missing_behavior_key(self):
        p = self._fully_valid_prog()
        acc = StandardEvolutionAcceptor(
            required_behavior_keys={"behavior_a", "behavior_b"},
        )
        assert not acc.is_accepted(p)

    def test_invalid_validity(self):
        p = _prog(
            metrics={
                VALIDITY_KEY: 0.0,
                "behavior_a": 0.5,
            },
            metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"},
        )
        acc = StandardEvolutionAcceptor(required_behavior_keys={"behavior_a"})
        assert not acc.is_accepted(p)

    def test_no_mutation_context(self):
        p = _prog(
            metrics={
                VALIDITY_KEY: 1.0,
                "behavior_a": 0.5,
            },
        )
        acc = StandardEvolutionAcceptor(required_behavior_keys={"behavior_a"})
        assert not acc.is_accepted(p)


class TestMutationContextAndBehaviorKeysAcceptor:
    def test_fully_valid(self):
        """DONE + metrics + behavior keys + mutation context -> accepted."""
        p = _prog(
            metrics={"behavior_a": 0.5, "score": 1.0},
            metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"},
        )
        acc = MutationContextAndBehaviorKeysAcceptor(
            required_behavior_keys={"behavior_a"},
        )
        assert acc.is_accepted(p)

    def test_missing_behavior_key(self):
        """Missing required behavior key -> rejected."""
        p = _prog(
            metrics={"score": 1.0},
            metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"},
        )
        acc = MutationContextAndBehaviorKeysAcceptor(
            required_behavior_keys={"behavior_a"},
        )
        assert not acc.is_accepted(p)

    def test_missing_mutation_context(self):
        """Missing mutation context -> rejected."""
        p = _prog(
            metrics={"behavior_a": 0.5, "score": 1.0},
        )
        acc = MutationContextAndBehaviorKeysAcceptor(
            required_behavior_keys={"behavior_a"},
        )
        assert not acc.is_accepted(p)

    def test_no_validity_check(self):
        """Unlike StandardEvolutionAcceptor, this one does NOT check validity."""
        p = _prog(
            metrics={
                VALIDITY_KEY: 0.0,  # invalid!
                "behavior_a": 0.5,
            },
            metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"},
        )
        acc = MutationContextAndBehaviorKeysAcceptor(
            required_behavior_keys={"behavior_a"},
        )
        # Should be accepted because there's no ValidityMetricAcceptor
        assert acc.is_accepted(p)
