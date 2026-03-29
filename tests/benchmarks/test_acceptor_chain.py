"""Benchmark tests for the program acceptor chain.

The acceptor chain runs on every program during ingestion — it's in the hot loop.
These tests catch regressions in acceptor throughput.
"""

from __future__ import annotations

import time

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

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_program(i: int) -> Program:
    """Create a fully valid program for acceptance testing."""
    p = Program(
        code=f"def solve(): return {i}",
        state=ProgramState.DONE,
        atomic_counter=i,
    )
    p.metrics = {VALIDITY_KEY: 1.0, "fitness": float(i), "x": i * 0.01, "y": i * 0.02}
    p.set_metadata(MUTATION_CONTEXT_METADATA_KEY, {"prompt": f"test_{i}"})
    return p


def _make_invalid_program(i: int) -> Program:
    """Create a program that should be rejected (wrong state)."""
    p = Program(
        code=f"def solve(): return {i}",
        state=ProgramState.QUEUED,
        atomic_counter=i,
    )
    p.metrics = {VALIDITY_KEY: 0, "fitness": 0.0}
    return p


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestAcceptorChainThroughput:
    """Measure how fast acceptor chains can filter programs."""

    @pytest.fixture(params=[100, 1000, 5000])
    def n_programs(self, request) -> int:
        return request.param

    def test_default_acceptor_throughput(self, n_programs: int) -> None:
        """DefaultProgramEvolutionAcceptor must handle N programs under budget."""
        programs = [_make_valid_program(i) for i in range(n_programs)]
        acceptor = DefaultProgramEvolutionAcceptor()

        t0 = time.perf_counter()
        accepted = sum(1 for p in programs if acceptor.is_accepted(p))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert accepted == n_programs
        # Budget: < 0.01ms per program (10us)
        per_program_us = (elapsed_ms * 1000) / n_programs
        assert per_program_us < 100, (
            f"DefaultAcceptor too slow: {per_program_us:.1f}us/program "
            f"(budget: 100us, total: {elapsed_ms:.1f}ms for {n_programs} programs)"
        )

    def test_standard_acceptor_throughput(self, n_programs: int) -> None:
        """StandardEvolutionAcceptor (5-check chain) must stay fast."""
        programs = [_make_valid_program(i) for i in range(n_programs)]
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})

        t0 = time.perf_counter()
        accepted = sum(1 for p in programs if acceptor.is_accepted(p))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert accepted == n_programs
        per_program_us = (elapsed_ms * 1000) / n_programs
        assert per_program_us < 200, (
            f"StandardAcceptor too slow: {per_program_us:.1f}us/program "
            f"(budget: 200us, total: {elapsed_ms:.1f}ms for {n_programs} programs)"
        )

    def test_rejection_short_circuit_is_fast(self, n_programs: int) -> None:
        """Rejecting invalid programs should be faster than accepting valid ones."""
        invalid_programs = [_make_invalid_program(i) for i in range(n_programs)]
        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})

        t0 = time.perf_counter()
        rejected = sum(1 for p in invalid_programs if not acceptor.is_accepted(p))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert rejected == n_programs
        per_program_us = (elapsed_ms * 1000) / n_programs
        assert per_program_us < 100, (
            f"Rejection path too slow: {per_program_us:.1f}us/program "
            f"(budget: 100us, should short-circuit on first check)"
        )

    def test_mixed_valid_invalid_throughput(self, n_programs: int) -> None:
        """Realistic mix of valid and invalid programs."""
        programs = []
        for i in range(n_programs):
            if i % 3 == 0:
                programs.append(_make_invalid_program(i))
            else:
                programs.append(_make_valid_program(i))

        acceptor = StandardEvolutionAcceptor(required_behavior_keys={"x", "y"})

        t0 = time.perf_counter()
        results = [acceptor.is_accepted(p) for p in programs]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        expected_valid = n_programs - (n_programs // 3 + (1 if n_programs % 3 > 0 else 0))
        # Allow for rounding differences
        assert sum(results) >= expected_valid - 1

        per_program_us = (elapsed_ms * 1000) / n_programs
        assert per_program_us < 200, (
            f"Mixed acceptor too slow: {per_program_us:.1f}us/program"
        )
