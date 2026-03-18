"""Tests for BusTopology and RingTopology filters."""

from __future__ import annotations

import pytest

from gigaevo.evolution.bus.topology import BusTopology, RingTopology
from gigaevo.evolution.bus.transport import MigrantEnvelope


def _envelope(source_run_id: str) -> MigrantEnvelope:
    return MigrantEnvelope(
        source_run_id=source_run_id,
        program_id="prog-1",
        program_data={"code": "x", "id": "prog-1"},
        published_at=0.0,
        generation=1,
    )


# ---------------------------------------------------------------------------
# BusTopology
# ---------------------------------------------------------------------------


class TestBusTopology:
    def test_accept_from_other(self) -> None:
        topo = BusTopology()
        assert topo.should_accept(_envelope("run@db0"), "run@db1") is True

    def test_reject_from_self(self) -> None:
        topo = BusTopology()
        assert topo.should_accept(_envelope("run@db0"), "run@db0") is False

    def test_accept_multiple_sources(self) -> None:
        topo = BusTopology()
        for src in ["run@db0", "run@db2", "run@db3"]:
            assert topo.should_accept(_envelope(src), "run@db1") is True


# ---------------------------------------------------------------------------
# RingTopology
# ---------------------------------------------------------------------------


class TestRingTopology:
    def test_accept_predecessor(self) -> None:
        topo = RingTopology(run_ids=["A", "B", "C"])
        # B's predecessor is A
        assert topo.should_accept(_envelope("A"), "B") is True

    def test_reject_non_predecessor(self) -> None:
        topo = RingTopology(run_ids=["A", "B", "C"])
        # B's predecessor is A, not C
        assert topo.should_accept(_envelope("C"), "B") is False

    def test_ring_wraps_around(self) -> None:
        topo = RingTopology(run_ids=["A", "B", "C"])
        # A's predecessor is C (wrap-around)
        assert topo.should_accept(_envelope("C"), "A") is True

    def test_reject_self(self) -> None:
        topo = RingTopology(run_ids=["A", "B", "C"])
        assert topo.should_accept(_envelope("A"), "A") is False

    def test_unknown_local_run_rejected(self) -> None:
        topo = RingTopology(run_ids=["A", "B", "C"])
        assert topo.should_accept(_envelope("A"), "X") is False

    def test_two_node_ring(self) -> None:
        topo = RingTopology(run_ids=["A", "B"])
        assert topo.should_accept(_envelope("A"), "B") is True
        assert topo.should_accept(_envelope("B"), "A") is True
        assert topo.should_accept(_envelope("A"), "A") is False

    def test_single_node_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            RingTopology(run_ids=["A"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            RingTopology(run_ids=[])
