"""Topology filters for the migration bus.

Determines which migrant envelopes a run should accept based on the
source run and the local run's identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gigaevo.evolution.bus.transport import MigrantEnvelope


class Topology(ABC):
    """Abstract topology filter."""

    @abstractmethod
    def should_accept(self, envelope: MigrantEnvelope, local_run_id: str) -> bool: ...


class BusTopology(Topology):
    """Fully-connected bus: accept from any run except self."""

    def should_accept(self, envelope: MigrantEnvelope, local_run_id: str) -> bool:
        return envelope.source_run_id != local_run_id


class RingTopology(Topology):
    """Ring topology: accept only from the predecessor in the ring.

    run_ids defines the ring order. Each run accepts migrants only from
    the run immediately before it in the list (wrapping around).
    """

    def __init__(self, run_ids: list[str]):
        if len(run_ids) < 2:
            raise ValueError("RingTopology requires at least 2 run_ids")
        self._run_ids = run_ids
        self._predecessor: dict[str, str] = {}
        for i, rid in enumerate(run_ids):
            self._predecessor[rid] = run_ids[i - 1]

    def should_accept(self, envelope: MigrantEnvelope, local_run_id: str) -> bool:
        if local_run_id not in self._predecessor:
            return False
        return envelope.source_run_id == self._predecessor[local_run_id]
