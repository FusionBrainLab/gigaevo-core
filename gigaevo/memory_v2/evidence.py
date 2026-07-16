"""Small evidence interfaces used by inference tests and alternative ledgers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gigaevo.memory_v2.models import (
    CausalObservation,
    EvidenceSnapshot,
    canonical_digest,
)


@runtime_checkable
class EvidenceRepository(Protocol):
    def snapshot(self) -> EvidenceSnapshot: ...


class StaticEvidenceRepository:
    def __init__(self, observations: Sequence[CausalObservation] = ()) -> None:
        ordered = tuple(
            sorted(observations, key=lambda row: (row.event_ordinal, row.decision_id))
        )
        self._snapshot = EvidenceSnapshot(
            version=canonical_digest([row.model_dump(mode="json") for row in ordered]),
            model_version=canonical_digest(
                [row.model_dump(mode="json") for row in ordered]
            ),
            observations=ordered,
            reward_observations=ordered,
        )

    def snapshot(self) -> EvidenceSnapshot:
        return self._snapshot
