"""Canonical analytics event carrying the durable v2 decision projection."""

from __future__ import annotations

from typing import ClassVar

from gigaevo.memory.events import MemoryEvent
from gigaevo.memory_v2.models import DecisionRecord


class MemoryV2Decision(MemoryEvent):
    event: ClassVar[str] = "MEMORY_V2_DECISION"
    description: ClassVar[str] = (
        "A durable content-addressed memory-v2 policy action was committed."
    )
    health_question: ClassVar[str] = (
        "Are proposal, offer, safety, and posterior probabilities well calibrated?"
    )

    record: DecisionRecord


class MemoryV2WriterSync(MemoryEvent):
    event: ClassVar[str] = "MEMORY_V2_WRITER_SYNC"
    description: ClassVar[str] = (
        "The content-only writer synchronized leases with causal evidence."
    )
    health_question: ClassVar[str] = (
        "Does content generation expand the bank without observational efficacy credit?"
    )

    evidence_version: str
    model_evidence_version: str
    evidence_count: int
    pending_count: int
    bank_size: int
    released_child_count: int
