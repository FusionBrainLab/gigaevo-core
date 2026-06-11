"""Append-only audit trail for card-bank writes.

One ``WriteLedgerRecord`` per ingest verdict and per sweep eviction, persisted
as JSONL next to the bank
(``<checkpoint_dir>/write_ledger.jsonl``). The ledger answers "what happened to
the card I submitted, and why" for every gate verdict — the write-path
counterpart of the read path's per-program auction slate. Recording must never
block the write path: any failure is logged and swallowed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class WriteOutcome(StrEnum):
    """Verdict of one card-bank ingest, one member per ledger row outcome."""

    ADDED = "added"
    UPDATED = "updated"
    MERGED = "merged"
    DISCARDED = "discarded"
    REJECTED_HARM = "rejected_harm"
    EVICTED = "evicted"


class WriteLedgerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: str
    incoming_id: str
    final_id: str
    outcome: WriteOutcome
    reason: str = ""
    duplicate_of: str = ""
    merge_targets: list[str] = Field(default_factory=list)
    category: str = ""


class WriteLedger:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        *,
        incoming_id: str,
        final_id: str,
        outcome: WriteOutcome | str,
        reason: str = "",
        duplicate_of: str = "",
        merge_targets: list[str] | None = None,
        category: str = "",
    ) -> None:
        try:
            row = WriteLedgerRecord(
                timestamp_utc=datetime.now(UTC).isoformat(),
                incoming_id=incoming_id,
                final_id=final_id,
                outcome=WriteOutcome(outcome),
                reason=reason,
                duplicate_of=duplicate_of,
                merge_targets=merge_targets or [],
                category=category,
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(row.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[Memory][WriteLedger] failed to record: {}", exc)
