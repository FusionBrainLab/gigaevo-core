"""Append-only JSONL event log — the sole raw record the paper is built from.

Two properties matter and both are enforced here rather than by convention:

Provenance is mandatory. Every event carries the git commit, the protocol-lock
hash and the candidate-source hash, because an event that cannot be traced back
to the exact code and protocol that produced it cannot be used as evidence.
Provenance is a required constructor argument, so it cannot be forgotten.

The log survives a hard kill. Each event is one write + flush + fsync, and the
reader tolerates a truncated final line — a run that is SIGKILLed mid-write must
still yield every event that was completed before it died.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from problems._harness.common.contracts import ProposalResult


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    protocol_lock_sha256: str
    code_sha256: str


class EventLogger:
    def __init__(
        self,
        path: Path | str,
        provenance: Provenance,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._provenance = provenance
        self._clock = clock
        self._handle = open(self._path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "event_type": event_type,
            "timestamp": self._clock(),
            **asdict(self._provenance),
            **fields,
        }
        self._handle.write(json.dumps(event, default=str) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return event

    def emit_proposal(self, result: ProposalResult, **fields: Any) -> dict[str, Any]:
        """Exactly one terminal event per proposal, whatever the outcome."""
        return self.emit(
            "proposal",
            label=result.label,
            status=str(result.status),
            elapsed_s=result.elapsed_s,
            error_type=result.error_type,
            error_message=result.error_message,
            stdout=result.stdout,
            stderr=result.stderr,
            **fields,
        )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_events(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield every complete event, ignoring a partial line left by a hard kill."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                return
            if line.strip():
                yield json.loads(line)
