"""Persistence seam for empirical-Bayes evicted-card evidence."""

from __future__ import annotations

from contextlib import AbstractContextManager
import fcntl
import json
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from loguru import logger

from gigaevo.memory.cards import Card


@runtime_checkable
class EvictedEvidenceSink(Protocol):
    def record(self, card: Card) -> None: ...


@runtime_checkable
class EvictedEvidenceSource(Protocol):
    def cards(self) -> tuple[Card, ...]: ...


class _JsonlFileLock(AbstractContextManager["_JsonlFileLock"]):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: BinaryIO | None = None

    def __enter__(self) -> _JsonlFileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = self._path.open("a+b")
        self._fh = fh
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            fh.close()
            self._fh = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class JsonlEvictedEvidence:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def record(self, card: Card) -> None:
        try:
            row = {
                "schema_version": "prior_evidence.v1",
                "card": card.model_dump(mode="json"),
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with _JsonlFileLock(self._path.with_suffix(self._path.suffix + ".lock")):
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.warning("[Memory][PriorEvidence] failed to record: {}", exc)

    def cards(self) -> tuple[Card, ...]:
        try:
            cards_by_id: dict[str, Card] = {}
            with self._path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        card = Card.model_validate(row["card"])
                    except Exception as exc:
                        logger.warning(
                            "[Memory][PriorEvidence] skipped malformed row: {}", exc
                        )
                        continue
                    cards_by_id[card.id] = card
            return tuple(cards_by_id.values())
        except FileNotFoundError:
            return ()
        except Exception as exc:
            logger.warning("[Memory][PriorEvidence] failed to read: {}", exc)
            return ()
