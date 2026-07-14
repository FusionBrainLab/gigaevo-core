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
    def __init__(self, path: Path | str, max_cards: int = 10_000) -> None:
        if max_cards <= 0:
            raise ValueError("max_cards must be positive")
        self._path = Path(path)
        self._max_cards = max_cards

    @staticmethod
    def _row(card: Card) -> dict:
        return {
            "schema_version": "prior_evidence.v1",
            "card": card.model_dump(mode="json"),
        }

    @staticmethod
    def _parse_card(line: str) -> Card | None:
        try:
            row = json.loads(line)
            if row.get("schema_version") != "prior_evidence.v1":
                raise ValueError("unsupported prior-evidence schema")
            return Card.model_validate(row["card"])
        except Exception as exc:
            logger.warning("[Memory][PriorEvidence] skipped malformed row: {}", exc)
            return None

    def record(self, card: Card) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with _JsonlFileLock(self._path.with_suffix(self._path.suffix + ".lock")):
                cards_by_id: dict[str, Card] = {}
                try:
                    lines = self._path.read_text(encoding="utf-8").splitlines()
                except FileNotFoundError:
                    lines = []
                for line in lines:
                    existing = self._parse_card(line)
                    if existing is not None:
                        cards_by_id.pop(existing.id, None)
                        cards_by_id[existing.id] = existing
                cards_by_id.pop(card.id, None)
                cards_by_id[card.id] = card
                bounded = list(cards_by_id.values())[-self._max_cards :]
                temporary = self._path.with_suffix(self._path.suffix + ".tmp")
                with temporary.open("w", encoding="utf-8") as f:
                    for retained in bounded:
                        f.write(json.dumps(self._row(retained)) + "\n")
                temporary.replace(self._path)
        except Exception as exc:
            logger.warning("[Memory][PriorEvidence] failed to record: {}", exc)

    def cards(self) -> tuple[Card, ...]:
        try:
            cards_by_id: dict[str, Card] = {}
            with self._path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        card = self._parse_card(line)
                    except Exception:
                        continue
                    if card is None:
                        continue
                    cards_by_id.pop(card.id, None)
                    cards_by_id[card.id] = card
            return tuple(cards_by_id.values())
        except FileNotFoundError:
            return ()
        except Exception as exc:
            logger.warning("[Memory][PriorEvidence] failed to read: {}", exc)
            return ()
