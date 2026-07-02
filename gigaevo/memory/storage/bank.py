"""In-process card map with atomic JSON persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import Card


def new_card_id() -> str:
    return f"mem-{uuid4().hex[:12]}"


class CardBank:
    """Authoritative card map, persisted as a single JSON file.

    :meth:`persist` writes atomically (tmp file + ``os.replace``) so a crash
    never leaves a torn bank. A stat watermark — (mtime_ns, size), since
    mtime alone is too coarse on NFS — makes external writers visible:
    :meth:`refresh_if_stale` reloads only when the file on disk differs from
    the last version this process wrote or read.

    A missing file is a legitimate cold start; a file that exists but does
    not parse into cards raises :class:`MemoryStorageError` — silently
    starting empty would discard the bank on the next persist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = RLock()
        self._cards: dict[str, Card] = {}
        self._watermark: tuple[int, int] = (0, 0)
        if self._path.exists():
            self._reload()

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        with self._lock:
            return len(self._cards)

    def __contains__(self, card_id: str) -> bool:
        with self._lock:
            return card_id in self._cards

    def get(self, card_id: str) -> Card | None:
        with self._lock:
            return self._cards.get(card_id)

    def put(self, card: Card) -> None:
        if not card.id:
            raise ValueError("cannot store a card with an empty id")
        with self._lock:
            self._cards[card.id] = card

    def remove(self, card_id: str) -> bool:
        with self._lock:
            return self._cards.pop(card_id, None) is not None

    def snapshot(self) -> tuple[Card, ...]:
        with self._lock:
            return tuple(sorted(self._cards.values(), key=lambda c: c.id))

    def persist(self) -> None:
        with self._lock:
            payload = {
                "cards": {
                    cid: card.model_dump(mode="json")
                    for cid, card in sorted(self._cards.items())
                }
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.parent / f"{self._path.name}.{os.getpid()}.tmp"
            try:
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp, self._path)
            finally:
                tmp.unlink(missing_ok=True)
            self._watermark = self._stat_signature()

    def refresh_if_stale(self) -> bool:
        with self._lock:
            if not self._path.exists():
                return False
            if self._stat_signature() == self._watermark:
                return False
            self._reload()
            return True

    def _stat_signature(self) -> tuple[int, int]:
        stat = self._path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def _reload(self) -> None:
        with self._lock:
            signature = self._stat_signature()
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                cards = {
                    cid: Card.model_validate(data)
                    for cid, data in payload["cards"].items()
                }
            except Exception as exc:
                raise MemoryStorageError(
                    f"corrupt card bank at {self._path}: {exc}"
                ) from exc
            self._cards = cards
            self._watermark = signature
