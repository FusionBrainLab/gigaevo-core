"""In-process card map with atomic JSON persistence."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
import fcntl
import json
import os
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import Card


def new_card_id() -> str:
    return f"mem-{uuid4().hex}"


class CardBankFileLock(AbstractContextManager["CardBankFileLock"]):
    """Advisory inter-process lock guarding one persisted file."""

    def __init__(self, path: str | Path, *, exclusive: bool = True) -> None:
        self._path = Path(path)
        self._exclusive = exclusive
        self._fh: BinaryIO | None = None

    def __enter__(self) -> CardBankFileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = self._path.open("a+b")
        self._fh = fh
        mode = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        fcntl.flock(fh.fileno(), mode)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class CardBank:
    """Authoritative card map, persisted as a single JSON file.

    :meth:`persist` writes atomically (tmp file + ``os.replace``) so a crash
    never leaves a torn bank. The store serializes mutations with
    :class:`CardBankFileLock` and reloads before each write, so multiple runs
    can share a bank directory without whole-snapshot last-writer loss.
    :meth:`reload` re-reads it from disk on demand (index rebuild/refresh).

    A missing file is a legitimate cold start; a file that exists but does
    not parse into cards raises :class:`MemoryStorageError` — silently
    starting empty would discard the bank on the next persist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = RLock()
        self._cards: dict[str, Card] = {}
        self._snapshot_cache: tuple[Card, ...] | None = None
        self._disk_token: tuple[int, int, int, int] | None = None
        if self._path.exists():
            self._reload()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def lock_path(self) -> Path:
        return self._path.with_name(f"{self._path.name}.lock")

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
            self._snapshot_cache = None

    def remove(self, card_id: str) -> bool:
        with self._lock:
            removed = self._cards.pop(card_id, None) is not None
            if removed:
                self._snapshot_cache = None
            return removed

    def snapshot(self) -> tuple[Card, ...]:
        with self._lock:
            if self._snapshot_cache is None:
                self._snapshot_cache = tuple(
                    sorted(self._cards.values(), key=lambda c: c.id)
                )
            return self._snapshot_cache

    def restore_snapshot(self, cards: Sequence[Card]) -> None:
        """Replace the in-memory map without persisting.

        Used by the owning store to roll back a failed durable write. The store is
        still the transaction boundary; CardBank keeps this primitive deliberately
        narrow so callers cannot accidentally persist stale state.
        """
        with self._lock:
            self._cards = {card.id: card for card in cards}
            self._snapshot_cache = None

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
                self._disk_token = self._stat_token()
            finally:
                tmp.unlink(missing_ok=True)

    def reload(self) -> None:
        """Re-read the bank from disk, discarding the in-memory map. A missing
        file leaves the current map untouched (nothing persisted yet)."""
        with self._lock:
            if not self._path.exists():
                return
            self._reload()

    def reload_if_changed(self) -> bool:
        """Reload and return True iff the durable bank advanced on disk."""
        with self._lock:
            token = self._stat_token()
            if token is None or token == self._disk_token:
                return False
            self._reload()
            return True

    def _reload(self) -> None:
        with self._lock:
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                cards = {}
                for cid, data in payload["cards"].items():
                    card = Card.model_validate(data)
                    if cid != card.id:
                        raise ValueError(
                            f"payload key {cid!r} != embedded card id {card.id!r}"
                        )
                    cards[cid] = card
            except Exception as exc:
                raise MemoryStorageError(
                    f"corrupt card bank at {self._path}: {exc}"
                ) from exc
            self._cards = cards
            self._snapshot_cache = None
            self._disk_token = self._stat_token()

    def _stat_token(self) -> tuple[int, int, int, int] | None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size, stat.st_dev, stat.st_ino)
