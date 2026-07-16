"""Selection leases protecting selected cards until outcome crediting."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
import socket
from threading import RLock
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from loguru import logger

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.storage.bank import CardBankFileLock

if TYPE_CHECKING:
    from gigaevo.memory.cards import Card
    from gigaevo.memory.storage.base import MemoryStore


def _card_ids(card_ids: Iterable[str]) -> set[str]:
    return {card_id.strip() for card_id in card_ids if card_id.strip()}


def _selection_token(counts: dict[str, int]) -> str:
    return json.dumps(sorted(counts.items()), ensure_ascii=False, separators=(",", ":"))


def _read_pid_start(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    comm_end = stat.rfind(")")
    if comm_end < 0:
        return None
    fields = stat[comm_end + 1 :].split()
    try:
        pid_start = int(fields[19])
    except (IndexError, ValueError):
        return None
    return pid_start if pid_start >= 0 else None


@dataclass(frozen=True)
class PendingSelectionSnapshot:
    counts: dict[str, int]
    version: str


@dataclass(frozen=True)
class SelectionReservation:
    committed: bool
    card_ids: tuple[str, ...] = ()


class SelectionLease:
    """Attempt-scoped handle whose ownership can move to a persisted child."""

    def __init__(self, registry: InFlightSelectionRegistry, attempt_id: str) -> None:
        self._registry = registry
        self.attempt_id = attempt_id

    def attach_cards(self, card_ids: Iterable[str]) -> tuple[str, ...]:
        return self._registry.attach_cards(self.attempt_id, card_ids)

    def reverify_cards(self, card_ids: Iterable[str]) -> tuple[str, ...]:
        return self._registry.reverify_cards(self.attempt_id, card_ids)

    def transfer_to_child(self, child_id: str, retain_ids: Iterable[str]) -> None:
        self._registry.transfer_to_child(self.attempt_id, child_id, retain_ids)

    def release(self) -> None:
        self._registry.release_attempt(self.attempt_id)


class InFlightSelectionRegistry:
    """Refcounted attempt/child ownership graph guarded across writer threads."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._attempt_cards: dict[str, set[str]] = {}
        self._attempt_parents: dict[str, set[str]] = {}
        self._parent_attempts: dict[str, set[str]] = {}
        self._child_cards: dict[str, set[str]] = {}
        self._card_owner_count: dict[str, int] = {}
        self._card_lookup: Callable[[str], Card | None] | None = None
        self._active_attempt_by_parent: dict[str, str] = {}
        self._revision = 0

    @contextmanager
    def eviction_guard(self):
        """Serialize lease acquisition with every guarded card deletion."""
        with self._lock:
            yield

    def bind_store(self, store: MemoryStore) -> None:
        """Register the store used for guarded coalesced-fresh revalidation."""
        with self._lock:
            self._card_lookup = store.get

    @contextmanager
    def activate_attempt(self, attempt_id: str, parent_ids: Iterable[str]):
        normalized = {
            parent_id.strip() for parent_id in parent_ids if parent_id.strip()
        }
        with self._lock:
            if attempt_id not in self._attempt_cards:
                raise MemoryStorageError(f"unknown selection attempt {attempt_id!r}")
            missing = normalized - self._attempt_parents.get(attempt_id, set())
            if missing:
                raise MemoryStorageError(
                    f"selection attempt {attempt_id!r} does not own parents "
                    f"{sorted(missing)}"
                )
            conflicts = {
                parent_id: active
                for parent_id in normalized
                if (active := self._active_attempt_by_parent.get(parent_id))
                not in (None, attempt_id)
            }
            if conflicts:
                raise MemoryStorageError(
                    f"selection attempt activation conflicts: {conflicts}"
                )
            for parent_id in normalized:
                self._active_attempt_by_parent[parent_id] = attempt_id
        try:
            yield
        finally:
            with self._lock:
                for parent_id in normalized:
                    if self._active_attempt_by_parent.get(parent_id) == attempt_id:
                        self._active_attempt_by_parent.pop(parent_id, None)

    def attempt_for_parent(self, parent_id: str) -> str | None:
        parent_id = parent_id.strip()
        with self._lock:
            active = self._active_attempt_by_parent.get(parent_id)
            if active is not None:
                return active
            attempts = self._parent_attempts.get(parent_id, set())
            if len(attempts) == 1:
                return next(iter(attempts))
            return None

    def active_attempt_for_parent(self, parent_id: str) -> str | None:
        """Return only an attempt currently scoped around a parent refresh."""

        with self._lock:
            return self._active_attempt_by_parent.get(parent_id.strip())

    def attempts_for_parent(self, parent_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._parent_attempts.get(parent_id.strip(), ())))

    def open_attempt(self, attempt_id: str, parent_id: str) -> SelectionLease:
        attempt_id = attempt_id.strip()
        parent_id = parent_id.strip()
        with self._lock:
            self._attempt_cards.setdefault(attempt_id, set())
            parents = self._attempt_parents.setdefault(attempt_id, set())
            if parent_id and parent_id not in parents:
                parents.add(parent_id)
                self._parent_attempts.setdefault(parent_id, set()).add(attempt_id)
        return SelectionLease(self, attempt_id)

    def attach_cards(self, attempt_id: str, card_ids: Iterable[str]) -> tuple[str, ...]:
        normalized = _card_ids(card_ids)
        with self._lock:
            self._attach_locked(attempt_id, normalized)
        return tuple(sorted(normalized))

    def attach_cards_for_parent(self, parent_id: str, card_ids: Iterable[str]) -> None:
        normalized = _card_ids(card_ids)
        with self._lock:
            for attempt_id in tuple(self._parent_attempts.get(parent_id, ())):
                self._attach_locked(attempt_id, normalized)

    def reverify_cards(
        self, attempt_id: str, card_ids: Iterable[str]
    ) -> tuple[str, ...]:
        ordered = tuple(card_ids)
        normalized = _card_ids(ordered)
        with self._lock:
            if self._card_lookup is None:
                return tuple(cid for cid in ordered if cid.strip() in normalized)
            existing = {cid for cid in normalized if self._card_lookup(cid) is not None}
            self._attach_locked(attempt_id, existing)
        return tuple(cid for cid in ordered if cid.strip() in existing)

    def transfer_to_child(
        self, attempt_id: str, child_id: str, retain_ids: Iterable[str]
    ) -> None:
        child_id = child_id.strip()
        retained = _card_ids(retain_ids)
        with self._lock:
            attempt_cards = self._attempt_cards.get(attempt_id, set())
            retained.intersection_update(attempt_cards)
            child_cards = self._child_cards.setdefault(child_id, set())
            for card_id in retained - child_cards:
                child_cards.add(card_id)
                self._increment_locked(card_id)
            self._release_attempt_locked(attempt_id)

    def release_attempt(self, attempt_id: str) -> None:
        with self._lock:
            self._release_attempt_locked(attempt_id)

    def release_child(self, child_id: str) -> None:
        with self._lock:
            for card_id in self._child_cards.pop(child_id, set()):
                self._decrement_locked(card_id)

    def abandon(self, owner_ids: Iterable[str]) -> None:
        with self._lock:
            for owner_id in set(owner_ids):
                self._release_attempt_locked(owner_id)
                for card_id in self._child_cards.pop(owner_id, set()):
                    self._decrement_locked(card_id)

    def leased_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._card_owner_count)

    def pending_counts(self) -> dict[str, int]:
        """Snapshot uncredited in-flight exposure counts by card id."""
        return self.selection_snapshot().counts

    def selection_snapshot(self) -> PendingSelectionSnapshot:
        with self._lock:
            return PendingSelectionSnapshot(
                counts=dict(self._card_owner_count),
                version=_selection_token(self._card_owner_count),
            )

    def reserve_selection(
        self,
        attempt_id: str,
        card_ids: Iterable[str],
        *,
        expected_version: str | None,
        card_lookup: Callable[[str], Card | None],
    ) -> SelectionReservation:
        ordered = tuple(dict.fromkeys(cid.strip() for cid in card_ids if cid.strip()))
        with self._lock:
            if (
                expected_version is not None
                and _selection_token(self._card_owner_count) != expected_version
            ):
                return SelectionReservation(committed=False)
            if attempt_id not in self._attempt_cards:
                raise MemoryStorageError(f"unknown selection attempt {attempt_id!r}")
            attached = self._attach_locked(attempt_id, set(ordered))
            try:
                kept = tuple(cid for cid in ordered if card_lookup(cid) is not None)
            except BaseException:
                self._rollback_attach_locked(attempt_id, attached)
                raise
            vanished = attached - set(kept)
            if vanished:
                self._rollback_attach_locked(attempt_id, vanished)
            return SelectionReservation(committed=True, card_ids=kept)

    def is_leased(self, card_id: str) -> bool:
        with self._lock:
            return self._card_owner_count.get(card_id.strip(), 0) > 0

    def _attach_locked(self, attempt_id: str, card_ids: set[str]) -> set[str]:
        owned = self._attempt_cards.get(attempt_id)
        if owned is None:
            return set()
        attached = card_ids - owned
        for card_id in attached:
            owned.add(card_id)
            self._increment_locked(card_id)
        return attached

    def _rollback_attach_locked(self, attempt_id: str, card_ids: set[str]) -> None:
        owned = self._attempt_cards.get(attempt_id)
        if owned is None:
            return
        for card_id in card_ids:
            owned.remove(card_id)
            self._decrement_locked(card_id)

    def _release_attempt_locked(self, attempt_id: str) -> None:
        for card_id in self._attempt_cards.pop(attempt_id, set()):
            self._decrement_locked(card_id)
        for parent_id in self._attempt_parents.pop(attempt_id, set()):
            attempts = self._parent_attempts.get(parent_id)
            if attempts is None:
                continue
            attempts.discard(attempt_id)
            if not attempts:
                self._parent_attempts.pop(parent_id, None)

    def _increment_locked(self, card_id: str) -> None:
        self._card_owner_count[card_id] = self._card_owner_count.get(card_id, 0) + 1
        self._revision += 1

    def _decrement_locked(self, card_id: str) -> None:
        remaining = self._card_owner_count[card_id] - 1
        if remaining:
            self._card_owner_count[card_id] = remaining
        else:
            self._card_owner_count.pop(card_id)
        self._revision += 1


class SharedSelectionRegistry(InFlightSelectionRegistry):
    """Durable cross-process leases over one shared card bank.

    ``ttl_seconds`` is an operational crash-expiry fallback for foreign-host
    owners; same-host ownership uses exact process-liveness probing.
    """

    def __init__(self, path: str | Path, ttl_seconds: float = 7200.0) -> None:
        super().__init__()
        ttl_seconds = float(ttl_seconds)
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        self._path = Path(path)
        self._lock_path = self._path.with_name(f"{self._path.name}.lock")
        self._ttl_seconds = ttl_seconds
        self._host = socket.gethostname()
        self._pid = os.getpid()
        self._pid_start = _read_pid_start(self._pid) or 0
        self._owner_key = f"{self._host}:{self._pid}:{uuid4().hex[:8]}"

    def attach_cards(self, attempt_id: str, card_ids: Iterable[str]) -> tuple[str, ...]:
        normalized = _card_ids(card_ids)
        with self._lock:
            attached = self._attach_locked(attempt_id, normalized)
            self._publish_acquisition_locked({attempt_id: attached})
        return tuple(sorted(normalized))

    def attach_cards_for_parent(self, parent_id: str, card_ids: Iterable[str]) -> None:
        normalized = _card_ids(card_ids)
        with self._lock:
            attached_by_attempt: dict[str, set[str]] = {}
            for attempt_id in tuple(self._parent_attempts.get(parent_id, ())):
                attached_by_attempt[attempt_id] = self._attach_locked(
                    attempt_id, normalized
                )
            self._publish_acquisition_locked(attached_by_attempt)

    def reverify_cards(
        self, attempt_id: str, card_ids: Iterable[str]
    ) -> tuple[str, ...]:
        ordered = tuple(card_ids)
        normalized = _card_ids(ordered)
        with self._lock:
            attached = self._attach_locked(attempt_id, normalized)
            self._publish_acquisition_locked({attempt_id: attached})
            if self._card_lookup is None:
                return tuple(cid for cid in ordered if cid.strip() in normalized)
            try:
                existing = {
                    cid for cid in normalized if self._card_lookup(cid) is not None
                }
            except BaseException:
                self._rollback_attach_locked(attempt_id, attached)
                self._sync_best_effort_locked()
                raise
            vanished = normalized - existing
            if vanished:
                owned_vanished = vanished & self._attempt_cards.get(attempt_id, set())
                self._rollback_attach_locked(attempt_id, owned_vanished)
                self._sync_best_effort_locked()
        return tuple(cid for cid in ordered if cid.strip() in existing)

    def transfer_to_child(
        self, attempt_id: str, child_id: str, retain_ids: Iterable[str]
    ) -> None:
        child_id = child_id.strip()
        retained = _card_ids(retain_ids)
        with self._lock:
            attempt_cards = self._attempt_cards.get(attempt_id, set())
            retained.intersection_update(attempt_cards)
            child_cards = self._child_cards.setdefault(child_id, set())
            for card_id in retained - child_cards:
                child_cards.add(card_id)
                self._increment_locked(card_id)
            self._release_attempt_locked(attempt_id)
            self._sync_best_effort_locked()

    def release_attempt(self, attempt_id: str) -> None:
        with self._lock:
            self._release_attempt_locked(attempt_id)
            self._sync_best_effort_locked()

    def release_child(self, child_id: str) -> None:
        with self._lock:
            for card_id in self._child_cards.pop(child_id, set()):
                self._decrement_locked(card_id)
            self._sync_best_effort_locked()

    def abandon(self, owner_ids: Iterable[str]) -> None:
        with self._lock:
            for owner_id in set(owner_ids):
                self._release_attempt_locked(owner_id)
                for card_id in self._child_cards.pop(owner_id, set()):
                    self._decrement_locked(card_id)
            self._sync_best_effort_locked()

    def leased_ids(self) -> frozenset[str]:
        with self._lock:
            own_ids = frozenset(self._card_owner_count)
            try:
                foreign_ids = self._read_live_foreign_ids_locked()
            except Exception as exc:
                self._warn_read_failure(exc)
                return own_ids
            return own_ids | foreign_ids

    def is_leased(self, card_id: str) -> bool:
        card_id = card_id.strip()
        with self._lock:
            own_lease = self._card_owner_count.get(card_id, 0) > 0
            try:
                foreign_ids = self._read_live_foreign_ids_locked()
            except Exception as exc:
                self._warn_read_failure(exc)
                return True
            return own_lease or card_id in foreign_ids

    def selection_snapshot(self) -> PendingSelectionSnapshot:
        with self._lock:
            try:
                with CardBankFileLock(self._lock_path, exclusive=False):
                    owners = self._live_owners_unlocked(datetime.now(UTC))
            except Exception as exc:
                self._warn_read_failure(exc)
                raise MemoryStorageError(
                    f"failed to snapshot selection lease sidecar at {self._path}"
                ) from exc
            counts = self._pending_counts_locked(owners)
            return PendingSelectionSnapshot(
                counts=counts, version=self._selection_token_locked(owners)
            )

    def reserve_selection(
        self,
        attempt_id: str,
        card_ids: Iterable[str],
        *,
        expected_version: str | None,
        card_lookup: Callable[[str], Card | None],
    ) -> SelectionReservation:
        ordered = tuple(dict.fromkeys(cid.strip() for cid in card_ids if cid.strip()))
        with self._lock:
            if attempt_id not in self._attempt_cards:
                raise MemoryStorageError(f"unknown selection attempt {attempt_id!r}")
            attached: set[str] = set()
            try:
                with CardBankFileLock(self._lock_path, exclusive=True):
                    now = datetime.now(UTC)
                    owners = self._live_owners_unlocked(now)
                    if (
                        expected_version is not None
                        and self._selection_token_locked(owners) != expected_version
                    ):
                        return SelectionReservation(committed=False)
                    attached = self._attach_locked(attempt_id, set(ordered))
                    self._replace_own_owner_locked(owners, now)
                    self._write_owners_unlocked(owners)
            except BaseException as exc:
                if attached:
                    self._rollback_attach_locked(attempt_id, attached)
                if not isinstance(exc, Exception):
                    raise
                logger.warning(
                    "[Memory][Leases] failed to publish sidecar {}: {}",
                    self._path,
                    exc,
                )
                raise MemoryStorageError(
                    f"failed to publish selection lease sidecar at {self._path}"
                ) from exc
            try:
                kept = tuple(cid for cid in ordered if card_lookup(cid) is not None)
            except BaseException:
                self._rollback_attach_locked(attempt_id, attached)
                self._sync_best_effort_locked()
                raise
            vanished = attached - set(kept)
            if vanished:
                self._rollback_attach_locked(attempt_id, vanished)
                self._sync_best_effort_locked()
            return SelectionReservation(committed=True, card_ids=kept)

    def _publish_acquisition_locked(
        self, attached_by_attempt: dict[str, set[str]]
    ) -> None:
        if self._sync_locked():
            return
        for attempt_id, card_ids in attached_by_attempt.items():
            self._rollback_attach_locked(attempt_id, card_ids)
        raise MemoryStorageError(
            f"failed to publish selection lease sidecar at {self._path}"
        )

    def _sync_best_effort_locked(self) -> None:
        # WHY: a stale release only over-protects cards and cannot expose them.
        self._sync_locked()

    def _sync_locked(self) -> bool:
        try:
            with CardBankFileLock(self._lock_path, exclusive=True):
                try:
                    owners = self._read_owners_unlocked()
                except ValueError as exc:
                    logger.error(
                        "[Memory][Leases] corrupt sidecar preserved at {}: {}",
                        self._path,
                        exc,
                    )
                    return False
                now = datetime.now(UTC)
                owners = {
                    owner_key: owner
                    for owner_key, owner in owners.items()
                    if owner_key == self._owner_key or self._owner_is_live(owner, now)
                }
                self._replace_own_owner_locked(owners, now)
                self._write_owners_unlocked(owners)
        except Exception as exc:
            logger.warning(
                "[Memory][Leases] failed to sync sidecar {}: {}", self._path, exc
            )
            return False
        return True

    def _expanded_own_cards_locked(self) -> list[str]:
        return [
            card_id
            for card_id, count in sorted(self._card_owner_count.items())
            for _ in range(count)
        ]

    def _pending_counts_locked(
        self, owners: dict[str, dict[str, object]]
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for owner_key, owner in owners.items():
            cards = (
                self._expanded_own_cards_locked()
                if owner_key == self._owner_key
                else cast(list[str], owner["cards"])
            )
            for card_id in cards:
                value = str(card_id)
                counts[value] = counts.get(value, 0) + 1
        return counts

    def _live_owners_unlocked(self, now: datetime) -> dict[str, dict[str, object]]:
        return {
            owner_key: owner
            for owner_key, owner in self._read_owners_unlocked().items()
            if owner_key == self._owner_key or self._owner_is_live(owner, now)
        }

    def _replace_own_owner_locked(
        self, owners: dict[str, dict[str, object]], now: datetime
    ) -> None:
        own_cards = self._expanded_own_cards_locked()
        if not own_cards:
            owners.pop(self._owner_key, None)
            return
        # WHY: TTL only bounds foreign-host crash residue.
        deadline = now + timedelta(seconds=self._ttl_seconds)
        owners[self._owner_key] = {
            "pid": self._pid,
            "pid_start": self._pid_start,
            "host": self._host,
            "deadline_utc": deadline.isoformat(),
            "cards": own_cards,
        }

    def _selection_token_locked(self, owners: dict[str, dict[str, object]]) -> str:
        return _selection_token(self._pending_counts_locked(owners))

    def _read_live_foreign_ids_locked(self) -> frozenset[str]:
        with CardBankFileLock(self._lock_path, exclusive=False):
            owners = self._read_owners_unlocked()
        now = datetime.now(UTC)
        return frozenset(
            card_id
            for owner_key, owner in owners.items()
            if owner_key != self._owner_key and self._owner_is_live(owner, now)
            for card_id in cast(list[str], owner["cards"])
        )

    def _read_owners_unlocked(self) -> dict[str, dict[str, object]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise MemoryStorageError(
                f"selection lease sidecar unreadable at {self._path}: {exc}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"selection lease sidecar is not valid UTF-8: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"selection lease sidecar is not valid JSON: {exc}"
            ) from exc

        try:
            if not isinstance(raw, dict) or set(raw) != {"owners"}:
                raise TypeError("top-level object must contain only 'owners'")
            raw_owners = raw["owners"]
            if not isinstance(raw_owners, dict):
                raise TypeError("'owners' must be an object")
            owners: dict[str, dict[str, object]] = {}
            for owner_key, owner in raw_owners.items():
                owners[owner_key] = self._validate_owner(owner_key, owner)
            return owners
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid selection lease sidecar model: {exc}") from exc

    @staticmethod
    def _validate_owner(owner_key: object, owner: object) -> dict[str, object]:
        if not isinstance(owner_key, str) or not owner_key:
            raise TypeError("owner keys must be non-empty strings")
        if not isinstance(owner, dict) or set(owner) != {
            "pid",
            "pid_start",
            "host",
            "deadline_utc",
            "cards",
        }:
            raise TypeError(f"owner {owner_key!r} has invalid fields")
        pid = owner["pid"]
        pid_start = owner["pid_start"]
        host = owner["host"]
        deadline_utc = owner["deadline_utc"]
        cards = owner["cards"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise TypeError(f"owner {owner_key!r} has invalid pid")
        if (
            isinstance(pid_start, bool)
            or not isinstance(pid_start, int)
            or pid_start < 0
        ):
            raise TypeError(f"owner {owner_key!r} has invalid pid_start")
        if not isinstance(host, str) or not host:
            raise TypeError(f"owner {owner_key!r} has invalid host")
        if not isinstance(deadline_utc, str):
            raise TypeError(f"owner {owner_key!r} has invalid deadline")
        deadline = datetime.fromisoformat(deadline_utc)
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise ValueError(f"owner {owner_key!r} deadline must include a timezone")
        if not isinstance(cards, list) or any(
            not isinstance(card_id, str) or not card_id.strip() for card_id in cards
        ):
            raise TypeError(f"owner {owner_key!r} has invalid cards")
        return {
            "pid": pid,
            "pid_start": pid_start,
            "host": host,
            "deadline_utc": deadline_utc,
            "cards": list(cards),
        }

    def _owner_is_live(self, owner: dict[str, object], now: datetime) -> bool:
        if owner["host"] == self._host:
            pid = cast(int, owner["pid"])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                pass
            except OSError:
                pass
            pid_start = cast(int, owner["pid_start"])
            if pid_start == 0:
                return True
            current_pid_start = _read_pid_start(pid)
            return current_pid_start is None or current_pid_start == pid_start
        deadline = datetime.fromisoformat(cast(str, owner["deadline_utc"]))
        return deadline.astimezone(UTC) > now

    def _write_owners_unlocked(self, owners: dict[str, dict[str, object]]) -> None:
        if not owners and not self._path.exists():
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.parent / f"{self._path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        try:
            payload = {"owners": dict(sorted(owners.items()))}
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self._path)
        finally:
            tmp.unlink(missing_ok=True)

    def _warn_read_failure(self, exc: Exception) -> None:
        logger.warning(
            "[Memory][Leases] failed closed reading sidecar {}: {}", self._path, exc
        )


__all__ = [
    "InFlightSelectionRegistry",
    "SelectionLease",
    "SharedSelectionRegistry",
]
