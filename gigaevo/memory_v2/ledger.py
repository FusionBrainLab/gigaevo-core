"""Durable, conflict-detecting causal ledger for memory v2."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
from threading import RLock
import uuid

from loguru import logger

from gigaevo.memory.events import MemoryOutcome
from gigaevo.memory_v2.credit import LineageCreditResolver
from gigaevo.memory_v2.models import (
    CausalObservation,
    DecisionRecord,
    EnvironmentFingerprint,
    EvidenceSnapshot,
    OutcomeMeasurement,
    TerminalOutcome,
    canonical_digest,
)


class CausalLedgerConflict(RuntimeError):
    """A supposedly immutable decision or terminal was rewritten."""


class CausalLedgerInactive(RuntimeError):
    """The causal ledger was used outside its exclusive active lifetime."""


_NETWORK_FILESYSTEMS = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "lustre", "ceph", "glusterfs"}
)


def _filesystem_type(path: Path) -> str:
    """Resolve the longest matching mount entry without invoking a subprocess."""

    target = Path(os.path.realpath(path))
    best_mount = Path("/")
    best_type = "unknown"
    try:
        rows = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return best_type
    for row in rows:
        fields = row.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace("\\040", " "))
        try:
            target.relative_to(mount)
        except ValueError:
            continue
        if len(mount.parts) >= len(best_mount.parts):
            best_mount = mount
            best_type = fields[2]
    return best_type


class SqliteCausalLedger:
    """Crash-durable SQLite ledger shared by provider, engine, and analytics."""

    def __init__(
        self,
        *,
        path: str | Path,
        environment: EnvironmentFingerprint,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive")
        self.path = Path(path)
        self.environment = environment
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = RLock()
        self._mirror_path: Path | None = None
        self._database_path = self.path
        self._active = False
        self._process_lock_fd: int | None = None
        self._trajectory_id: str | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def trajectory_id(self) -> str:
        if not self._active or self._trajectory_id is None:
            raise CausalLedgerInactive("trajectory id requires an active causal ledger")
        return self._trajectory_id

    def activate(self) -> None:
        """Acquire exclusive ownership and initialize durable state.

        Hydra constructs the full object graph before the program-storage lock is
        acquired. Keeping construction inert prevents a losing process from
        copying or publishing the active process's NFS mirror.
        """

        with self._lock:
            if self._active:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(f".{self.path.name}.writer.lock")
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(lock_fd)
                raise CausalLedgerConflict(
                    f"causal ledger is already active: {self.path}"
                ) from exc
            metadata = f"pid={os.getpid()} environment={self.environment.digest}\n"
            os.ftruncate(lock_fd, 0)
            os.write(lock_fd, metadata.encode("ascii"))
            os.fsync(lock_fd)
            self._process_lock_fd = lock_fd

            filesystem = _filesystem_type(self.path.parent)
            if filesystem in _NETWORK_FILESYSTEMS:
                staging_root = (
                    Path(os.environ.get("TMPDIR", "/tmp")) / "gigaevo-memory-v2"
                )
                staging_root.mkdir(parents=True, exist_ok=True)
                staging_key = canonical_digest(
                    {
                        "path": str(self.path.resolve()),
                        "environment": self.environment.digest,
                    }
                )
                self._database_path = staging_root / (
                    f"{staging_key}.{os.getpid()}.{uuid.uuid4().hex}.sqlite3"
                )
                self._mirror_path = self.path
                if self.path.exists():
                    shutil.copyfile(self.path, self._database_path)
                logger.warning(
                    "[MemoryV2][Ledger] {} is {}; using local SQLite {} with a "
                    "synchronous checkpoint mirror",
                    self.path,
                    filesystem,
                    self._database_path,
                )
            else:
                self._database_path = self.path
                self._mirror_path = None
            self._active = True
            try:
                self._initialize()
                self._trajectory_id = self._load_or_create_trajectory_id()
                self._sync_mirror()
            except Exception:
                self._deactivate(remove_staging=True)
                raise

    def close(self) -> None:
        with self._lock:
            if not self._active and self._process_lock_fd is None:
                return
            try:
                if self._active:
                    self._sync_mirror()
            finally:
                self._deactivate(remove_staging=True)

    def _deactivate(self, *, remove_staging: bool) -> None:
        database_path = self._database_path
        is_staging = self._mirror_path is not None
        self._active = False
        self._trajectory_id = None
        self._mirror_path = None
        self._database_path = self.path
        if remove_staging and is_staging:
            database_path.unlink(missing_ok=True)
            database_path.with_name(f"{database_path.name}-journal").unlink(
                missing_ok=True
            )
        if self._process_lock_fd is not None:
            fcntl.flock(self._process_lock_fd, fcntl.LOCK_UN)
            os.close(self._process_lock_fd)
            self._process_lock_fd = None

    def next_event_ordinal(self, task_key: str) -> int:
        key = task_key.strip() or "<unknown>"
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT next_ordinal FROM event_ordinals WHERE task_key = ?",
                (key,),
            ).fetchone()
            ordinal = int(row[0]) if row is not None else 0
            connection.execute(
                """
                INSERT INTO event_ordinals(task_key, next_ordinal) VALUES (?, ?)
                ON CONFLICT(task_key) DO UPDATE SET next_ordinal = excluded.next_ordinal
                """,
                (key, ordinal + 1),
            )
            connection.commit()
        self._sync_mirror()
        return ordinal

    def record_decision(self, record: DecisionRecord) -> None:
        if record.context.environment != self.environment:
            raise ValueError("decision environment does not match its causal ledger")
        payload = record.model_dump_json(exclude_computed_fields=True)
        digest = canonical_digest(
            record.model_dump(mode="json", exclude_computed_fields=True)
        )
        inserted = False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT record_json, record_hash FROM decisions WHERE decision_id = ?",
                (record.decision_id,),
            ).fetchone()
            if existing is not None:
                self._load_decision(existing[0], existing[1])
                if existing[1] != digest:
                    raise CausalLedgerConflict(
                        f"conflicting decision record {record.decision_id!r}"
                    )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO decisions(
                        decision_id, attempt_id, environment_hash, event_ordinal,
                        record_json, record_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.decision_id,
                        record.attempt_id,
                        self.environment.digest,
                        record.event_ordinal,
                        payload,
                        digest,
                    ),
                )
                inserted = True
            connection.commit()
        try:
            self._sync_mirror()
        except Exception:
            if inserted:
                with self._connection() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "DELETE FROM decisions WHERE decision_id = ?",
                        (record.decision_id,),
                    )
                    connection.commit()
            raise

    def link_attempt_child(
        self,
        *,
        attempt_id: str,
        child_id: str,
        completion_ordinal: int,
    ) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT decision_id, record_json, record_hash FROM decisions "
                "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return False
            decision_id, record_json, record_hash = row
            record = self._load_decision(record_json, record_hash)
            payload = (
                decision_id,
                attempt_id,
                child_id,
                record.context.parent_id,
                completion_ordinal,
            )
            existing = connection.execute(
                "SELECT decision_id, attempt_id, child_id, base_id, completion_ordinal "
                "FROM decision_children WHERE decision_id = ? OR child_id = ?",
                (decision_id, child_id),
            ).fetchone()
            if existing is not None and tuple(existing) != payload:
                raise CausalLedgerConflict(
                    f"conflicting child link for attempt {attempt_id!r}"
                )
            if existing is None:
                connection.execute(
                    "INSERT INTO decision_children(decision_id, attempt_id, child_id, "
                    "base_id, completion_ordinal) VALUES (?, ?, ?, ?, ?)",
                    payload,
                )
            connection.commit()
        self._sync_mirror()
        return True

    def record_attempt_failure(
        self,
        *,
        attempt_id: str,
        status: str,
        failure_stage: str,
        completion_ordinal: int,
    ) -> bool:
        if status not in ("invalid", "censored"):
            raise ValueError(f"unsupported attempt failure status {status!r}")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT d.record_json, d.record_hash, c.child_id, c.base_id
                FROM decisions AS d
                LEFT JOIN decision_children AS c USING(decision_id)
                WHERE d.attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        if row is None:
            return False
        record = self._load_decision(row[0], row[1])
        child_id = row[2] or f"no-child:{attempt_id}"
        base_id = row[3] or record.context.parent_id
        self.record_terminal(
            TerminalOutcome(
                decision_id=record.decision_id,
                child_id=child_id,
                base_id=base_id,
                primary_metric=record.context.reward.primary_metric,
                higher_is_better=record.context.reward.higher_is_better,
                ope_eligible=True,
                status=status,
                censor_reason=failure_stage if status == "censored" else "",
                failure_stage=failure_stage,
                completion_ordinal=completion_ordinal,
            )
        )
        return True

    def record_missing_child(self, child_id: str, *, failure_stage: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT c.decision_id, c.base_id, c.completion_ordinal,
                       d.record_json, d.record_hash
                FROM decision_children AS c
                JOIN decisions AS d USING(decision_id)
                WHERE c.child_id = ?
                """,
                (child_id,),
            ).fetchone()
        if row is None:
            return False
        record = self._load_decision(row[3], row[4])
        self.record_terminal(
            TerminalOutcome(
                decision_id=row[0],
                child_id=child_id,
                base_id=row[1],
                primary_metric=record.context.reward.primary_metric,
                higher_is_better=record.context.reward.higher_is_better,
                ope_eligible=True,
                status="censored",
                censor_reason=failure_stage,
                failure_stage=failure_stage,
                completion_ordinal=row[2],
            )
        )
        return True

    def pending_child_ids(self) -> tuple[str, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.child_id FROM decision_children AS c
                JOIN decisions AS d USING(decision_id)
                LEFT JOIN terminals AS t USING(decision_id)
                WHERE t.decision_id IS NULL AND d.environment_hash = ?
                ORDER BY c.completion_ordinal, c.child_id
                """,
                (self.environment.digest,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def reconcile_unlinked_attempts(self, *, completion_ordinal: int) -> int:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.attempt_id FROM decisions AS d
                LEFT JOIN decision_children AS c USING(decision_id)
                LEFT JOIN terminals AS t USING(decision_id)
                WHERE c.decision_id IS NULL AND t.decision_id IS NULL
                  AND d.environment_hash = ?
                ORDER BY d.event_ordinal, d.decision_id
                """,
                (self.environment.digest,),
            ).fetchall()
        for (attempt_id,) in rows:
            self.record_attempt_failure(
                attempt_id=attempt_id,
                status="censored",
                failure_stage="startup_orphan_reconciliation",
                completion_ordinal=completion_ordinal,
            )
        return len(rows)

    def record_memory_outcome(self, event: MemoryOutcome) -> None:
        measurement = None
        if event.status == "outcome":
            if event.fitness_delta is None:
                raise ValueError("numeric terminal is missing fitness_delta")
            measurement = OutcomeMeasurement(
                value=event.fitness_delta,
                se=event.fitness_delta_se,
                n_pairs=event.n_pairs,
                kind=event.measurement_kind,
                pairing_signature=event.pairing_signature,
            )
        terminal = TerminalOutcome(
            decision_id=event.decision_id,
            child_id=event.child_id,
            base_id=event.base_id,
            primary_metric=event.primary_metric,
            higher_is_better=event.higher_is_better,
            ope_eligible=event.ope_eligible,
            status=event.status,
            measurement=measurement,
            censor_reason=event.censor_reason,
            failure_stage=event.failure_stage,
            completion_ordinal=event.completion_ordinal,
        )
        self.record_terminal(terminal)

    def record_terminal(self, terminal: TerminalOutcome) -> None:
        payload = terminal.model_dump_json(exclude_computed_fields=True)
        digest = canonical_digest(
            terminal.model_dump(mode="json", exclude_computed_fields=True)
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            decision = connection.execute(
                """
                SELECT record_json, record_hash FROM decisions
                WHERE decision_id = ? AND environment_hash = ?
                """,
                (terminal.decision_id, self.environment.digest),
            ).fetchone()
            if decision is None:
                raise CausalLedgerConflict(
                    f"terminal precedes decision {terminal.decision_id!r}"
                )
            record = self._load_decision(decision[0], decision[1])
            reward = record.context.reward
            if terminal.base_id != record.context.parent_id:
                raise CausalLedgerConflict(
                    f"terminal base does not match decision {terminal.decision_id!r}"
                )
            child_link = connection.execute(
                "SELECT child_id, base_id FROM decision_children WHERE decision_id = ?",
                (terminal.decision_id,),
            ).fetchone()
            if child_link is None:
                if not terminal.child_id.startswith("no-child:"):
                    raise CausalLedgerConflict(
                        f"terminal child is not linked to decision "
                        f"{terminal.decision_id!r}"
                    )
            elif (terminal.child_id, terminal.base_id) != tuple(child_link):
                raise CausalLedgerConflict(
                    f"terminal child link does not match decision "
                    f"{terminal.decision_id!r}"
                )
            if (
                terminal.primary_metric != reward.primary_metric
                or terminal.higher_is_better != reward.higher_is_better
            ):
                raise CausalLedgerConflict(
                    f"terminal reward semantics do not match decision "
                    f"{terminal.decision_id!r}"
                )
            if terminal.measurement is not None:
                parent_value = record.context.parent_metrics[reward.primary_metric]
                if reward.higher_is_better:
                    gain_lower = reward.metric_lower_bound - parent_value
                    gain_upper = reward.metric_upper_bound - parent_value
                else:
                    gain_lower = parent_value - reward.metric_upper_bound
                    gain_upper = parent_value - reward.metric_lower_bound
                if not (
                    gain_lower - 1e-9
                    <= terminal.measurement.value
                    <= gain_upper + 1e-9
                ):
                    raise CausalLedgerConflict(
                        "terminal gain exceeds parent-specific metric bounds for "
                        f"{terminal.decision_id!r}"
                    )
            existing = connection.execute(
                "SELECT terminal_json, terminal_hash FROM terminals WHERE decision_id = ?",
                (terminal.decision_id,),
            ).fetchone()
            if existing is not None:
                self._load_terminal(existing[0], existing[1])
                if existing[1] != digest:
                    raise CausalLedgerConflict(
                        f"conflicting terminal for decision {terminal.decision_id!r}"
                    )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO terminals(decision_id, terminal_json, terminal_hash)
                    VALUES (?, ?, ?)
                    """,
                    (terminal.decision_id, payload, digest),
                )
            connection.commit()
        self._sync_mirror()

    def snapshot(self) -> EvidenceSnapshot:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.record_json, d.record_hash,
                       t.terminal_json, t.terminal_hash
                FROM decisions AS d
                LEFT JOIN terminals AS t USING(decision_id)
                WHERE d.environment_hash = ?
                ORDER BY d.event_ordinal, d.decision_id
                """,
                (self.environment.digest,),
            ).fetchall()
        observations: list[CausalObservation] = []
        decisions: list[DecisionRecord] = []
        terminals: dict[str, TerminalOutcome] = {}
        pending_treatment: dict[str, int] = {}
        pending_bank: dict[str, int] = {}
        censored = 0
        ineligible = 0
        version_parts: list[str] = []
        model_version_parts: list[str] = []
        for record_json, record_hash, terminal_json, terminal_hash in rows:
            record = self._load_decision(record_json, record_hash)
            decisions.append(record)
            version_parts.extend((record_hash, terminal_hash or "pending"))
            terminal = (
                self._load_terminal(terminal_json, terminal_hash)
                if terminal_json is not None
                else None
            )
            if terminal is not None:
                terminals[record.decision_id] = terminal
            proposed_id = record.proposed_treatment_id
            if proposed_id is None:
                continue
            card = next(
                row for row in record.candidates if row.treatment_id == proposed_id
            )
            if terminal_json is None:
                pending_treatment[proposed_id] = (
                    pending_treatment.get(proposed_id, 0) + 1
                )
                pending_bank[card.bank_card_id] = (
                    pending_bank.get(card.bank_card_id, 0) + 1
                )
                continue
            assert terminal is not None
            if terminal.status == "censored":
                censored += 1
                continue
            if not terminal.ope_eligible:
                ineligible += 1
                continue
            if terminal.status == "outcome" and terminal.measurement is None:
                raise CausalLedgerConflict(
                    f"valid terminal {terminal.decision_id!r} lacks a measurement"
                )
            if record.offer_probability is None or record.proposal_probability is None:
                raise CausalLedgerConflict(
                    f"proposal {record.decision_id!r} lacks action propensities"
                )
            if record.joint_action_probability is None:
                raise CausalLedgerConflict(
                    f"proposal {record.decision_id!r} lacks joint propensity"
                )
            q_hats = (
                record.reward_q_hat_control,
                record.reward_q_hat_treated,
                record.risk_q_hat_control,
                record.risk_q_hat_treated,
            )
            if any(value is None for value in q_hats):
                raise CausalLedgerConflict(
                    f"proposal {record.decision_id!r} lacks frozen q-hats"
                )
            observations.append(
                CausalObservation(
                    decision_id=record.decision_id,
                    event_ordinal=record.event_ordinal,
                    card=card,
                    context=record.context,
                    treatment=record.delivered,
                    offer_propensity=record.offer_probability,
                    proposal_propensity=record.proposal_probability,
                    joint_action_propensity=record.joint_action_probability,
                    status=terminal.status,
                    measurement=terminal.measurement,
                    reward_q_hat_control=record.reward_q_hat_control,
                    reward_q_hat_treated=record.reward_q_hat_treated,
                    risk_q_hat_control=record.risk_q_hat_control,
                    risk_q_hat_treated=record.risk_q_hat_treated,
                )
            )
            model_version_parts.extend((record_hash, terminal_hash))
        lineage_outcomes, reward_observations = LineageCreditResolver().resolve(
            decisions,
            terminals,
            {row.decision_id: row for row in observations},
        )
        decisions_by_id = {row.decision_id: row for row in decisions}
        for lineage in lineage_outcomes:
            if lineage.status != "pending":
                continue
            record = decisions_by_id[lineage.decision_id]
            treatment_id = record.proposed_treatment_id
            if treatment_id is None:
                raise CausalLedgerConflict("pending lineage has no proposed treatment")
            card = next(
                row for row in record.candidates if row.treatment_id == treatment_id
            )
            pending_treatment[treatment_id] = (
                pending_treatment.get(treatment_id, 0) + 1
            )
            pending_bank[card.bank_card_id] = (
                pending_bank.get(card.bank_card_id, 0) + 1
            )
        version = canonical_digest(version_parts)
        return EvidenceSnapshot(
            version=version,
            model_version=canonical_digest(
                {
                    "immediate": model_version_parts,
                    "reward": [
                        row.model_dump(mode="json") for row in reward_observations
                    ],
                }
            ),
            observations=tuple(observations),
            reward_observations=reward_observations,
            lineage_outcomes=lineage_outcomes,
            pending_by_treatment=pending_treatment,
            pending_by_bank_card=pending_bank,
            censored_count=censored,
            ineligible_count=ineligible,
        )

    def decisions(self) -> tuple[DecisionRecord, ...]:
        with self._connection() as connection:
            rows: Sequence[tuple[str, str]] = connection.execute(
                """
                SELECT record_json, record_hash FROM decisions
                WHERE environment_hash = ?
                ORDER BY event_ordinal, decision_id
                """,
                (self.environment.digest,),
            ).fetchall()
        return tuple(self._load_decision(row[0], row[1]) for row in rows)

    def terminals(self) -> tuple[TerminalOutcome, ...]:
        with self._connection() as connection:
            rows: Sequence[tuple[str, str]] = connection.execute(
                """
                SELECT t.terminal_json, t.terminal_hash
                FROM terminals AS t
                JOIN decisions AS d USING(decision_id)
                WHERE d.environment_hash = ?
                ORDER BY t.rowid
                """,
                (self.environment.digest,),
            ).fetchall()
        return tuple(self._load_terminal(row[0], row[1]) for row in rows)

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions(
                    decision_id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL UNIQUE,
                    environment_hash TEXT NOT NULL,
                    event_ordinal INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL DEFAULT
                        (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE TABLE IF NOT EXISTS event_ordinals(
                    task_key TEXT PRIMARY KEY,
                    next_ordinal INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger_metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS decisions_environment_ordinal
                    ON decisions(environment_hash, event_ordinal);
                CREATE TABLE IF NOT EXISTS terminals(
                    decision_id TEXT PRIMARY KEY
                        REFERENCES decisions(decision_id) ON DELETE RESTRICT,
                    terminal_json TEXT NOT NULL,
                    terminal_hash TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL DEFAULT
                        (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE TABLE IF NOT EXISTS decision_children(
                    decision_id TEXT PRIMARY KEY
                        REFERENCES decisions(decision_id) ON DELETE RESTRICT,
                    attempt_id TEXT NOT NULL UNIQUE,
                    child_id TEXT NOT NULL UNIQUE,
                    base_id TEXT NOT NULL,
                    completion_ordinal INTEGER NOT NULL,
                    linked_at_utc TEXT NOT NULL DEFAULT
                        (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                """
            )

    def _load_or_create_trajectory_id(self) -> str:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM ledger_metadata WHERE key = 'trajectory_id'"
            ).fetchone()
            if row is None:
                trajectory_id = f"trajectory-{uuid.uuid4().hex}"
                connection.execute(
                    "INSERT INTO ledger_metadata(key, value) VALUES (?, ?)",
                    ("trajectory_id", trajectory_id),
                )
            else:
                trajectory_id = str(row[0])
            connection.commit()
        return trajectory_id

    @staticmethod
    def _load_decision(record_json: str, record_hash: str) -> DecisionRecord:
        record = DecisionRecord.model_validate_json(record_json)
        actual = canonical_digest(
            record.model_dump(mode="json", exclude_computed_fields=True)
        )
        if actual != record_hash:
            raise CausalLedgerConflict(
                f"decision payload hash mismatch for {record.decision_id!r}"
            )
        return record

    @staticmethod
    def _load_terminal(terminal_json: str, terminal_hash: str) -> TerminalOutcome:
        terminal = TerminalOutcome.model_validate_json(terminal_json)
        actual = canonical_digest(
            terminal.model_dump(mode="json", exclude_computed_fields=True)
        )
        if actual != terminal_hash:
            raise CausalLedgerConflict(
                f"terminal payload hash mismatch for {terminal.decision_id!r}"
            )
        return terminal

    def _sync_mirror(self) -> None:
        if self._mirror_path is None:
            return
        with self._lock:
            temporary = self._mirror_path.with_name(
                f".{self._mirror_path.name}.{os.getpid()}.tmp"
            )
            try:
                with (
                    self._database_path.open("rb") as source,
                    temporary.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                os.replace(temporary, self._mirror_path)
                directory_fd = os.open(self._mirror_path.parent, os.O_RDONLY)
                try:
                    try:
                        os.fsync(directory_fd)
                    except OSError:
                        # Atomic replace already published a fully fsynced file.
                        # Treat a directory-fsync limitation as committed so the
                        # caller cannot misclassify an exposed treatment.
                        logger.opt(exception=True).warning(
                            "[MemoryV2][Ledger] directory fsync failed after "
                            "atomic mirror publication"
                        )
                finally:
                    os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if not self._active:
                raise CausalLedgerInactive(
                    "causal ledger must be activated after the storage instance lock"
                )
            connection = sqlite3.connect(
                self._database_path,
                timeout=self.busy_timeout_ms / 1000.0,
                isolation_level=None,
            )
            try:
                connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute("PRAGMA synchronous = FULL")
                connection.execute("PRAGMA foreign_keys = ON")
                yield connection
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
