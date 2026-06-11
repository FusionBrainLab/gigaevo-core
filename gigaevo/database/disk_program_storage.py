from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.exceptions import StorageError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.json import dumps as _dumps
from gigaevo.utils.json import loads as _loads

if TYPE_CHECKING:
    from gigaevo.utils.trackers.base import LogWriter

__all__ = ["DiskProgramStorageConfig", "DiskProgramStorage"]

PROGRAMS_DIR = "programs"
STATUS_SETS_FILE = "status_sets.json"
RUN_STATE_FILE = "run_state.json"
INSTANCE_LOCK_FILE = "instance.lock"


class DiskProgramStorageConfig(BaseModel):
    root_dir: str
    key_prefix: str = Field(min_length=1)
    read_only: bool = False


class DiskProgramStorage(ProgramStorage):
    """Single-process, asyncio-only program storage backed by JSON files.

    Authoritative state lives in memory (programs dict + status sets +
    run-state dict); every mutation is written through to disk under
    ``root_dir/<prefix>`` so a later process can resume. Atomicity here means
    single-process atomicity: all mutations run under one ``asyncio.Lock``,
    not cross-process transactions.
    """

    def __init__(
        self, config: DiskProgramStorageConfig, writer: LogWriter | None = None
    ) -> None:
        super().__init__(read_only=config.read_only)
        self.config = config
        self.base_dir = Path(config.root_dir) / config.key_prefix.replace("/", "_")
        self._programs: dict[str, Program] = {}
        self._status_sets: dict[str, set[str]] = {}
        self._run_state: dict[str, str] = {}
        self._counter = 0
        self._loaded = False
        self._lock = asyncio.Lock()
        self._owns_instance_lock = False

    @property
    def key_prefix(self) -> str:
        return self.config.key_prefix

    @property
    def _programs_dir(self) -> Path:
        return self.base_dir / PROGRAMS_DIR

    @property
    def _lock_path(self) -> Path:
        return self.base_dir / INSTANCE_LOCK_FILE

    # --------------------- Load / persist helpers (lock held) ---------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._programs_dir.is_dir():
            for path in sorted(self._programs_dir.glob("*.json")):
                try:
                    program = Program.from_dict(_loads(path.read_text()))
                except Exception as e:
                    logger.warning(
                        "[DiskProgramStorage] Corrupt program file {}: {}", path, e
                    )
                    continue
                self._programs[program.id] = program
                self._counter = max(self._counter, program.atomic_counter)
        status_path = self.base_dir / STATUS_SETS_FILE
        if status_path.is_file():
            raw = _loads(status_path.read_text())
            self._status_sets = {s: set(ids) for s, ids in raw.items()}
        else:
            for program in self._programs.values():
                self._status_sets.setdefault(program.state.value, set()).add(program.id)
        run_state_path = self.base_dir / RUN_STATE_FILE
        if run_state_path.is_file():
            self._run_state = dict(_loads(run_state_path.read_text()))

    def _persist_program(self, program: Program) -> None:
        self._programs_dir.mkdir(parents=True, exist_ok=True)
        (self._programs_dir / f"{program.id}.json").write_text(
            program.model_dump_json()
        )

    def _persist_status_sets(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        payload = {s: sorted(ids) for s, ids in self._status_sets.items()}
        (self.base_dir / STATUS_SETS_FILE).write_text(_dumps(payload))

    def _persist_run_state(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / RUN_STATE_FILE).write_text(_dumps(self._run_state))

    def _store(self, program: Program) -> Program:
        """Deep-copy, stamp revision counter, keep in memory + write through."""
        self._counter += 1
        stored = program.model_copy(deep=True)
        stored.atomic_counter = self._counter
        self._programs[stored.id] = stored
        self._persist_program(stored)
        return stored

    def _discard_from_all_sets(self, program_id: str) -> None:
        for ids in self._status_sets.values():
            ids.discard(program_id)

    # --------------------- CRUD Operations ---------------------

    async def add(self, program: Program) -> None:
        self.require_writable("add")
        async with self._lock:
            self._ensure_loaded()
            self._discard_from_all_sets(program.id)
            stored = self._store(program)
            self._status_sets.setdefault(stored.state.value, set()).add(stored.id)
            self._persist_status_sets()

    async def update(self, program: Program) -> None:
        self.require_writable("update")
        async with self._lock:
            self._ensure_loaded()
            self._store(program)

    async def get(self, program_id: str) -> Program | None:
        async with self._lock:
            self._ensure_loaded()
            program = self._programs.get(program_id)
            return program.model_copy(deep=True) if program is not None else None

    async def mget(
        self,
        program_ids: list[str],
        *,
        exclude: frozenset[str] | None = None,
    ) -> list[Program]:
        async with self._lock:
            self._ensure_loaded()
            return [
                self._programs[pid].model_copy(deep=True)
                for pid in program_ids
                if pid in self._programs
            ]

    async def exists(self, program_id: str) -> bool:
        async with self._lock:
            self._ensure_loaded()
            return program_id in self._programs

    async def remove(self, program_id: str) -> None:
        self.require_writable("remove")
        async with self._lock:
            self._ensure_loaded()
            self._programs.pop(program_id, None)
            (self._programs_dir / f"{program_id}.json").unlink(missing_ok=True)
            self._discard_from_all_sets(program_id)
            self._persist_status_sets()

    async def get_all(self, *, exclude: frozenset[str] | None = None) -> list[Program]:
        async with self._lock:
            self._ensure_loaded()
            return [p.model_copy(deep=True) for p in self._programs.values()]

    async def get_all_program_ids(self) -> list[str]:
        async with self._lock:
            self._ensure_loaded()
            return list(self._programs)

    async def size(self) -> int:
        async with self._lock:
            self._ensure_loaded()
            return len(self._programs)

    async def has_data(self) -> bool:
        return await self.size() > 0

    # --------------------- Status Operations ---------------------

    async def transition_status(
        self, program_id: str, old: str | None, new: str
    ) -> None:
        self.require_writable("transition_status")
        async with self._lock:
            self._ensure_loaded()
            if old:
                self._status_sets.setdefault(old, set()).discard(program_id)
            self._status_sets.setdefault(new, set()).add(program_id)
            self._persist_status_sets()

    async def publish_status_event(
        self, status: str, program_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        logger.debug(
            "[DiskProgramStorage] status event {} for {} (no-op)", status, program_id
        )

    async def get_all_by_status(
        self, status: str, *, exclude: frozenset[str] | None = None
    ) -> list[Program]:
        async with self._lock:
            self._ensure_loaded()
            ids = self._status_sets.get(status, set())
            return [
                self._programs[pid].model_copy(deep=True)
                for pid in ids
                if pid in self._programs and self._programs[pid].state.value == status
            ]

    async def count_by_status(self, status: str) -> int:
        async with self._lock:
            self._ensure_loaded()
            return len(self._status_sets.get(status, set()))

    async def get_ids_by_status(self, status: str) -> list[str]:
        async with self._lock:
            self._ensure_loaded()
            return list(self._status_sets.get(status, set()))

    async def atomic_state_transition(
        self, program: Program, old_state: str | None, new_state: str
    ) -> None:
        self.require_writable("atomic_state_transition")
        async with self._lock:
            self._ensure_loaded()
            stored = self._store(program)
            self._discard_from_all_sets(stored.id)
            self._status_sets.setdefault(stored.state.value, set()).add(stored.id)
            self._persist_status_sets()

    # --------------------- Run State (resume support) ---------------------

    async def save_run_state(self, field: str, value: int | str) -> None:
        self.require_writable("save_run_state")
        async with self._lock:
            self._ensure_loaded()
            self._run_state[field] = str(value)
            self._persist_run_state()

    async def load_run_state(self, field: str) -> int | None:
        async with self._lock:
            self._ensure_loaded()
            raw = self._run_state.get(field)
            return int(raw) if raw is not None else None

    async def load_run_state_str(self, field: str) -> str | None:
        async with self._lock:
            self._ensure_loaded()
            return self._run_state.get(field)

    async def recover_stranded_programs(self) -> int:
        self.require_writable("recover_stranded_programs")
        async with self._lock:
            self._ensure_loaded()
            running = ProgramState.RUNNING.value
            queued = ProgramState.QUEUED.value
            recovered = 0
            for pid in sorted(self._status_sets.get(running, set())):
                program = self._programs.get(pid)
                self._status_sets[running].discard(pid)
                if program is None:
                    continue
                program.state = ProgramState.QUEUED
                self._store(program)
                self._status_sets.setdefault(queued, set()).add(pid)
                recovered += 1
            self._persist_status_sets()
        logger.info(
            "[DiskProgramStorage] Recovered {} stranded RUNNING → QUEUED", recovered
        )
        return recovered

    # --------------------- Admin Operations ---------------------

    async def clear(self) -> None:
        self.require_writable("clear")
        async with self._lock:
            shutil.rmtree(self.base_dir, ignore_errors=True)
            self._programs = {}
            self._status_sets = {}
            self._run_state = {}
            self._counter = 0
            self._owns_instance_lock = False
            self._loaded = True

    # --------------------- Instance Locking ---------------------

    def _read_lock_pid(self) -> int | None:
        try:
            return int(self._lock_path.read_text().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    async def acquire_instance_lock(self) -> bool:
        if self.config.read_only:
            logger.info(
                "[DiskProgramStorage] Skipping instance lock (read-only mode) "
                "for prefix '{}'",
                self.key_prefix,
            )
            return True
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = self._read_lock_pid()
            if pid == os.getpid() and self._owns_instance_lock:
                return True
            if pid is not None and self._pid_alive(pid):
                raise StorageError(
                    f"Cannot start: another instance (pid {pid}) is using disk "
                    f"storage prefix '{self.key_prefix}'. "
                    f"If this is a stale lock from a crashed instance, "
                    f"manually delete: {self._lock_path}"
                )
            logger.warning(
                "[DiskProgramStorage] Replacing stale instance lock (pid {})", pid
            )
            self._lock_path.write_text(str(os.getpid()))
        else:
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
        self._owns_instance_lock = True
        logger.info(
            "[DiskProgramStorage] Acquired exclusive lock for prefix '{}'",
            self.key_prefix,
        )
        return True

    async def renew_instance_lock(self) -> bool:
        if self.config.read_only:
            return True
        if not self._owns_instance_lock:
            return False
        if self._read_lock_pid() != os.getpid():
            logger.error(
                "[DiskProgramStorage] Lost instance lock! "
                "Another instance may have taken over."
            )
            return False
        return True

    async def release_instance_lock(self) -> None:
        if self.config.read_only or not self._owns_instance_lock:
            return
        self._lock_path.unlink(missing_ok=True)
        self._owns_instance_lock = False
        logger.info(
            "[DiskProgramStorage] Released lock for prefix '{}'", self.key_prefix
        )

    # --------------------- Shutdown ---------------------

    async def close(self) -> None:
        # Everything is write-through; nothing to flush beyond the lock.
        await self.release_instance_lock()
        logger.debug("[DiskProgramStorage] close() for prefix '{}'", self.key_prefix)

    def __repr__(self) -> str:
        return (
            f"<DiskProgramStorage "
            f"prefix={self.key_prefix!r} "
            f"dir={str(self.base_dir)!r} "
            f"read_only={self.config.read_only}>"
        )
