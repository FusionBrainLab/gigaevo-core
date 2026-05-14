"""Test-only fake DAG runner.

Any program flipped to QUEUED is automatically pushed through
QUEUED → RUNNING → DONE on the next loop iteration. Used to exercise
:class:`gigaevo.evolution.engine.refresh.ParentRefresher` without
spinning up the real DagRunner.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from gigaevo.evolution.engine.refresh import ParentRefresher
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


class FakeDag:
    """In-memory DAG runner stand-in driven by an asyncio loop."""

    def __init__(self, storage):
        self.storage = storage
        self.evaluations = 0
        self._flip_count: dict[str, int] = defaultdict(int)
        self._task: asyncio.Task | None = None
        # When True, ``_loop`` stops touching any program in this set.
        # Lets tests freeze a program mid-QUEUED to test failure modes.
        self.frozen_ids: set[str] = set()

    async def add_program(self, name: str) -> Program:
        prog = Program(code=f"def {name}(): return 0")
        prog.state = ProgramState.DONE
        await self.storage.add(prog)
        return prog

    async def discard(self, pid: str) -> None:
        prog = await self.storage.get(pid)
        if prog is None:
            return
        await self.storage.batch_transition_by_ids(
            [pid], prog.state.value, ProgramState.DISCARDED.value
        )

    def flip_count_for(self, pid: str) -> int:
        return self._flip_count[pid]

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                queued_ids = await self.storage.get_ids_by_status(
                    ProgramState.QUEUED.value
                )
                active = [pid for pid in queued_ids if pid not in self.frozen_ids]
                if active:
                    # QUEUED -> RUNNING -> DONE (two-step; the schema does
                    # not permit QUEUED -> DONE directly).
                    await self.storage.batch_transition_by_ids(
                        active,
                        ProgramState.QUEUED.value,
                        ProgramState.RUNNING.value,
                    )
                    await self.storage.batch_transition_by_ids(
                        active,
                        ProgramState.RUNNING.value,
                        ProgramState.DONE.value,
                    )
                    for pid in active:
                        self._flip_count[pid] += 1
                        self.evaluations += 1
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Stay alive across transient storage hiccups in tests.
                await asyncio.sleep(0.01)


async def build_test_refresher(storage) -> tuple[ParentRefresher, Program, FakeDag]:
    """Wire a ParentRefresher with a FakeDag against the given storage.

    Returns ``(refresher, seed_program, fake_dag)``.
    """
    fake_dag = FakeDag(storage)
    fake_dag.start()
    seed = await fake_dag.add_program("p1")
    refresher = ParentRefresher(storage=storage, poll_interval=0.02)
    return refresher, seed, fake_dag
