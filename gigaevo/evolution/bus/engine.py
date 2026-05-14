"""BusedEvolutionEngine — SteadyStateEvolutionEngine with cross-run migration bus.

Inherits from :class:`SteadyStateEvolutionEngine` rather than the
generational base. Bus arrivals are imported on a periodic background
task that runs alongside the dispatcher and ingestor loops; rejected
valid programs are still published via the ``_notify_hook`` override.
"""

from __future__ import annotations

import asyncio
import contextlib

from loguru import logger

from gigaevo.evolution.bus.node import MigrationNode
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import Program


class BusedEvolutionEngine(SteadyStateEvolutionEngine):
    """Steady-state engine with cross-run migration bus.

    Publish: overrides ``_notify_hook`` to publish strategy-rejected valid
    programs to the migration node.

    Consume: spawns a periodic drain task in ``run()`` that imports any
    bus arrivals into the local archive.
    """

    def __init__(
        self,
        migration_node: MigrationNode,
        max_imports_per_generation: int = 10,
        bus_drain_interval: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._migration_node = migration_node
        self._max_imports = max_imports_per_generation
        self._bus_drain_interval = bus_drain_interval
        self._node_started = False
        self._bus_drain_task: asyncio.Task | None = None

    async def run(self) -> None:
        await self._migration_node.start()
        self._node_started = True
        self._bus_drain_task = asyncio.create_task(
            self._bus_drain_loop(), name="bus-drain"
        )
        try:
            await super().run()
        finally:
            if self._bus_drain_task is not None:
                self._bus_drain_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._bus_drain_task
                self._bus_drain_task = None

    async def stop(self) -> None:
        if self._node_started:
            try:
                await self._migration_node.stop()
            except Exception as exc:
                logger.warning("[MigrationBus] node.stop() failed: {}", exc)
            self._node_started = False
        await super().stop()

    async def _bus_drain_loop(self) -> None:
        """Periodically pull arrivals from the migration node and import them."""
        logger.info(
            "[MigrationBus] drain-loop start (interval={:.1f}s, max_imports={})",
            self._bus_drain_interval,
            self._max_imports,
        )
        try:
            while self._running:
                try:
                    await self._import_bus_arrivals()
                except Exception as exc:
                    logger.warning("[MigrationBus] drain pass failed: {}", exc)
                await asyncio.sleep(self._bus_drain_interval)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info("[MigrationBus] drain-loop stop")

    async def _import_bus_arrivals(self) -> None:
        """Import exclusively-claimed orphans into archive with trusted metrics."""
        arrivals = self._migration_node.drain_received(self._max_imports)
        if not arrivals:
            return

        imported = 0
        for program in arrivals:
            try:
                if not self.config.program_acceptor.is_accepted(program):
                    logger.debug(
                        "[MigrationBus] Migrant {} rejected by acceptor",
                        program.short_id,
                    )
                    continue

                await self.storage.add(program)
                if await self.strategy.add(program):
                    imported += 1
                    logger.info(
                        "[MigrationBus] Imported {} (fitness={}) from {}",
                        program.short_id,
                        program.metrics.get("fitness", "?"),
                        program.metadata.get("migration_source_run", "?"),
                    )
            except Exception as exc:
                logger.warning(
                    "[MigrationBus] Import failed for migrant {}: {}",
                    program.short_id,
                    exc,
                )

        logger.info(
            "[MigrationBus] Imported {}/{} bus arrivals",
            imported,
            len(arrivals),
        )

    async def _notify_hook(self, prog: Program, outcome: MutationOutcome) -> None:
        """Publish strategy-rejected valid programs to bus."""
        await super()._notify_hook(prog, outcome)
        if outcome != MutationOutcome.REJECTED_STRATEGY:
            return
        if prog.metrics.get("is_valid", 0) <= 0:
            return
        try:
            gen = prog.iteration
            await self._migration_node.publish(prog, gen)
        except Exception as exc:
            logger.warning(
                "[MigrationBus] Publish failed for {}: {}",
                prog.short_id,
                exc,
            )
