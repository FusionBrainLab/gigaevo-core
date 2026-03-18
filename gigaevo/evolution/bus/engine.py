"""BusedEvolutionEngine — EvolutionEngine with cross-run migration bus.

Single subclass that handles both publishing (rejected-but-valid programs)
and consuming (drain bus arrivals before elite selection).
"""

from __future__ import annotations

from loguru import logger

from gigaevo.evolution.bus.node import MigrationNode
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import Program


class BusedEvolutionEngine(EvolutionEngine):
    """EvolutionEngine with cross-run migration bus.

    Publish: overrides _notify_hook to publish strategy-rejected valid programs.
    Consume: overrides step to drain bus arrivals before elite selection.
    """

    def __init__(
        self,
        migration_node: MigrationNode,
        max_imports_per_generation: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._migration_node = migration_node
        self._max_imports = max_imports_per_generation
        self._node_started = False

    async def step(self) -> None:
        """Drain bus arrivals, then run normal generation step."""
        if not self._node_started:
            await self._migration_node.start()
            self._node_started = True

        await self._import_bus_arrivals()
        await super().step()

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

        if arrivals:
            logger.info(
                "[MigrationBus] Imported {}/{} bus arrivals",
                imported,
                len(arrivals),
            )

    async def _notify_hook(self, prog: Program, outcome: MutationOutcome) -> None:
        """Publish strategy-rejected valid programs to bus."""
        await super()._notify_hook(prog, outcome)
        if outcome == MutationOutcome.REJECTED_STRATEGY:
            if prog.metrics.get("is_valid", 0) > 0:
                try:
                    gen = prog.metadata.get("iteration", 0)
                    await self._migration_node.publish(prog, gen)
                except Exception as exc:
                    logger.warning(
                        "[MigrationBus] Publish failed for {}: {}",
                        prog.short_id,
                        exc,
                    )

    async def stop(self) -> None:
        """Stop bus node then engine."""
        if self._node_started:
            await self._migration_node.stop()
        await super().stop()
