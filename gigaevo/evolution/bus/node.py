"""MigrationNode — buffer, background polling, and orphan conversion.

Bridges the raw ``Transport`` and ``Topology`` into a high-level API
consumed by ``BusedEvolutionEngine``.
"""

from __future__ import annotations

import asyncio
from collections import deque
import uuid

from loguru import logger

from gigaevo.evolution.bus.topology import Topology
from gigaevo.evolution.bus.transport import MigrantEnvelope, Transport
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState


class MigrationNode:
    """Manages publishing, polling, buffering, and orphan conversion for one run."""

    def __init__(
        self,
        run_id: str,
        transport: Transport,
        topology: Topology,
        max_buffer_size: int = 50,
        consume_interval: float = 5.0,
        max_consume_per_poll: int = 20,
    ):
        self.run_id = run_id
        self._transport = transport
        self._topology = topology
        self._max_buffer_size = max_buffer_size
        self._consume_interval = consume_interval
        self._max_consume_per_poll = max_consume_per_poll

        # No maxlen — we use manual check with "drop newest" policy
        self._buffer: deque[MigrantEnvelope] = deque()
        self._poll_task: asyncio.Task[None] | None = None
        self._running = False
        self._published_count = 0
        self._received_count = 0
        self._dropped_count = 0

    async def start(self) -> None:
        await self._transport.start()
        await self._transport.restore_cursor()
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(),
            name=f"migration-poll-{self.run_id}",
        )
        logger.info("[MigrationBus] Node started | run_id={}", self.run_id)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self._transport.save_cursor()
        await self._transport.stop()
        logger.info(
            "[MigrationBus] Node stopped | run_id={} published={} received={} dropped={}",
            self.run_id,
            self._published_count,
            self._received_count,
            self._dropped_count,
        )

    async def publish(self, program: Program, generation: int) -> None:
        envelope = MigrantEnvelope(
            source_run_id=self.run_id,
            program_id=program.id,
            program_data=program.to_dict(),
            published_at=program.created_at.timestamp(),
            generation=generation,
        )
        await self._transport.publish(envelope)
        self._published_count += 1
        logger.debug(
            "[MigrationBus] Published {} (gen={}) from run {}",
            program.short_id,
            generation,
            self.run_id,
        )

    def drain_received(self, max_count: int) -> list[Program]:
        """Drain up to ``max_count`` orphan programs from the receive buffer."""
        result: list[Program] = []
        while self._buffer and len(result) < max_count:
            envelope = self._buffer.popleft()
            result.append(self._envelope_to_orphan(envelope))
        return result

    def _envelope_to_orphan(self, envelope: MigrantEnvelope) -> Program:
        """Convert a MigrantEnvelope into an orphan Program with a fresh UUID."""
        program = Program.from_dict(envelope.program_data)
        # Fresh UUID to avoid cross-run ID collisions
        program.id = str(uuid.uuid4())
        program.lineage = Lineage(parents=[], children=[], mutation=None)
        program.set_metadata("migration_source_run", envelope.source_run_id)
        program.set_metadata("migration_source_id", envelope.program_id)
        program.set_metadata("migration_generation", envelope.generation)
        program.set_metadata("is_migrant", True)
        program.state = ProgramState.DONE
        return program

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                envelopes = await self._transport.consume(
                    self._max_consume_per_poll, self._topology, self.run_id
                )
                for env in envelopes:
                    if len(self._buffer) >= self._max_buffer_size:
                        self._dropped_count += 1
                        logger.warning(
                            "[MigrationBus] Buffer full — dropping migrant {} from {}",
                            env.program_id[:8],
                            env.source_run_id,
                        )
                        continue
                    self._buffer.append(env)
                    self._received_count += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[MigrationBus] Poll error: {}", exc)

            await asyncio.sleep(self._consume_interval)
