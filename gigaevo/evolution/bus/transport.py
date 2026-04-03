"""Transport layer for cross-run migration bus.

Provides ``MigrantEnvelope`` (the wire format) and ``RedisStreamTransport``
(XADD/XREAD + SETNX exclusive claiming).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel
import redis.asyncio as aioredis

if TYPE_CHECKING:
    from gigaevo.evolution.bus.topology import Topology


class MigrantEnvelope(BaseModel):
    """Wire format for a program migrating between runs."""

    source_run_id: str
    program_id: str
    program_data: dict[str, Any]
    published_at: float
    generation: int

    def to_stream_fields(self) -> dict[str, str]:
        return {
            "source_run_id": self.source_run_id,
            "program_id": self.program_id,
            "program_data": json.dumps(self.program_data),
            "published_at": str(self.published_at),
            "generation": str(self.generation),
        }

    @classmethod
    def from_stream_fields(
        cls, fields: dict[bytes | str, bytes | str]
    ) -> MigrantEnvelope:
        def _s(v: bytes | str) -> str:
            return v.decode() if isinstance(v, bytes) else v

        program_data_raw = _s(
            fields[b"program_data" if b"program_data" in fields else "program_data"]
        )
        parsed = json.loads(program_data_raw)

        return cls(
            source_run_id=_s(
                fields.get(b"source_run_id", fields.get("source_run_id", b""))
            ),
            program_id=_s(fields.get(b"program_id", fields.get("program_id", b""))),
            program_data=parsed,
            published_at=float(
                _s(fields.get(b"published_at", fields.get("published_at", b"0")))
            ),
            generation=int(
                _s(fields.get(b"generation", fields.get("generation", b"0")))
            ),
        )


class Transport(ABC):
    """Abstract transport for publishing and consuming migrant envelopes."""

    @abstractmethod
    async def publish(self, envelope: MigrantEnvelope) -> None: ...

    @abstractmethod
    async def consume(
        self, max_count: int, topology: Topology, local_run_id: str
    ) -> list[MigrantEnvelope]: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def save_cursor(self) -> None: ...

    @abstractmethod
    async def restore_cursor(self) -> None: ...


class RedisStreamTransport(Transport):
    """Redis Streams transport with SETNX exclusive claiming.

    Uses XADD with MAXLEN for bounded stream, XREAD with own cursor
    (NOT consumer groups), and SET NX EX for exclusive claiming.
    Topology filtering happens BEFORE SETNX to avoid stealing claims
    from the correct consumer (critical for RingTopology).
    """

    def __init__(
        self,
        run_id: str,
        stream_key: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 15,
        max_stream_len: int = 1000,
        claim_ttl: int = 120,
        block_ms: int = 5000,
    ):
        self.run_id = run_id
        self.stream_key = stream_key
        self._host = host
        self._port = port
        self._db = db
        self._max_stream_len = max_stream_len
        self._claim_ttl = claim_ttl
        self._block_ms = block_ms
        self._last_id = "0-0"
        self._redis: aioredis.Redis | None = None

    async def start(self) -> None:
        self._redis = aioredis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
        )
        logger.info(
            "[MigrationBus] Transport started | run_id={} stream={} db={}",
            self.run_id,
            self.stream_key,
            self._db,
        )

    async def stop(self) -> None:
        if self._redis:
            await self._redis.aclose()  # type: ignore[attr-defined]
            self._redis = None

    async def publish(self, envelope: MigrantEnvelope) -> None:
        assert self._redis is not None, "Transport not started"
        await self._redis.xadd(
            self.stream_key,
            envelope.to_stream_fields(),  # type: ignore[arg-type]
            maxlen=self._max_stream_len,
            approximate=True,
        )
        logger.debug(
            "[MigrationBus] Published {} from run {}",
            envelope.program_id[:8],
            envelope.source_run_id,
        )

    async def consume(
        self, max_count: int, topology: Topology, local_run_id: str
    ) -> list[MigrantEnvelope]:
        assert self._redis is not None, "Transport not started"
        result: list[MigrantEnvelope] = []

        # XREAD with bounded block so the poll loop can check _running
        response = await self._redis.xread(
            {self.stream_key: self._last_id},
            count=max_count,
            block=self._block_ms,
        )
        if not response:
            return result

        for _stream_name, messages in response:
            for msg_id, fields in messages:
                msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                self._last_id = msg_id_str

                envelope = MigrantEnvelope.from_stream_fields(fields)

                # Skip self-messages
                if envelope.source_run_id == self.run_id:
                    continue

                # Topology filter BEFORE SETNX — critical for RingTopology
                if not topology.should_accept(envelope, local_run_id):
                    continue

                # Claim key uses msg_id (unique, monotonic) not program_id
                claim_key = f"{self.stream_key}:claim:{msg_id_str}"
                claimed = await self._redis.set(
                    claim_key,
                    self.run_id,
                    nx=True,
                    ex=self._claim_ttl,
                )
                if claimed:
                    result.append(envelope)
                else:
                    logger.debug(
                        "[MigrationBus] Claim lost for {} (already claimed)",
                        envelope.program_id[:8],
                    )

        return result

    async def save_cursor(self) -> None:
        if self._redis is not None:
            await self._redis.hset(
                f"{self.stream_key}:cursors",
                self.run_id,
                self._last_id,
            )

    async def restore_cursor(self) -> None:
        if self._redis is not None:
            raw = await self._redis.hget(
                f"{self.stream_key}:cursors",
                self.run_id,
            )
            if raw:
                self._last_id = raw.decode() if isinstance(raw, bytes) else raw
                logger.info(
                    "[MigrationBus] Restored cursor={} for run={}",
                    self._last_id,
                    self.run_id,
                )
