from __future__ import annotations

from gigaevo.database.redis.config import RedisKeyConfig


class RedisProgramKeys:
    """Key generation for Redis program storage."""

    def __init__(self, config: RedisKeyConfig):
        self.config = config

    @property
    def prefix(self) -> str:
        return self.config.key_prefix

    def program(self, pid: str) -> str:
        """Key for storing a program by ID."""
        return self.config.program_key_tpl.format(
            prefix=self.config.key_prefix, pid=pid
        )

    def status_set(self, status: str) -> str:
        """Key for the set of program IDs with a given status."""
        return self.config.status_set_tpl.format(
            prefix=self.config.key_prefix, status=status
        )

    def status_stream(self) -> str:
        """Key for the status events stream."""
        return self.config.status_stream_tpl.format(prefix=self.config.key_prefix)

    def timestamp(self) -> str:
        """Key for the atomic counter/timestamp."""
        return f"{self.config.key_prefix}:ts"

    def instance_lock(self) -> str:
        """Key for the instance lock."""
        return f"{self.config.key_prefix}:__instance_lock__"

    def program_pattern(self) -> str:
        """Pattern for matching all program keys (for SCAN)."""
        return self.program("*")

    def archive(self) -> str:
        """Key for the elite archive hash."""
        return f"{self.config.key_prefix}:archive"

    def archive_reverse(self) -> str:
        """Key for the reverse index: program_id -> cells."""
        return f"{self.config.key_prefix}:archive:reverse"
