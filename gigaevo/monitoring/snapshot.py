from __future__ import annotations

from dataclasses import dataclass, field

from gigaevo.monitoring.run_spec import RunSpec


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable snapshot of a single run's state at a point in time.

    Produced by redis_queries, consumed by alerts, watchdog plugins,
    and notification formatters. Never written back to Redis.
    """

    run_spec: RunSpec
    generation: int | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    total_programs: int | None = None
    valid_programs: int | None = None
    running_programs: int | None = None
    queued_programs: int | None = None
    done_programs: int | None = None
    validator_mean_s: float | None = None
    validator_max_s: float | None = None
    total_keys: int | None = None
    pid: int | None = None
    pid_alive: bool | None = None
    error: str | None = None

    @property
    def invalid_rate(self) -> float | None:
        """Fraction of programs that failed validation (0.0-1.0)."""
        if self.total_programs is None or self.valid_programs is None:
            return None
        if self.total_programs == 0:
            return 0.0
        return (self.total_programs - self.valid_programs) / self.total_programs

    @property
    def has_error(self) -> bool:
        """True if this snapshot captured an error during collection."""
        return self.error is not None

    def is_stalled(self, previous: RunSnapshot) -> bool:
        """True if no progress between this snapshot and the previous one.

        Multi-signal: generation unchanged AND no running programs AND
        no new program submissions. All three must agree.
        """
        if self.generation is None or previous.generation is None:
            return False
        gen_unchanged = self.generation == previous.generation
        no_running = self.running_programs is not None and self.running_programs == 0
        no_new_submissions = (
            self.total_programs is not None
            and previous.total_programs is not None
            and self.total_programs == previous.total_programs
        )
        return gen_unchanged and no_running and no_new_submissions

    @classmethod
    def empty(cls, run_spec: RunSpec) -> RunSnapshot:
        """Create an empty snapshot (used when Redis is unreachable)."""
        return cls(run_spec=run_spec)
