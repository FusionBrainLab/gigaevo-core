from __future__ import annotations

from typing import Any
import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gigaevo.evolution.engine.acceptor import (
    DefaultProgramEvolutionAcceptor,
    ProgramEvolutionAcceptor,
)
from gigaevo.evolution.engine.stopper import EvolutionStopper
from gigaevo.evolution.mutation.parent_selector import (
    ParentSelector,
    RandomParentSelector,
)

# Retired engine knobs that older yaml configs may still set. We drop them
# silently with a one-shot DeprecationWarning so external users get a single
# upgrade signal instead of a Pydantic crash. Each entry: ``key → reason``.
_DEPRECATED_FIELDS: dict[str, str] = {
    "max_mutations_per_generation": (
        "epoch size is no longer a knob; progress is driven entirely by the "
        "stopper (see config/stopper/) and JIT parent refresh"
    ),
    "generation_timeout": (
        "was already a deprecated no-op; per-program timeouts live on "
        "dag_timeout / stage_timeout"
    ),
    "refresh_passes": (
        "per-epoch archive refresh removed; ParentRefresher re-evaluates "
        "parents JIT as they are selected"
    ),
    "refresh_order": (
        "ordering knob removed alongside the per-epoch refresh; JIT refresh "
        "order is implicit"
    ),
}


class EngineConfig(BaseModel):
    """Configuration for the steady-state evolution engine.

    Parents are re-evaluated only when they are themselves selected as parents
    (:class:`gigaevo.evolution.engine.refresh.ParentRefresher`); there is no
    global archive refresh. Progress is governed entirely by ``stopper``
    (e.g. :class:`MaxMutantsStopper`).
    """

    loop_interval: float = Field(default=1.0, gt=0)
    max_elites_per_generation: int = Field(default=20, gt=0)
    metrics_collection_interval: float = Field(
        default=1.0, gt=0, description="Interval in seconds for metrics collection"
    )
    backpressure_sample_interval: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Cadence (seconds) at which BACKPRESSURE_SAMPLE events are emitted. "
            "Decoupled from ``loop_interval`` because the engine ticks at 1Hz "
            "for snapshotting work, but a 1Hz BACKPRESSURE_SAMPLE stream floods "
            "the log on long runs (~3.6k lines/hour, ~86k/day). 10s keeps a "
            "useful time-series for the flow profiler while dropping log "
            "volume 10x."
        ),
    )
    max_in_flight: int = Field(
        default=5,
        gt=0,
        description=(
            "Backpressure cap. Sizes BOTH the producer pool (concurrent "
            "LLM/refresh tasks; ``_producer_sema``) AND the buffer of "
            "produced-but-not-yet-ingested mutants (``_buffer_sema``). "
            "Steady-state pipeline depth is therefore ~2 × max_in_flight: "
            "~N producers alive (mix of LLM-running and holding ready "
            "result) plus ~N buffered (DAG queue + running + waiting "
            "ingest). The dispatcher acquires producer_sema and the "
            "ingestor releases buffer_sema as programs reach "
            "DONE/DISCARDED. ~4 concurrent producers per GPU server is "
            "the sweet spot (measured on Qwen3-235B). Default 5 is tuned "
            "for 3-4 servers with 4 runs."
        ),
    )
    parent_selector: ParentSelector = Field(
        default_factory=lambda: RandomParentSelector(num_parents=1)
    )
    program_acceptor: ProgramEvolutionAcceptor = Field(
        default_factory=lambda: DefaultProgramEvolutionAcceptor(),
        description="Acceptor for determining if programs should be accepted for evolution",
    )
    stopper: EvolutionStopper = Field(
        default_factory=EvolutionStopper,
        description="Pluggable stopping criterion. Authoritative termination signal for the "
        "engine loop. Configured via the ``stopper`` Hydra group "
        "(``config/stopper/``). Default is a no-op stopper that never stops.",
    )
    coalesce_refresh: bool = Field(
        default=True,
        description=(
            "When True (default), coalesce parent refreshes across "
            "concurrent mutations: a refreshed parent stays valid until "
            "any of its children completes (DONE or DISCARDED), and "
            "subsequent mutations of that parent skip the refresh while "
            "it is fresh. Refreshes of the same parent are still mutually "
            "exclusive (no double-flip). Set to False to opt out and "
            "restore the legacy behaviour, where the parent refresh lock "
            "is held across the entire child-DAG and only one child of a "
            "given parent can be in flight at a time."
        ),
    )
    post_step_hook_timeout_s: float = Field(
        default=300.0,
        gt=0,
        description=(
            "Wall-clock bound on a single ``post_step_hook`` invocation. The "
            "hook is user-configurable (production: CompositionInjectionHook "
            "walks the entire G archive); a hung hook would otherwise wedge "
            "the ingestor — no further sweeps fire and no new mutants reach "
            "the archive. Default 300s is generous enough for a 10k-program "
            "archive walk (~30s in production) plus 10× headroom. Tune up "
            "for archives in the 100k+ range or slow-storage backends."
        ),
    )
    post_step_hook_cancel_grace_s: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Grace period after cancelling an over-budget ``post_step_hook`` "
            "task. If the hook ignores ``CancelledError`` within this window "
            "we log and abandon — better an orphan coroutine than a wedged "
            "ingestor."
        ),
    )
    # ``extra="forbid"`` keeps typos loud (``max_inflight: 5`` etc.), while
    # the ``_strip_deprecated_keys`` validator below provides a soft landing
    # for keys that USED to exist and have been retired (see
    # ``_DEPRECATED_FIELDS``). Together they give the major-version upgrade a
    # one-stop deprecation path without silently accepting unknown extras.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _strip_deprecated_keys(cls, data: Any) -> Any:
        """Drop retired knobs from old yaml configs with a one-shot warning."""
        if not isinstance(data, dict):
            return data
        for dead_key, reason in _DEPRECATED_FIELDS.items():
            if dead_key in data:
                warnings.warn(
                    (
                        f"{dead_key!r} is no longer an engine config field "
                        f"and will be ignored ({reason}). Drop it from your "
                        f"yaml — this shim will be removed in a future "
                        f"release."
                    ),
                    DeprecationWarning,
                    stacklevel=2,
                )
                data.pop(dead_key)
        return data


class SteadyStateEngineConfig(EngineConfig):
    """Alias of :class:`EngineConfig` for Hydra ``_target_`` back-compat.

    The steady-state engine is the only engine; this class exists so
    existing config files that target it (``_target_:
    gigaevo.evolution.engine.SteadyStateEngineConfig``) keep resolving.
    No extra fields.
    """
