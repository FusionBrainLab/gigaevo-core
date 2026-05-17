"""PromptFetcher abstraction for decoupling prompt acquisition from LLM agents.

Two concrete implementations:
  - FixedDirPromptFetcher: reads from a directory or package defaults.
  - GigaEvoArchivePromptFetcher: reads champion from a co-evolving GigaEvo archive.

Outcome-stats key layout for :class:`GigaEvoArchivePromptFetcher`:

    prompt:trials:{prompt_id}         G-counter, total trials
    prompt:successes:{prompt_id}      G-counter, improvement count
    prompt:metrics_count:{prompt_id}  G-counter, denominator for
                                       per-metric mean sums
    prompt:metric:{prompt_id}:{key}   G-counter, fixed-point metric sum
                                       (scale :data:`_METRIC_FIXED_POINT_SCALE`)
    prompt:metric_keys:{prompt_id}    Set of observed metric keys
    prompt:fitnesses:{prompt_id}      Bounded list capped at
                                       :data:`_FITNESS_WINDOW`

Every increment is partitioned by :class:`ActorIdentity` so two engines
on the same prompt id never collide; the consensus value is the sum
across actors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.dataplane import (
    ActorIdentity,
    CounterKey,
    DataPlane,
    Err,
)
from gigaevo.prompts import load_prompt
from gigaevo.prompts.coevolution.stats import prompt_text_to_id

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.llm.bandit import MutationOutcome


@dataclass
class _PromptPack:
    """Internal holder for co-evolved system + optional user prompt texts."""

    system: str
    user: str | None
    prompt_id: str


_CURRENT_PACK: ContextVar[_PromptPack | None] = ContextVar(
    "gigaevo.prompts.fetcher._current_pack", default=None
)
"""Per-task pack handle. The system-prompt fetch stores the sampled
pack; the user-prompt fetch reads it back. A :class:`ContextVar`
isolates concurrent mutations even when they share one fetcher."""

_FITNESS_WINDOW: int = 20
"""Max recent fitness samples retained per prompt id; enforced
server-side via :meth:`DataPlane.bounded_list_push` (LPUSH + LTRIM)."""

_METRIC_FIXED_POINT_SCALE: int = 1000
"""Scale used to coerce float metric values into integer HINCRBY
deltas; three decimal digits of precision."""


def _trials_key(prompt_id: str) -> CounterKey:
    """G-counter key for the per-prompt trial count."""
    return CounterKey(f"prompt:trials:{prompt_id}")


def _successes_key(prompt_id: str) -> CounterKey:
    """G-counter key for the per-prompt success count."""
    return CounterKey(f"prompt:successes:{prompt_id}")


def _metrics_count_key(prompt_id: str) -> CounterKey:
    """G-counter key for the count of trials carrying ``child_metrics``."""
    return CounterKey(f"prompt:metrics_count:{prompt_id}")


def _metric_sum_key(prompt_id: str, metric_key: str) -> CounterKey:
    """G-counter key for the fixed-point sum of one metric across trials."""
    return CounterKey(f"prompt:metric:{prompt_id}:{metric_key}")


def _metric_directory_key(prompt_id: str) -> str:
    """Set name carrying every metric key observed for this prompt id."""
    return f"prompt:metric_keys:{prompt_id}"


def _fitness_list_key(prompt_id: str) -> str:
    """Bounded-list key carrying the recent fitness samples."""
    return f"prompt:fitnesses:{prompt_id}"


def _metric_to_fixed_point(value: float) -> int:
    """Coerce a metric float into an HINCRBY-friendly signed integer.

    Non-finite values are squashed to zero rather than corrupting the
    counter. Rounding error is bounded by 1 / ``_METRIC_FIXED_POINT_SCALE``.
    """
    if not isinstance(value, (int, float)):
        return 0
    if value != value or value in (float("inf"), float("-inf")):
        return 0
    return int(round(float(value) * _METRIC_FIXED_POINT_SCALE))


@dataclass
class FetchedPrompt:
    """Result of a prompt fetch operation."""

    text: str
    prompt_id: str | None  # None for fixed prompts (no tracking)


class PromptFetcher(ABC):
    """Abstracts how a system/user prompt is obtained by an LLM agent.

    Two concrete implementations:
      - FixedDirPromptFetcher: reads from files (default).
      - GigaEvoArchivePromptFetcher: reads champion from a co-evolving GigaEvo archive.
    """

    @property
    def is_dynamic(self) -> bool:
        """Whether this fetcher returns different prompts across calls.

        FixedDirPromptFetcher returns False (static prompts).
        GigaEvoArchivePromptFetcher returns True (champion changes over time).
        Used by agents to decide whether to re-fetch on every call.
        """
        return False

    @abstractmethod
    def fetch(self, agent_name: str, prompt_type: str) -> FetchedPrompt:
        """Fetch a prompt template for the given agent and prompt type.

        Args:
            agent_name: Agent type directory (insights, lineage, scoring, mutation)
            prompt_type: Prompt file type (system, user)

        Returns:
            FetchedPrompt with template text and optional tracking ID
        """
        ...

    async def record_outcome(
        self,
        prompt_id: str | None,
        child_fitness: float,
        parent_fitness: float,
        higher_is_better: bool,
        outcome: MutationOutcome,
        child_metrics: dict[str, float] | None = None,
    ) -> None:
        """Called after mutation outcome known. Default no-op.

        Args:
            prompt_id: ID of the prompt used (None to skip)
            child_fitness: Fitness of the child program
            parent_fitness: Fitness of the best parent
            higher_is_better: Whether higher fitness is better
            outcome: Outcome enum (ACCEPTED, REJECTED_STRATEGY, REJECTED_ACCEPTOR)
            child_metrics: Full metrics dict of the child program (optional)
        """

    async def start(self, storage: ProgramStorage | None = None) -> None:
        """Optional lifecycle hook called when the fetcher is started."""

    async def stop(self) -> None:
        """Optional lifecycle hook called when the fetcher is stopped."""

    def get_stats(self) -> dict[str, Any]:
        """Return stats dict for logging/monitoring."""
        return {}


class FixedDirPromptFetcher(PromptFetcher):
    """Reads from a directory or package defaults.

    Caches loaded templates in memory to avoid repeated disk reads.
    """

    def __init__(self, prompts_dir: str | Path | None = None):
        self._prompts_dir = prompts_dir
        self._cache: dict[tuple[str, str], FetchedPrompt] = {}
        logger.info(
            "[FixedDirPromptFetcher] Initialized with prompts_dir={}",
            prompts_dir or "(package defaults)",
        )

    def fetch(self, agent_name: str, prompt_type: str) -> FetchedPrompt:
        key = (agent_name, prompt_type)
        if key not in self._cache:
            text = load_prompt(agent_name, prompt_type, prompts_dir=self._prompts_dir)
            self._cache[key] = FetchedPrompt(text=text, prompt_id=None)
        return self._cache[key]


class GigaEvoArchivePromptFetcher(PromptFetcher):
    """Reads the current MAP-Elites champion from a co-running prompt GigaEvo instance.

    On fetch():
      1. Reads all elite program IDs from the prompt run's Redis archive (TTL-cached)
      2. Fetches programs from Redis and finds the one with the best fitness
      3. Executes its entrypoint() in-process to get the prompt text
      4. Returns FetchedPrompt(text, prompt_id) for outcome tracking
      Falls back to FixedDirPromptFetcher until the first champion is available.

    On record_outcome():
      Writes per-prompt stats through the :class:`DataPlane` G-counter
      and bounded-list primitives so :class:`PromptFitnessStage` can
      compute fitness for the prompt run. Skips ``REJECTED_ACCEPTOR``
      outcomes (no reliable fitness).

    Args:
        prompt_redis_db: Redis DB of the prompt GigaEvo run
        main_redis_prefix: Key prefix of the main run (for stats keys)
        main_redis_db: Redis DB of the main run (required for stats writes).
            Set to the same value as redis.db of the main run — in Hydra config
            use ``main_redis_db: ${redis.db}``.
        prompt_prefix: Key prefix of the prompt run (default: "prompt_evolution")
        host: Redis host (default: localhost)
        port: Redis port (default: 6379)
        cache_ttl_seconds: How long to cache the current champion (default: 30s)
        fallback_prompts_dir: Directory for fallback prompts while no champion exists
        fitness_key: Metric key used for champion selection (default: "fitness")
        main_dataplane: Pre-wired :class:`DataPlane` for main-run writes; when
            ``None`` the fetcher lazily constructs one in :meth:`start`.
        prompt_dataplane: Pre-wired :class:`DataPlane` for prompt-run archive
            reads; when ``None`` the fetcher lazily constructs one.
        actor: Optional :class:`ActorIdentity` carrying ``(run_id, worker_id)``
            for the G-counter sub-counts. When unset the fetcher derives one
            from ``main_redis_prefix`` plus the process pid so two engines on
            the same prefix still pick up distinct identities.
    """

    @property
    def is_dynamic(self) -> bool:
        return True

    def __init__(
        self,
        prompt_redis_db: int,
        main_redis_prefix: str,
        main_redis_db: int | None = None,
        prompt_prefix: str = "prompt_evolution",
        archive_prefix: str = "island_fitness_island",
        host: str = "localhost",
        port: int = 6379,
        cache_ttl_seconds: float = 30.0,
        fallback_prompts_dir: str | Path | None = None,
        fitness_key: str = "fitness",
        *,
        main_dataplane: DataPlane | None = None,
        prompt_dataplane: DataPlane | None = None,
        actor: ActorIdentity | None = None,
    ):
        self._prompt_redis_db = prompt_redis_db
        self._main_redis_prefix = main_redis_prefix
        self._main_redis_db = main_redis_db
        self._prompt_prefix = prompt_prefix
        self._archive_prefix = archive_prefix
        self._host = host
        self._port = port
        self._cache_ttl = cache_ttl_seconds
        self._fallback = FixedDirPromptFetcher(fallback_prompts_dir)
        self._fitness_key = fitness_key

        # Candidate list is shared per fetcher; the sampled pack is
        # per-task via ``_CURRENT_PACK``.
        self._cached_candidates: list[tuple[str, float, str]] | None = None
        self._cached_pack: _PromptPack | None = None  # last sampled (for get_stats)
        self._cache_timestamp: float = 0.0

        # DataPlane handles. ``_owned`` is True only when constructed
        # lazily by :meth:`start`; injected handles live with the engine.
        self._main_dp: DataPlane | None = main_dataplane
        self._prompt_dp: DataPlane | None = prompt_dataplane
        self._main_dp_owned: bool = False
        self._prompt_dp_owned: bool = False

        self._actor: ActorIdentity = actor or self._derive_default_actor()

        self._fetch_errors: int = 0
        self._cache_hits: int = 0
        self._using_archive: bool = False
        self._empty_refresh_count: int = 0
        self._system_fetch_count: int = 0

        logger.info(
            "[GigaEvoArchivePromptFetcher] Init | prompt_db={} prompt_prefix={!r} "
            "archive_prefix={!r} main_prefix={!r} main_db={} fitness_key={!r} "
            "cache_ttl={}s fallback={}",
            self._prompt_redis_db,
            self._prompt_prefix,
            self._archive_prefix,
            self._main_redis_prefix,
            main_redis_db,
            self._fitness_key,
            self._cache_ttl,
            fallback_prompts_dir or "(package defaults)",
        )

    def _derive_default_actor(self) -> ActorIdentity:
        """Synthesise an :class:`ActorIdentity` from prefix + pid.

        Used when the caller does not supply one; the resulting actor
        is process-unique under one main-run prefix.
        """
        import os
        import socket

        from gigaevo.dataplane import RunId, WorkerId

        # ActorIdentity rejects ':' / '/' so sanitise before construction.
        clean_prefix = self._main_redis_prefix.replace(":", "-").replace("/", "-")
        if not clean_prefix:
            clean_prefix = "prompts"
        return ActorIdentity(
            run_id=RunId(clean_prefix),
            worker_id=WorkerId(f"{socket.gethostname()}-{os.getpid()}"),
        )

    def _build_dataplane(self, db: int, key_prefix: str) -> DataPlane:
        """Construct a DataPlane bound to the given Redis DB and prefix."""
        url = f"redis://{self._host}:{self._port}/{db}"
        return DataPlane(url, key_prefix=key_prefix)

    def attach_dataplane(
        self,
        main_dataplane: DataPlane,
        prompt_dataplane: DataPlane,
        actor: ActorIdentity,
    ) -> None:
        """Rebind engine-owned DataPlane handles + actor onto this fetcher.

        Both handles are recorded as engine-owned so :meth:`stop` does
        not shut them down. Idempotent on identical triples; a different
        triple raises :class:`RuntimeError`. Must be called before
        :meth:`start`.
        """
        if (
            self._main_dp is main_dataplane
            and self._prompt_dp is prompt_dataplane
            and self._actor == actor
        ):
            return
        if (
            self._main_dp is not None
            or self._prompt_dp is not None
            or self._main_dp_owned
            or self._prompt_dp_owned
        ) and (
            self._main_dp is not main_dataplane
            or self._prompt_dp is not prompt_dataplane
            or self._actor != actor
        ):
            raise RuntimeError(
                "GigaEvoArchivePromptFetcher already has a different DataPlane "
                "or actor attached; refusing to overwrite. Construct a fresh "
                "fetcher or detach the existing handles before reattaching."
            )
        self._main_dp = main_dataplane
        self._prompt_dp = prompt_dataplane
        self._main_dp_owned = False
        self._prompt_dp_owned = False
        self._actor = actor
        logger.info(
            "[GigaEvoArchivePromptFetcher] DataPlane attached | main_prefix={!r} "
            "prompt_prefix={!r} actor={}",
            main_dataplane.key_prefix,
            prompt_dataplane.key_prefix,
            actor.pack(),
        )

    async def start(self, storage: ProgramStorage | None = None) -> None:
        """Lazily construct + start DataPlane handles.

        Already-started handles are returned in place. Safe to call
        multiple times.
        """
        del storage  # reserved for ProgramStorage-aware tests
        if self._main_dp is None and self._main_redis_db is not None:
            self._main_dp = self._build_dataplane(
                self._main_redis_db, self._main_redis_prefix
            )
            self._main_dp_owned = True
        if self._prompt_dp is None:
            self._prompt_dp = self._build_dataplane(
                self._prompt_redis_db, self._prompt_prefix
            )
            self._prompt_dp_owned = True
        if self._main_dp is not None:
            await self._main_dp.startup()
        if self._prompt_dp is not None:
            await self._prompt_dp.startup()

    async def stop(self) -> None:
        """Tear down DataPlane handles the fetcher constructed itself.

        Injected handles are left alone. Idempotent.
        """
        if self._main_dp is not None and self._main_dp_owned:
            try:
                await self._main_dp.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown best effort
                logger.warning(
                    "[GigaEvoArchivePromptFetcher] main DataPlane shutdown failed: {}",
                    exc,
                )
            self._main_dp = None
            self._main_dp_owned = False
        if self._prompt_dp is not None and self._prompt_dp_owned:
            try:
                await self._prompt_dp.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown best effort
                logger.warning(
                    "[GigaEvoArchivePromptFetcher] prompt DataPlane shutdown failed: {}",
                    exc,
                )
            self._prompt_dp = None
            self._prompt_dp_owned = False

    def _is_cache_stale(self) -> bool:
        return (time.monotonic() - self._cache_timestamp) >= self._cache_ttl

    async def _refresh_candidates_async(self) -> list[tuple[str, float, str]] | None:
        """Read candidate prompts from the prompt run's archive via the dataplane.

        Returns a list of (program_id, fitness, code) tuples, or None when
        the archive is empty / unreachable. The archive lives in the prompt
        run's DB under ``{prompt_prefix}:archive`` (a hash whose values are
        elite program IDs) and ``{prompt_prefix}:program:{pid}`` (a JSON
        blob per program).
        """
        if self._prompt_dp is None:
            return None
        archive_key = f"{self._archive_prefix}:archive"
        try:
            ids_result = await self._prompt_dp.raw_hash_values(archive_key)
        except Exception as exc:  # noqa: BLE001 - read boundary, surface as miss
            self._fetch_errors += 1
            logger.warning(
                "[GigaEvoArchivePromptFetcher] Archive hvals error (#{}): {}",
                self._fetch_errors,
                exc,
            )
            return None
        if isinstance(ids_result, Err):
            self._fetch_errors += 1
            logger.warning(
                "[GigaEvoArchivePromptFetcher] Archive hvals failed: {}",
                ids_result.error,
            )
            return None
        program_ids = ids_result.value
        if not program_ids:
            self._empty_refresh_count += 1
            if self._empty_refresh_count == 1 or self._empty_refresh_count % 10 == 0:
                logger.info(
                    "[GigaEvoArchivePromptFetcher] Archive empty (check #{}) "
                    "— using fallback",
                    self._empty_refresh_count,
                )
            return None

        candidates: list[tuple[str, float, str]] = []
        for pid in program_ids:
            program_key = f"{self._prompt_prefix}:program:{pid}"
            raw_result = await self._prompt_dp.raw_get(program_key)
            if isinstance(raw_result, Err) or raw_result.value is None:
                continue
            try:
                data = json.loads(raw_result.value)
                metrics = data.get("metrics", {})
                fitness = float(metrics.get(self._fitness_key, 0.0))
                code = data.get("code", "")
                if code:
                    candidates.append((pid, fitness, code))
            except Exception as exc:  # noqa: BLE001 - per-entry diagnostic
                logger.debug(
                    "[GigaEvoArchivePromptFetcher] Error parsing program {}: {}",
                    pid,
                    exc,
                )
                continue
        return candidates if candidates else None

    def _refresh_candidates(self) -> list[tuple[str, float, str]] | None:
        """Sync bridge to :meth:`_refresh_candidates_async`."""
        try:
            return asyncio.run(self._refresh_candidates_async())
        except RuntimeError:
            # Already inside an event loop; run the coroutine on a
            # fresh loop in a worker thread to avoid re-entry.
            import threading

            holder: list[list[tuple[str, float, str]] | None] = [None]

            def _runner() -> None:
                loop = asyncio.new_event_loop()
                try:
                    holder[0] = loop.run_until_complete(
                        self._refresh_candidates_async()
                    )
                finally:
                    loop.close()

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            return holder[0]

    def _sample_prompt(self) -> _PromptPack | None:
        """Sample a prompt from cached candidates using fitness-proportional selection.

        Called on every fetch() for mutation — each mutation gets an independent sample.

        Returns:
            _PromptPack if successful, None if no candidates or entrypoint fails
        """
        candidates = self._cached_candidates
        if not candidates:
            return None

        epsilon = 0.01
        weights = [max(f, epsilon) for _, f, _ in candidates]
        chosen_pid, chosen_fitness, chosen_code = random.choices(
            candidates, weights=weights, k=1
        )[0]

        pack = self._execute_entrypoint(chosen_code)
        if pack is None:
            return None

        user_preview = repr(pack.user[:300]) if pack.user else "None"
        logger.info(
            f"[GigaEvoArchivePromptFetcher] Sampled: {chosen_pid[:8]} "
            f"fitness={chosen_fitness:.4f} prompt_id={pack.prompt_id} "
            f"has_user={pack.user is not None} "
            f"(from {len(candidates)} candidates)\n"
            f"  SYSTEM[:{min(300, len(pack.system))}]: "
            f"{pack.system[:300]!r}\n"
            f"  USER[:{min(300, len(pack.user)) if pack.user else 0}]: "
            f"{user_preview}"
        )
        return pack

    def _execute_entrypoint(self, code: str) -> _PromptPack | None:
        """Execute a program's entrypoint() in a clean namespace.

        Computes prompt_id from the system prompt TEXT (not the program UUID)
        so it matches the ID used by PromptFitnessStage on the read side.

        Args:
            code: Python source code with entrypoint() function that returns
                  either a str (system prompt only) or a dict with keys
                  "system" (required) and "user" (optional).

        Returns:
            _PromptPack with system/user texts and text-derived prompt_id, or None on error
        """
        try:
            namespace: dict[str, Any] = {}
            exec(compile(code, "<prompt_program>", "exec"), namespace)  # noqa: S102
            entrypoint_fn = namespace.get("entrypoint")
            if not callable(entrypoint_fn):
                logger.warning(
                    "[GigaEvoArchivePromptFetcher] Champion code has no callable entrypoint()"
                )
                return None
            result = entrypoint_fn()
            if isinstance(result, str):
                if not result.strip():
                    logger.warning(
                        "[GigaEvoArchivePromptFetcher] entrypoint() returned empty string"
                    )
                    return None
                pid = prompt_text_to_id(result)
                return _PromptPack(system=result, user=None, prompt_id=pid)
            elif isinstance(result, dict):
                system = result.get("system", "")
                if not isinstance(system, str) or not system.strip():
                    logger.warning(
                        "[GigaEvoArchivePromptFetcher] dict entrypoint() missing valid 'system' key"
                    )
                    return None
                user = result.get("user")
                if user is not None and (not isinstance(user, str) or not user.strip()):
                    logger.warning(
                        "[GigaEvoArchivePromptFetcher] dict entrypoint() has invalid 'user' key — ignoring"
                    )
                    user = None
                pid = prompt_text_to_id(system, user_text=user)
                return _PromptPack(system=system, user=user, prompt_id=pid)
            else:
                logger.warning(
                    f"[GigaEvoArchivePromptFetcher] entrypoint() returned {type(result)}, "
                    f"expected str or dict"
                )
                return None
        except Exception as exc:
            logger.warning(
                f"[GigaEvoArchivePromptFetcher] entrypoint() execution error: {exc}"
            )
            return None

    def fetch(self, agent_name: str, prompt_type: str) -> FetchedPrompt:
        """Fetch the current champion's prompt, falling back to fixed if unavailable.

        For "mutation" agent, serves co-evolved system and user prompts from the
        champion pack. Only mutation prompts are tracked; all others use fallback.

        Args:
            agent_name: Agent type (only "mutation" prompts are co-evolved)
            prompt_type: "system" or "user"

        Returns:
            FetchedPrompt with champion text and tracking ID, or fallback
        """
        # Only co-evolve mutation prompts
        if agent_name != "mutation":
            return self._fallback.fetch(agent_name, prompt_type)

        # Refresh candidates list if cache is stale (expensive Redis read)
        if self._is_cache_stale():
            new_candidates = self._refresh_candidates()
            if new_candidates is not None:
                self._cached_candidates = new_candidates
            self._cache_timestamp = time.monotonic()

        # Sample fresh on "system" (first call per mutation), reuse on "user"
        if prompt_type == "system":
            self._system_fetch_count += 1
            if self._system_fetch_count % 50 == 0:
                logger.info(
                    "[GigaEvoArchivePromptFetcher] Health | fetches={} archive_hits={} "
                    "errors={} candidates={} using_archive={}",
                    self._system_fetch_count,
                    self._cache_hits,
                    self._fetch_errors,
                    len(self._cached_candidates) if self._cached_candidates else 0,
                    self._using_archive,
                )
            pack = self._sample_prompt()
            if pack is not None:
                _CURRENT_PACK.set(pack)
                self._cached_pack = pack  # track last sample for get_stats()
        pack = _CURRENT_PACK.get()

        if pack is not None:
            if not self._using_archive:
                self._using_archive = True
                logger.info(
                    "[GigaEvoArchivePromptFetcher] TRANSITION: first champion "
                    "available, switching from fallback to co-evolved prompts",
                )
            self._cache_hits += 1
            if prompt_type == "system":
                return FetchedPrompt(
                    text=pack.system,
                    prompt_id=pack.prompt_id,
                )
            elif prompt_type == "user" and pack.user is not None:
                return FetchedPrompt(
                    text=pack.user,
                    prompt_id=pack.prompt_id,
                )
            # user prompt not co-evolved yet — fall through to fallback

        # No champion yet or no user in pack: use fallback
        if self._using_archive and prompt_type == "system":
            logger.warning(
                "[GigaEvoArchivePromptFetcher] Archive lost — reverting to fallback",
            )
            self._using_archive = False
        return self._fallback.fetch(agent_name, prompt_type)

    async def record_outcome(
        self,
        prompt_id: str | None,
        child_fitness: float,
        parent_fitness: float,
        higher_is_better: bool,
        outcome: MutationOutcome,
        child_metrics: dict[str, float] | None = None,
    ) -> None:
        """Record one mutation outcome through the dataplane.

        See the module docstring for the key layout. Skips
        ``REJECTED_ACCEPTOR`` outcomes; no-op when no main DataPlane
        is configured.
        """
        if prompt_id is None:
            return

        from gigaevo.llm.bandit import MutationOutcome as _MutationOutcome

        if outcome == _MutationOutcome.REJECTED_ACCEPTOR:
            return

        if self._main_dp is None:
            logger.debug(
                "[GigaEvoArchivePromptFetcher] No main DataPlane configured for stats write"
            )
            return

        is_improvement = (
            (child_fitness > parent_fitness)
            if higher_is_better
            else (child_fitness < parent_fitness)
        )

        dp_ = self._main_dp
        actor = self._actor
        try:
            await dp_.crdt_inc(_trials_key(prompt_id), actor=actor, delta=1)
            if is_improvement:
                await dp_.crdt_inc(_successes_key(prompt_id), actor=actor, delta=1)
            await dp_.bounded_list_push(
                _fitness_list_key(prompt_id),
                round(float(child_fitness), 4),
                cap=_FITNESS_WINDOW,
            )
            if child_metrics:
                await dp_.crdt_inc(_metrics_count_key(prompt_id), actor=actor, delta=1)
                metric_dir = _metric_directory_key(prompt_id)
                for k, v in child_metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    fixed = _metric_to_fixed_point(float(v))
                    if fixed == 0 and v != 0:
                        # Non-zero value coerced to zero (out of range);
                        # skip the write rather than corrupt the sum.
                        continue
                    await dp_.crdt_inc(
                        _metric_sum_key(prompt_id, k), actor=actor, delta=fixed
                    )
                    await dp_.set_add(metric_dir, k)
            logger.debug(
                f"[GigaEvoArchivePromptFetcher] Recorded outcome for {prompt_id}: "
                f"improvement={is_improvement} child_fitness={child_fitness:.4f}"
            )
        except Exception as exc:  # noqa: BLE001 - record outcome boundary
            logger.warning(f"[GigaEvoArchivePromptFetcher] Stats write error: {exc}")

    def get_stats(self) -> dict[str, Any]:
        return {
            "cache_hits": self._cache_hits,
            "fetch_errors": self._fetch_errors,
            "has_champion": self._cached_pack is not None,
            "champion_has_user": (
                self._cached_pack is not None and self._cached_pack.user is not None
            ),
            "n_candidates": (
                len(self._cached_candidates) if self._cached_candidates else 0
            ),
        }
