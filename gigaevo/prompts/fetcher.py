"""PromptFetcher abstraction for decoupling prompt acquisition from LLM agents.

Two concrete implementations:
  - FixedDirPromptFetcher: reads from a directory or package defaults (current behavior)
  - GigaEvoArchivePromptFetcher: reads champion from a co-evolving GigaEvo archive
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.prompts import load_prompt

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.llm.bandit import MutationOutcome


@dataclass
class _PromptPack:
    """Internal holder for co-evolved system + optional user prompt texts."""

    system: str
    user: str | None
    prompt_id: str


@dataclass
class FetchedPrompt:
    """Result of a prompt fetch operation."""

    text: str
    prompt_id: str | None  # None for fixed prompts (no tracking)


class PromptFetcher(ABC):
    """Abstracts how a system/user prompt is obtained by an LLM agent.

    Replaces prompts_dir: str | Path | None throughout agent factories.
    Two concrete implementations:
      - FixedDirPromptFetcher: reads from files (default, current behavior)
      - GigaEvoArchivePromptFetcher: reads champion from a co-evolving GigaEvo archive
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

    def record_outcome(
        self,
        prompt_id: str | None,
        child_fitness: float,
        parent_fitness: float,
        higher_is_better: bool,
        outcome: "MutationOutcome",
    ) -> None:
        """Called after mutation outcome known. Default no-op.

        Args:
            prompt_id: ID of the prompt used (None to skip)
            child_fitness: Fitness of the child program
            parent_fitness: Fitness of the best parent
            higher_is_better: Whether higher fitness is better
            outcome: Outcome enum (ACCEPTED, REJECTED_STRATEGY, REJECTED_ACCEPTOR)
        """

    async def start(self, storage: "ProgramStorage | None" = None) -> None:
        """Optional lifecycle hook called when the fetcher is started."""

    async def stop(self) -> None:
        """Optional lifecycle hook called when the fetcher is stopped."""

    def get_stats(self) -> dict[str, Any]:
        """Return stats dict for logging/monitoring."""
        return {}


class FixedDirPromptFetcher(PromptFetcher):
    """Reads from a directory or package defaults. Current behavior, backward-compat.

    Caches loaded templates in memory to avoid repeated disk reads.
    """

    def __init__(self, prompts_dir: str | Path | None = None):
        self._prompts_dir = prompts_dir
        self._cache: dict[tuple[str, str], FetchedPrompt] = {}

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
      Writes {successes, trials} stats to Redis so PromptFitnessStage
      can compute fitness for the prompt run.
      Skips REJECTED_ACCEPTOR outcomes (no reliable fitness).

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
        host: str = "localhost",
        port: int = 6379,
        cache_ttl_seconds: float = 30.0,
        fallback_prompts_dir: str | Path | None = None,
        fitness_key: str = "fitness",
    ):
        self._prompt_redis_db = prompt_redis_db
        self._main_redis_prefix = main_redis_prefix
        self._prompt_prefix = prompt_prefix
        self._host = host
        self._port = port
        self._cache_ttl = cache_ttl_seconds
        self._fallback = FixedDirPromptFetcher(fallback_prompts_dir)
        self._fitness_key = fitness_key

        # Cache state
        self._cached_pack: "_PromptPack | None" = None
        self._cache_timestamp: float = 0.0

        # Lazy-imported redis client for archive reads (prompt run DB)
        self._redis_sync: Any = None

        # Synchronous Redis client for main run stats writes.
        # Initialized immediately if main_redis_db is provided.
        if main_redis_db is not None:
            import redis as sync_redis

            self._redis_main_sync: Any = sync_redis.Redis(
                host=host,
                port=port,
                db=main_redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        else:
            self._redis_main_sync = None

        self._fetch_errors: int = 0
        self._cache_hits: int = 0

    def _get_sync_redis(self) -> Any:
        """Lazy-create synchronous Redis client for archive reads."""
        if self._redis_sync is None:
            import redis as sync_redis

            self._redis_sync = sync_redis.Redis(
                host=self._host,
                port=self._port,
                db=self._prompt_redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        return self._redis_sync

    def _is_cache_stale(self) -> bool:
        return (time.monotonic() - self._cache_timestamp) >= self._cache_ttl

    def _refresh_champion(self) -> "_PromptPack | None":
        """Read the current champion from the prompt run's Redis archive.

        Returns:
            _PromptPack if a champion was found, None if archive is empty
        """
        try:
            r = self._get_sync_redis()

            # Get all elite program IDs from archive hash
            archive_key = f"{self._prompt_prefix}:archive"
            program_ids = list(r.hvals(archive_key))
            if not program_ids:
                logger.debug(
                    "[GigaEvoArchivePromptFetcher] Archive empty, using fallback"
                )
                return None

            # Fetch all programs and find the champion
            best_program_id: str | None = None
            best_fitness: float = float("-inf")
            best_code: str | None = None

            for pid in program_ids:
                program_key = f"{self._prompt_prefix}:program:{pid}"
                raw = r.get(program_key)
                if not raw:
                    continue
                try:
                    import json

                    data = json.loads(raw)
                    metrics = data.get("metrics", {})
                    fitness = float(metrics.get(self._fitness_key, float("-inf")))
                    code = data.get("code", "")
                    if fitness > best_fitness and code:
                        best_fitness = fitness
                        best_program_id = pid
                        best_code = code
                except Exception as exc:
                    logger.debug(
                        f"[GigaEvoArchivePromptFetcher] Error parsing program {pid}: {exc}"
                    )
                    continue

            if best_code is None or best_program_id is None:
                return None

            # Execute the champion's entrypoint() to get the prompt pack
            prompt_id = hashlib.sha256(best_program_id.encode()).hexdigest()[:16]
            pack = self._execute_entrypoint(best_code, prompt_id)
            if pack is None:
                return None

            logger.debug(
                f"[GigaEvoArchivePromptFetcher] Champion: {best_program_id[:8]} "
                f"fitness={best_fitness:.4f} prompt_id={prompt_id} "
                f"has_user={pack.user is not None}"
            )
            return pack

        except Exception as exc:
            self._fetch_errors += 1
            logger.warning(
                f"[GigaEvoArchivePromptFetcher] Archive read error (#{self._fetch_errors}): {exc}"
            )
            return None

    def _execute_entrypoint(self, code: str, prompt_id: str) -> "_PromptPack | None":
        """Execute a program's entrypoint() in a clean namespace.

        Args:
            code: Python source code with entrypoint() function that returns
                  either a str (system prompt only) or a dict with keys
                  "system" (required) and "user" (optional).
            prompt_id: Pre-computed prompt_id to attach to the resulting pack.

        Returns:
            _PromptPack with system/user texts, or None on error
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
                return _PromptPack(system=result, user=None, prompt_id=prompt_id)
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
                return _PromptPack(system=system, user=user, prompt_id=prompt_id)
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

        # Refresh champion pack if cache is stale
        if self._is_cache_stale():
            new_pack = self._refresh_champion()
            if new_pack is not None:
                self._cached_pack = new_pack
            self._cache_timestamp = time.monotonic()

        if self._cached_pack is not None:
            self._cache_hits += 1
            if prompt_type == "system":
                return FetchedPrompt(
                    text=self._cached_pack.system,
                    prompt_id=self._cached_pack.prompt_id,
                )
            elif prompt_type == "user" and self._cached_pack.user is not None:
                return FetchedPrompt(
                    text=self._cached_pack.user,
                    prompt_id=self._cached_pack.prompt_id,
                )
            # user prompt not co-evolved yet — fall through to fallback

        # No champion yet or no user in pack: use fallback
        return self._fallback.fetch(agent_name, prompt_type)

    def record_outcome(
        self,
        prompt_id: str | None,
        child_fitness: float,
        parent_fitness: float,
        higher_is_better: bool,
        outcome: "MutationOutcome",
    ) -> None:
        """Write mutation outcome stats to Redis for the prompt run to read.

        Skips REJECTED_ACCEPTOR (no reliable fitness).

        Args:
            prompt_id: Tracking ID of the prompt used
            child_fitness: Fitness of the resulting program
            parent_fitness: Best parent fitness
            higher_is_better: Whether higher fitness is better
            outcome: Mutation outcome
        """
        if prompt_id is None:
            return

        from gigaevo.llm.bandit import MutationOutcome as _MutationOutcome

        if outcome == _MutationOutcome.REJECTED_ACCEPTOR:
            return  # No reliable fitness — skip

        if self._redis_main_sync is None:
            logger.debug(
                "[GigaEvoArchivePromptFetcher] No main Redis configured for stats write"
            )
            return

        try:
            import json as _json

            stats_key = f"{self._main_redis_prefix}:prompt_stats:{prompt_id}"
            raw = self._redis_main_sync.get(stats_key)
            if raw:
                stats = _json.loads(raw)
            else:
                stats = {"trials": 0, "successes": 0}

            stats["trials"] += 1
            is_improvement = (
                (child_fitness > parent_fitness)
                if higher_is_better
                else (child_fitness < parent_fitness)
            )
            if is_improvement:
                stats["successes"] += 1

            self._redis_main_sync.set(stats_key, _json.dumps(stats))
            logger.debug(
                f"[GigaEvoArchivePromptFetcher] Stats updated for {prompt_id}: "
                f"trials={stats['trials']} successes={stats['successes']}"
            )
        except Exception as exc:
            logger.warning(f"[GigaEvoArchivePromptFetcher] Stats write error: {exc}")

    def get_stats(self) -> dict[str, Any]:
        return {
            "cache_hits": self._cache_hits,
            "fetch_errors": self._fetch_errors,
            "has_champion": self._cached_pack is not None,
            "champion_has_user": (
                self._cached_pack is not None and self._cached_pack.user is not None
            ),
        }
