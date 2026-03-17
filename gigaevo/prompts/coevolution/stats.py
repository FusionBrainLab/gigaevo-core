"""Prompt mutation statistics for co-evolution feedback.

The main run writes per-prompt success/trial counts; the prompt run
reads them to compute fitness for its MAP-Elites archive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json

from loguru import logger


@dataclass
class PromptMutationStats:
    """Per-prompt mutation outcome statistics."""

    trials: int
    successes: int
    success_rate: float  # 0.0 when trials < min_trials (insufficient data)
    mean_child_fitness: float = 0.0  # average fitness of programs using this prompt
    recent_fitnesses: list[float] | None = None  # last N fitness values
    mean_metrics: dict[str, float] | None = None  # per-metric means (e.g. EM, F1)


class PromptStatsProvider(ABC):
    """Abstract interface for fetching per-prompt mutation statistics."""

    @abstractmethod
    async def get_stats(self, prompt_id: str) -> PromptMutationStats:
        """Fetch stats for the given prompt ID.

        Args:
            prompt_id: Short hex ID (sha256[:16]) of the prompt

        Returns:
            PromptMutationStats with trials, successes, success_rate
        """
        ...


class RedisPromptStatsProvider(PromptStatsProvider):
    """Reads prompt stats written by main run(s) to their Redis DBs.

    Supports 1-to-many coupling: aggregates trials/successes across multiple
    main runs that all test prompts from the same prompt evolution archive.

    Stats are written by GigaEvoArchivePromptFetcher.record_outcome() and
    stored as JSON under the key:
        {prefix}:prompt_stats:{prompt_id}

    Args:
        host: Redis host
        port: Redis port
        db: Redis DB of a single main run (for backwards compat)
        prefix: Key prefix of a single main run (for backwards compat)
        sources: List of {"db": int, "prefix": str} dicts for multi-source
            aggregation. If provided, ``db`` and ``prefix`` are ignored.
        min_trials: Minimum trials before reporting real success rate
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int | None = None,
        prefix: str | None = None,
        sources: list[dict[str, int | str]] | None = None,
        min_trials: int = 5,
    ):
        self._host = host
        self._port = port
        self._min_trials = min_trials

        # Build list of (db, prefix) sources
        if sources:
            self._sources = [(int(s["db"]), str(s["prefix"])) for s in sources]
        elif db is not None and prefix is not None:
            self._sources = [(db, prefix)]
        else:
            raise ValueError(
                "RedisPromptStatsProvider requires either (db, prefix) "
                "or sources=[{db, prefix}, ...]"
            )

        self._redis_clients: dict[int, object] = {}

    def _get_redis(self, db: int) -> object:
        if db not in self._redis_clients:
            from redis import asyncio as aioredis

            self._redis_clients[db] = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=db,
                decode_responses=True,
            )
        return self._redis_clients[db]

    async def get_stats(self, prompt_id: str) -> PromptMutationStats:
        """Fetch and aggregate mutation stats across all source DBs.

        Args:
            prompt_id: Short hex ID of the prompt

        Returns:
            PromptMutationStats with aggregated trials/successes and enriched data
        """
        total_trials = 0
        total_successes = 0
        total_metrics_count = 0
        all_fitnesses: list[float] = []
        merged_metrics_sums: dict[str, float] = {}

        for db, prefix in self._sources:
            try:
                r = self._get_redis(db)
                key = f"{prefix}:prompt_stats:{prompt_id}"
                raw = await r.get(key)  # type: ignore[attr-defined]
                if raw:
                    data = json.loads(raw)
                    total_trials += int(data.get("trials", 0))
                    total_successes += int(data.get("successes", 0))
                    total_metrics_count += int(data.get("metrics_count", 0))
                    all_fitnesses.extend(data.get("fitnesses", []))
                    for k, v in data.get("metrics_sums", {}).items():
                        merged_metrics_sums[k] = merged_metrics_sums.get(
                            k, 0.0
                        ) + float(v)
            except Exception as exc:
                logger.warning(
                    f"[RedisPromptStatsProvider] Error reading stats from "
                    f"db={db} for {prompt_id}: {exc}"
                )

        if total_trials < self._min_trials:
            success_rate = 0.0
        else:
            success_rate = total_successes / total_trials if total_trials > 0 else 0.0

        mean_child_fitness = (
            sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0.0
        )
        # Keep last 20 fitnesses across all sources
        recent = all_fitnesses[-20:] if all_fitnesses else None

        # Compute per-metric means using metrics_count (only trials with metrics)
        mean_metrics = None
        if total_metrics_count > 0 and merged_metrics_sums:
            mean_metrics = {
                k: round(v / total_metrics_count, 4)
                for k, v in merged_metrics_sums.items()
            }

        return PromptMutationStats(
            trials=total_trials,
            successes=total_successes,
            success_rate=success_rate,
            mean_child_fitness=round(mean_child_fitness, 4),
            recent_fitnesses=recent,
            mean_metrics=mean_metrics,
        )


def prompt_text_to_id(prompt_text: str, user_text: str | None = None) -> str:
    """Compute a stable short ID for a prompt text string.

    When ``user_text`` is provided the hash covers both system and user
    text so that two programs with identical system prompts but different
    user prompts receive distinct IDs.

    Args:
        prompt_text: The system prompt text
        user_text: Optional user prompt text

    Returns:
        16-char hex string (sha256[:16])
    """
    blob = prompt_text
    if user_text is not None:
        blob = prompt_text + "\x00" + user_text
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
