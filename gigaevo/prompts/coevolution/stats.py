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
    """Reads prompt stats written by the main run to its Redis DB.

    Stats are written by GigaEvoArchivePromptFetcher.record_outcome() and
    stored as JSON under the key:
        {prefix}:prompt_stats:{prompt_id}

    All coupling is through injected constructor arguments — no global variables.

    Args:
        host: Redis host
        port: Redis port
        db: Redis DB of the main run
        prefix: Key prefix of the main run (e.g. "chains/hotpotqa")
        min_trials: Minimum trials before reporting real success rate (avoids
            cold-start noise); returns 0.0 success_rate if below threshold
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        prefix: str,
        min_trials: int = 5,
    ):
        self._host = host
        self._port = port
        self._db = db
        self._prefix = prefix
        self._min_trials = min_trials
        self._redis: object | None = None

    def _get_redis(self) -> object:
        if self._redis is None:
            from redis import asyncio as aioredis

            self._redis = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True,
            )
        return self._redis

    async def get_stats(self, prompt_id: str) -> PromptMutationStats:
        """Fetch mutation stats for the given prompt ID from Redis.

        Args:
            prompt_id: Short hex ID of the prompt

        Returns:
            PromptMutationStats; success_rate=0.0 if insufficient trials
        """
        try:
            r = self._get_redis()
            key = f"{self._prefix}:prompt_stats:{prompt_id}"
            raw = await r.get(key)  # type: ignore[attr-defined]
            if not raw:
                return PromptMutationStats(trials=0, successes=0, success_rate=0.0)
            data = json.loads(raw)
            trials = int(data.get("trials", 0))
            successes = int(data.get("successes", 0))
            if trials < self._min_trials:
                success_rate = 0.0
            else:
                success_rate = successes / trials if trials > 0 else 0.0
            return PromptMutationStats(
                trials=trials, successes=successes, success_rate=success_rate
            )
        except Exception as exc:
            logger.warning(
                f"[RedisPromptStatsProvider] Error reading stats for {prompt_id}: {exc}"
            )
            return PromptMutationStats(trials=0, successes=0, success_rate=0.0)


def prompt_text_to_id(prompt_text: str) -> str:
    """Compute a stable short ID for a prompt text string.

    Args:
        prompt_text: The prompt text

    Returns:
        16-char hex string (sha256[:16])
    """
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
