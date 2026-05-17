"""Prompt mutation statistics for co-evolution feedback.

The main run writes per-prompt success/trial counts; the prompt run
reads them to compute fitness for its MAP-Elites archive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Final

from loguru import logger

from gigaevo.dataplane import CounterKey, DataPlane, Err, Ok

_RECENT_FITNESS_WINDOW: Final[int] = 20
"""Cap on entries returned in ``PromptMutationStats.recent_fitnesses``.
Per-source lists are also writer-capped; this bounds the multi-source
concatenation independently."""

_METRIC_FIXED_POINT_SCALE: Final[int] = 1000
"""Mirror of :data:`gigaevo.prompts.fetcher._METRIC_FIXED_POINT_SCALE`.
The reader divides the cross-actor sum by
``metrics_count * _METRIC_FIXED_POINT_SCALE`` to recover the mean."""


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
    """Reads prompt stats written by main run(s) via :class:`DataPlane`.

    Aggregates trials/successes across multiple main runs that share a
    prompt evolution archive. Each source carries an independent
    :class:`DataPlane` handle bound to its DB + prefix.

    Args:
        host: Redis host
        port: Redis port
        db: Redis DB of a single main run.
        prefix: Key prefix of a single main run.
        sources: List of ``{"db", "prefix"}`` dicts for multi-source
            aggregation; takes precedence over ``db`` / ``prefix``.
        min_trials: Minimum trials before reporting a real success rate.
        dataplanes: Optional pre-wired :class:`DataPlane` handles (one
            per source); length must match ``sources``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int | None = None,
        prefix: str | None = None,
        sources: list[dict[str, int | str]] | None = None,
        min_trials: int = 5,
        *,
        dataplanes: list[DataPlane] | None = None,
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

        if dataplanes is not None:
            if len(dataplanes) != len(self._sources):
                raise ValueError(
                    "RedisPromptStatsProvider: dataplanes length "
                    f"({len(dataplanes)}) must match sources length "
                    f"({len(self._sources)})"
                )
            self._dataplanes: list[DataPlane | None] = list(dataplanes)
            self._dp_owned: list[bool] = [False] * len(dataplanes)
        else:
            self._dataplanes = [None] * len(self._sources)
            self._dp_owned = [False] * len(self._sources)

    async def _get_dataplane(self, idx: int) -> DataPlane:
        """Resolve (and lazily construct) the DataPlane for source ``idx``."""
        dp = self._dataplanes[idx]
        if dp is not None:
            return dp
        db, prefix = self._sources[idx]
        url = f"redis://{self._host}:{self._port}/{db}"
        dp = DataPlane(url, key_prefix=prefix)
        await dp.startup()
        self._dataplanes[idx] = dp
        self._dp_owned[idx] = True
        return dp

    async def close(self) -> None:
        """Tear down DataPlane handles the provider constructed itself.

        Injected handles are left alone. Idempotent.
        """
        for i, dp in enumerate(self._dataplanes):
            if dp is None or not self._dp_owned[i]:
                continue
            try:
                await dp.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                logger.warning(
                    "[RedisPromptStatsProvider] DataPlane[{}] shutdown failed: {}",
                    i,
                    exc,
                )
            self._dataplanes[i] = None
            self._dp_owned[i] = False

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
        merged_metrics_sums: dict[str, int] = {}

        for idx in range(len(self._sources)):
            try:
                dp = await self._get_dataplane(idx)
            except Exception as exc:  # noqa: BLE001 - read boundary
                logger.warning(
                    "[RedisPromptStatsProvider] DataPlane[{}] startup failed: {}",
                    idx,
                    exc,
                )
                continue
            (
                trials,
                successes,
                metrics_count,
                fitnesses,
                per_metric,
            ) = await _read_one_source(dp, prompt_id)
            total_trials += trials
            total_successes += successes
            total_metrics_count += metrics_count
            all_fitnesses.extend(fitnesses)
            for k, v in per_metric.items():
                merged_metrics_sums[k] = merged_metrics_sums.get(k, 0) + v

        if total_trials < self._min_trials:
            success_rate = 0.0
        else:
            success_rate = total_successes / total_trials if total_trials > 0 else 0.0

        mean_child_fitness = (
            sum(all_fitnesses) / len(all_fitnesses) if all_fitnesses else 0.0
        )
        # Per-source lists are newest-first (LPUSH + LTRIM); take the
        # first _RECENT_FITNESS_WINDOW across the concatenation.
        recent = all_fitnesses[:_RECENT_FITNESS_WINDOW] if all_fitnesses else None

        # Fixed-point sums divided by metrics_count * scale → mean.
        mean_metrics: dict[str, float] | None = None
        if total_metrics_count > 0 and merged_metrics_sums:
            divisor = total_metrics_count * _METRIC_FIXED_POINT_SCALE
            mean_metrics = {
                k: round(v / divisor, 4) for k, v in merged_metrics_sums.items()
            }

        return PromptMutationStats(
            trials=total_trials,
            successes=total_successes,
            success_rate=success_rate,
            mean_child_fitness=round(mean_child_fitness, 4),
            recent_fitnesses=recent,
            mean_metrics=mean_metrics,
        )


async def _read_one_source(
    dp: DataPlane, prompt_id: str
) -> tuple[int, int, int, list[float], dict[str, int]]:
    """Read ``(trials, successes, metrics_count, fitnesses, per_metric)``.

    Each field defaults to its empty value on a per-key miss; an unknown
    prompt id surfaces as zeros rather than an exception.
    """
    from gigaevo.prompts.fetcher import (
        _fitness_list_key,
        _metric_directory_key,
        _metric_sum_key,
        _metrics_count_key,
        _successes_key,
        _trials_key,
    )

    async def _counter(key: CounterKey) -> int:
        result = await dp.crdt_read(key)
        if isinstance(result, Err):
            logger.debug(
                "[RedisPromptStatsProvider] crdt_read({}) error: {}",
                key,
                result.error,
            )
            return 0
        return int(result.value.value)

    trials = await _counter(_trials_key(prompt_id))
    successes = await _counter(_successes_key(prompt_id))
    metrics_count = await _counter(_metrics_count_key(prompt_id))

    fitnesses: list[float] = []
    fitness_result = await dp.bounded_list_range(
        _fitness_list_key(prompt_id), count=_RECENT_FITNESS_WINDOW
    )
    if isinstance(fitness_result, Ok):
        for raw in fitness_result.value:
            try:
                fitnesses.append(float(raw))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue

    per_metric: dict[str, int] = {}
    if metrics_count > 0:
        members_result = await dp.set_members(_metric_directory_key(prompt_id))
        if isinstance(members_result, Ok):
            for metric_key in members_result.value:
                value = await _counter(_metric_sum_key(prompt_id, metric_key))
                if value:
                    per_metric[metric_key] = value

    return trials, successes, metrics_count, fitnesses, per_metric


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
