"""Bandit-based adaptive model selection for LLM ensembles.

Implements UCB1 with sliding window and running-percentile reward normalization,
inspired by ShinkaEvolve (arxiv 2509.19349).  The ``BanditModelRouter`` subclass
of ``MultiModelRouter`` replaces static probability-based selection with an
adaptive strategy that learns which LLM produces the best fitness improvements.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from loguru import logger
import numpy as np

from gigaevo.llm.models import MultiModelRouter, _StructuredOutputRouter
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


# ---------------------------------------------------------------------------
# Mutation outcome discriminator
# ---------------------------------------------------------------------------


class MutationOutcome(Enum):
    """Outcome of a mutated program after DAG evaluation."""

    ACCEPTED = "accepted"  # entered archive
    REJECTED_STRATEGY = "rejected_strategy"  # valid but not good enough
    REJECTED_ACCEPTOR = "rejected_acceptor"  # crashed/invalid, no reliable fitness


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

#: Cap improvement before ``exp()`` to prevent overflow.  ``exp(20) ≈ 4.85e8``
#: is already enormous; the normalizer clips anything above p95 to 1.0 anyway.
_MAX_IMPROVEMENT: float = 20.0


def compute_bandit_reward(
    child_fitness: float,
    best_parent_fitness: float,
    higher_is_better: bool = True,
) -> float:
    """Compute raw bandit reward from fitness improvement.

    Uses ``exp(min(max(improvement, 0), _MAX_IMPROVEMENT)) - 1`` so improvements
    are always non-negative and super-linear in magnitude (ShinkaEvolve formula).
    The upper clamp prevents overflow from pathological fitness values (e.g.
    sentinel values for crashed programs).

    Args:
        child_fitness: Fitness of the child program.
        best_parent_fitness: Best fitness among the parent programs.
        higher_is_better: Whether higher fitness is better.

    Returns:
        Non-negative raw reward, capped at ``exp(_MAX_IMPROVEMENT) - 1``.
    """
    improvement = child_fitness - best_parent_fitness
    if not higher_is_better:
        improvement = -improvement
    clamped = min(max(improvement, 0.0), _MAX_IMPROVEMENT)
    return math.exp(clamped) - 1.0


# ---------------------------------------------------------------------------
# Running-percentile normalizer
# ---------------------------------------------------------------------------


class RunningPercentileNormalizer:
    """Normalize raw rewards to [0, 1] using a running percentile reference.

    During warmup (fewer than ``min_samples`` observations) the normalizer
    returns 0.5 (neutral) to avoid noisy early estimates.

    Args:
        percentile: The reference percentile for normalization (default 95).
        min_samples: Minimum observations before real normalization begins.
        max_samples: Maximum stored observations (oldest evicted). ``None`` for unbounded.
    """

    def __init__(
        self,
        percentile: float = 95.0,
        min_samples: int = 10,
        max_samples: int = 1000,
    ):
        self.percentile = percentile
        self.min_samples = min_samples
        self._rewards: deque[float] = deque(maxlen=max_samples)

    def normalize(self, reward: float) -> float:
        """Normalize *reward* to [0, 1]."""
        self._rewards.append(reward)
        if len(self._rewards) < self.min_samples:
            return 0.5
        p = float(np.percentile(self._rewards, self.percentile))
        if p <= 0:
            return 0.5
        return float(np.clip(reward / p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Sliding-window UCB1
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    """Per-arm statistics for the sliding-window UCB1 bandit."""

    rewards: deque[float]
    total_pulls: int = 0


@dataclass
class SlidingWindowUCB1:
    """Upper Confidence Bound (UCB1) bandit with a sliding reward window.

    Arms that have never been pulled are selected first (round-robin warmup).
    After warmup, the arm with the highest UCB1 score is selected:

        UCB1 = mean(windowed_rewards) + c * sqrt(ln(N) / n_i)

    Args:
        arm_names: Names identifying each arm.
        exploration_constant: UCB1 exploration parameter *c*.
        window_size: Maximum number of recent rewards kept per arm.
    """

    arm_names: list[str]
    exploration_constant: float = 1.41
    window_size: int = 100
    arms: dict[str, ArmStats] = field(init=False)
    _total_pulls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.arms = {
            name: ArmStats(rewards=deque(maxlen=self.window_size))
            for name in self.arm_names
        }

    def select(self) -> str:
        """Return the arm name with the highest UCB1 score.

        The exploitation term (mean reward) uses the sliding reward window
        for recency adaptation.  The exploration term uses ``total_pulls``
        so that every pull — rewarded or not — correctly reduces the
        confidence bonus for that arm.
        """
        # Warmup: round-robin — pull each arm at least once.
        for name, stats in self.arms.items():
            if stats.total_pulls == 0:
                return name

        best_name = self.arm_names[0]
        best_score = -math.inf
        for name, stats in self.arms.items():
            # Exploitation: mean of recent (windowed) rewards.
            window_n = len(stats.rewards)
            mean_reward = sum(stats.rewards) / window_n if window_n else 0.0
            # Exploration: uses total pulls (not window size) so that
            # unrewarded pulls still decrease this arm's bonus.
            n_i = stats.total_pulls  # guaranteed >= 1 after warmup
            exploration = self.exploration_constant * math.sqrt(
                math.log(self._total_pulls) / n_i
            )
            score = mean_reward + exploration
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def record_pull(self, arm_name: str) -> None:
        """Increment pull counter for *arm_name*."""
        self.arms[arm_name].total_pulls += 1
        self._total_pulls += 1

    def update_reward(self, arm_name: str, reward: float) -> None:
        """Append *reward* to the sliding window for *arm_name*."""
        self.arms[arm_name].rewards.append(reward)

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-friendly summary of per-arm statistics."""
        out: dict[str, dict[str, Any]] = {}
        for name, stats in self.arms.items():
            rewards = list(stats.rewards)
            out[name] = {
                "total_pulls": stats.total_pulls,
                "window_size": len(rewards),
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            }
        return out


# ---------------------------------------------------------------------------
# BanditModelRouter
# ---------------------------------------------------------------------------


class BanditModelRouter(MultiModelRouter):
    """Adaptive model router using UCB1 bandit selection.

    Replaces the static probability-based selection of ``MultiModelRouter``
    with a sliding-window UCB1 strategy that learns from mutation outcomes.

    Reward is deferred: ``_select()`` picks a model and records the pull;
    ``on_mutation_outcome()`` is called later (after DAG evaluation) with the
    resulting fitness to update the bandit.

    Args:
        models: List of LLM model instances.
        probabilities: Initial probabilities (used only by the base class as fallback).
        writer: Optional metrics writer.
        name: Router name for logging/metrics.
        exploration_constant: UCB1 exploration parameter *c*.
        window_size: Number of recent rewards kept per arm.
        fitness_key: Metric key used to read fitness from ``Program.metrics``.
        higher_is_better: Whether higher fitness values are better.
    """

    def __init__(
        self,
        models: list[ChatOpenAI],
        probabilities: list[float],
        *,
        writer: LogWriter | None = None,
        name: str = "default",
        exploration_constant: float = 1.41,
        window_size: int = 100,
        fitness_key: str,
        higher_is_better: bool = True,
    ):
        super().__init__(models, probabilities, writer=writer, name=name)
        self.fitness_key = fitness_key
        self.higher_is_better = higher_is_better
        self._bandit = SlidingWindowUCB1(
            arm_names=self.model_names,
            exploration_constant=exploration_constant,
            window_size=window_size,
        )
        self._reward_normalizer = RunningPercentileNormalizer()
        logger.info(
            "[BanditModelRouter:{}] UCB1 bandit enabled | arms={} c={} W={}",
            name,
            self.model_names,
            exploration_constant,
            window_size,
        )

    # -- selection ----------------------------------------------------------

    def _select(self) -> tuple[ChatOpenAI, str]:
        """Select a model via UCB1 and record the pull."""
        name = self._bandit.select()
        self._bandit.record_pull(name)
        tid = self._current_task_id()
        if tid is not None:
            self._task_model_map[tid] = name
        idx = self.model_names.index(name)
        return self.models[idx], name

    # -- mutation outcome ---------------------------------------------------

    def on_mutation_outcome(
        self,
        program: Program,
        parents: list[Program],
        outcome: MutationOutcome = MutationOutcome.ACCEPTED,
    ) -> None:
        """Update the bandit with the reward from a completed mutation.

        Called for **every** mutation outcome — accepted, rejected by strategy,
        or rejected by acceptor — so the bandit sees the full distribution of
        each model's outputs.
        """
        model_name = program.get_metadata("mutation_model")
        if not model_name:
            return

        if outcome == MutationOutcome.REJECTED_ACCEPTOR:
            # No reliable fitness — inject zero reward directly.
            normalized = self._reward_normalizer.normalize(0.0)
            self._bandit.update_reward(model_name, normalized)
            logger.debug(
                "[BanditModelRouter] Reward for {} ({}): raw=0.0 norm={:.4f}",
                model_name,
                outcome.value,
                normalized,
            )
            return

        # For ACCEPTED and REJECTED_STRATEGY: compute from actual fitness.
        child_f = program.metrics.get(self.fitness_key)
        if child_f is None:
            # Fitness missing despite passing acceptor — treat as failure.
            normalized = self._reward_normalizer.normalize(0.0)
            self._bandit.update_reward(model_name, normalized)
            return

        parent_fs = [
            p.metrics[self.fitness_key]
            for p in parents
            if self.fitness_key in p.metrics
        ]
        if not parent_fs:
            normalized = self._reward_normalizer.normalize(0.0)
            self._bandit.update_reward(model_name, normalized)
            return

        best_parent = max(parent_fs) if self.higher_is_better else min(parent_fs)
        raw = compute_bandit_reward(child_f, best_parent, self.higher_is_better)
        normalized = self._reward_normalizer.normalize(raw)
        self._bandit.update_reward(model_name, normalized)
        logger.debug(
            "[BanditModelRouter] Reward for {} ({}): raw={:.4f} norm={:.4f}",
            model_name,
            outcome.value,
            raw,
            normalized,
        )

    # -- stats --------------------------------------------------------------

    def get_bandit_stats(self) -> dict[str, dict[str, Any]]:
        """Return per-arm bandit statistics."""
        return self._bandit.get_stats()

    # -- structured output --------------------------------------------------

    def with_structured_output(self, schema: Any, **kwargs) -> _StructuredOutputRouter:
        """Create a structured-output router that delegates selection to the bandit."""
        wrapped = [
            m.with_structured_output(schema, include_raw=True, **kwargs)
            for m in self.models
        ]

        def _bandit_select() -> tuple[Any, str]:
            name = self._bandit.select()
            self._bandit.record_pull(name)
            tid = self._current_task_id()
            if tid is not None:
                self._task_model_map[tid] = name
            idx = self.model_names.index(name)
            return wrapped[idx], name

        return _StructuredOutputRouter(
            wrapped,
            self.model_names,
            self.probabilities,
            self._langfuse,
            self._tracker,
            task_model_map=self._task_model_map,
            select_override=_bandit_select,
        )
