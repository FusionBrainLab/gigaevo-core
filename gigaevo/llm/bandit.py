"""Bandit-based adaptive model selection for LLM ensembles.

Implements UCB1 with sliding window and running-percentile reward
normalization, inspired by ShinkaEvolve (arxiv 2509.19349). The
``BanditModelRouter`` subclass of ``MultiModelRouter`` replaces static
probability-based selection with an adaptive strategy that learns which
LLM produces the best fitness improvements.

Reward policy
-------------

``ACCEPTED`` and ``REJECTED_STRATEGY`` outcomes carry a real
fitness-derived reward (normalised improvement over the best parent).
``REJECTED_ACCEPTOR`` has unreliable fitness — the failure is not
necessarily attributable to the model — and is gated on the
``skip_reward_on_acceptor_reject`` constructor flag: when set, no
reward is recorded for that outcome and only the UCB1 exploration term
moves; when unset, a normalised 0.0 is appended for callers that need
the historical signal shape.

Optional Redis-backed ledger
----------------------------

When constructed with a :class:`DataPlane` and an
:class:`ActorIdentity`, the bandit mirrors trial counts and reward
sums into per-arm CRDT G-counters
(``bandit:trials:{arm}`` / ``bandit:reward_sum:{arm}``). Rewards are
scaled by :data:`_REWARD_FIXED_POINT_SCALE` to fit ``HINCRBY``'s
integer contract. The in-process windowed UCB stays authoritative for
selection; :meth:`SlidingWindowUCB1.refresh_from_redis` folds the
consensus trial count back into the local state.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from loguru import logger
import numpy as np

from gigaevo.dataplane import (
    ActorIdentity,
    CounterKey,
    DataPlane,
    Err,
    Freshness,
    FreshnessEventual,
    Ok,
    StaleReadError,
    Token,
)
from gigaevo.llm.models import MultiModelRouter, _StructuredOutputRouter
from gigaevo.utils.trackers.base import LogWriter

if TYPE_CHECKING:
    from gigaevo.dataplane.engine_startup import EngineRoot
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

#: Fixed-point scale for the Redis reward-sum G-counter. Floats in ``[0, 1]``
#: from :meth:`RunningPercentileNormalizer.normalize` are multiplied by this
#: factor and rounded to an integer so ``HINCRBY`` (integer-only) suffices.
#: The doc-prescribed value is 1000 (three decimal digits of precision); the
#: reverse conversion lives in :meth:`SlidingWindowUCB1.get_mean_reward`.
_REWARD_FIXED_POINT_SCALE: int = 1000


def _trials_key(arm: str) -> CounterKey:
    """Return the CRDT G-counter key for an arm's trial count."""
    return CounterKey(f"bandit:trials:{arm}")


def _reward_sum_key(arm: str) -> CounterKey:
    """Return the CRDT G-counter key for an arm's fixed-point reward sum."""
    return CounterKey(f"bandit:reward_sum:{arm}")


def _reward_to_fixed_point(reward: float) -> int:
    """Convert a normalized reward in ``[0, 1]`` to a storable integer.

    Banker's rounding is bias-free across actors. ``reward`` is clamped
    defensively in case the normalizer ever produces out-of-range values.
    """
    clamped = max(0.0, min(1.0, reward))
    return round(clamped * _REWARD_FIXED_POINT_SCALE)


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


class SlidingWindowUCB1:
    """Upper Confidence Bound (UCB1) bandit with a sliding reward window.

    Arms that have never been pulled are selected first (round-robin
    warmup). After warmup the arm with the highest UCB1 score is
    selected: ``UCB1 = mean(windowed_rewards) + c * sqrt(ln(N) / n_i)``.

    Supplying ``dataplane`` and ``actor`` enables a non-blocking CRDT
    mirror of trial counts and reward sums; selection still reads the
    in-process window so the hot path stays sync. Supplying
    ``engine_root`` makes every mirror write carry a per-call token
    split from the engine's counter root.

    Args:
        arm_names: Names identifying each arm.
        exploration_constant: UCB1 exploration parameter *c*.
        window_size: Maximum number of recent rewards kept per arm.
        dataplane: Optional :class:`DataPlane` for the CRDT mirror.
        actor: Optional :class:`ActorIdentity`; must accompany ``dataplane``.
        engine_root: Optional :class:`EngineRoot`; requires the dataplane.
    """

    arm_names: list[str]
    exploration_constant: float
    window_size: int
    arms: dict[str, ArmStats]
    _total_pulls: int

    def __init__(
        self,
        arm_names: list[str],
        exploration_constant: float = 1.41,
        window_size: int = 100,
        *,
        dataplane: DataPlane | None = None,
        actor: ActorIdentity | None = None,
        engine_root: EngineRoot | None = None,
    ) -> None:
        self.arm_names = arm_names
        self.exploration_constant = exploration_constant
        self.window_size = window_size
        self.arms = {
            name: ArmStats(rewards=deque(maxlen=window_size)) for name in arm_names
        }
        self._total_pulls = 0
        if (dataplane is None) != (actor is None):
            raise ValueError(
                "SlidingWindowUCB1: dataplane and actor must be supplied together "
                "or both omitted"
            )
        if engine_root is not None and dataplane is None:
            raise ValueError(
                "SlidingWindowUCB1: engine_root requires a dataplane (and actor)"
            )
        self._dataplane: DataPlane | None = dataplane
        self._actor: ActorIdentity | None = actor
        self._engine_root: EngineRoot | None = engine_root
        # Strong refs to in-flight background increments; without these
        # asyncio.create_task results are GC-eligible and may be cancelled
        # before reaching Redis.
        self._pending_writes: set[asyncio.Task[Any]] = set()

    @property
    def is_redis_backed(self) -> bool:
        """``True`` when both a dataplane and an actor are configured."""
        return self._dataplane is not None and self._actor is not None

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
        """Increment pull counter for *arm_name*.

        Schedules a background ``crdt_inc`` when Redis-backed and a
        running event loop is present; otherwise in-process only.
        """
        self.arms[arm_name].total_pulls += 1
        self._total_pulls += 1
        self._schedule_crdt_inc(_trials_key(arm_name), delta=1)

    def update_reward(self, arm_name: str, reward: float) -> None:
        """Append *reward* to the sliding window for *arm_name*.

        Mirrors the reward into the per-arm sum G-counter when Redis-
        backed; zero-delta bumps are skipped to avoid no-op round trips.
        """
        self.arms[arm_name].rewards.append(reward)
        delta = _reward_to_fixed_point(reward)
        if delta != 0:
            self._schedule_crdt_inc(_reward_sum_key(arm_name), delta=delta)

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

    # -- Redis-backed helpers ---------------------------------------------

    def _schedule_crdt_inc(self, key: CounterKey, *, delta: int) -> None:
        """Fire-and-forget a ``crdt_inc`` on the running loop, if any.

        Sync callers (no running loop) silently fall back to in-process
        only. Background errors are logged and dropped — the in-process
        state remains correct.
        """
        if not self.is_redis_backed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        assert self._dataplane is not None
        assert self._actor is not None
        # Mint the per-call counter token synchronously so the engine
        # root's rotation happens before the task is enqueued; the token
        # is consumed inside ``crdt_inc``.
        counter_token: Token[CounterKey] | None = None
        if self._engine_root is not None:
            counter_token = self._engine_root.split_counter_token(key)
        task = loop.create_task(
            self._do_crdt_inc(self._dataplane, self._actor, key, delta, counter_token)
        )
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)

    @staticmethod
    async def _do_crdt_inc(
        dataplane: DataPlane,
        actor: ActorIdentity,
        key: CounterKey,
        delta: int,
        token: Token[CounterKey] | None,
    ) -> None:
        """Issue one ``crdt_inc`` and log any error without propagating.

        Surfacing the error to the hot path would force every LLM call
        to depend on Redis availability; the bandit degrades gracefully
        on in-process state instead.
        """
        try:
            result = await dataplane.crdt_inc(
                key, actor=actor, delta=delta, token=token
            )
        except Exception as exc:  # noqa: BLE001 — background-task safety net
            logger.warning(
                "[SlidingWindowUCB1] crdt_inc({}, delta={}) raised: {}",
                key,
                delta,
                exc,
            )
            return
        if not isinstance(result, Ok):
            logger.warning(
                "[SlidingWindowUCB1] crdt_inc({}, delta={}) failed: {}",
                key,
                delta,
                result.error,
            )

    async def drain_pending_writes(self) -> None:
        """Await every in-flight background increment.

        Tests pin determinism by waiting for the mirror to settle before
        reading from Redis; production callers may use this as part of a
        graceful shutdown sequence.
        """
        if not self._pending_writes:
            return
        pending = tuple(self._pending_writes)
        await asyncio.gather(*pending, return_exceptions=True)

    async def refresh_from_redis(self) -> None:
        """Fold the Redis consensus into the in-process trial counters.

        Raises each ``arm.total_pulls`` to the cross-actor sum if it
        exceeds the local value. Local-exceeds-consensus is logged and
        the local count is preserved — lowering it would retroactively
        shift past UCB1 scores. Reward windows are not modified.
        """
        if not self.is_redis_backed:
            return
        assert self._dataplane is not None
        await self.drain_pending_writes()
        delta_total = 0
        for name in self.arm_names:
            result = await self._dataplane.crdt_read(_trials_key(name))
            if not isinstance(result, Ok):
                logger.warning(
                    "[SlidingWindowUCB1] crdt_read(trials:{}) failed: {}",
                    name,
                    result.error,
                )
                continue
            consensus = int(result.value.value)
            local = self.arms[name].total_pulls
            if consensus > local:
                delta = consensus - local
                self.arms[name].total_pulls = consensus
                delta_total += delta
            elif consensus < local:
                logger.warning(
                    "[SlidingWindowUCB1] arm {} local pulls {} exceeds Redis "
                    "consensus {} — keeping local count",
                    name,
                    local,
                    consensus,
                )
        self._total_pulls += delta_total

    async def get_mean_reward(
        self,
        arm_name: str,
        *,
        freshness: Freshness | None = None,
    ) -> float:
        """Return the mean reward for *arm_name*, or ``0.0`` if untouched.

        Redis-backed: refreshes trial counters then divides the consensus
        reward sum (descaled from fixed point) by the consensus trial
        count. In-process: mean of the local sliding window.

        ``freshness`` defaults to :class:`FreshnessEventual`; a stricter
        floor propagates :class:`StaleReadError` to the caller.
        """
        if not self.is_redis_backed:
            window = self.arms[arm_name].rewards
            return sum(window) / len(window) if window else 0.0
        assert self._dataplane is not None
        effective_freshness: Freshness = (
            freshness if freshness is not None else FreshnessEventual()
        )
        await self.refresh_from_redis()
        trials_result = await self._dataplane.crdt_read(
            _trials_key(arm_name), freshness=effective_freshness
        )
        sum_result = await self._dataplane.crdt_read(
            _reward_sum_key(arm_name), freshness=effective_freshness
        )
        # Honour the freshness contract: a floor violation is opt-in
        # control flow; default-eventual reads cannot raise StaleReadError.
        for result in (trials_result, sum_result):
            if isinstance(result, Err) and isinstance(result.error, StaleReadError):
                raise result.error
        if not isinstance(trials_result, Ok) or not isinstance(sum_result, Ok):
            window = self.arms[arm_name].rewards
            return sum(window) / len(window) if window else 0.0
        trial_count = int(trials_result.value.value)
        if trial_count <= 0:
            return 0.0
        reward_sum_fp = int(sum_result.value.value)
        return reward_sum_fp / trial_count / _REWARD_FIXED_POINT_SCALE


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
        skip_reward_on_acceptor_reject: When ``True``, ``REJECTED_ACCEPTOR``
            outcomes do not append a synthetic 0.0 to the reward window.
        dataplane: Optional :class:`DataPlane` for the CRDT mirror.
        actor: Optional :class:`ActorIdentity`; must accompany ``dataplane``.
        engine_root: Optional :class:`EngineRoot`; requires the dataplane.
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
        skip_reward_on_acceptor_reject: bool = False,
        dataplane: DataPlane | None = None,
        actor: ActorIdentity | None = None,
        engine_root: EngineRoot | None = None,
    ):
        super().__init__(models, probabilities, writer=writer, name=name)
        self.fitness_key = fitness_key
        self.higher_is_better = higher_is_better
        self._skip_reward_on_acceptor_reject = skip_reward_on_acceptor_reject
        self._bandit = SlidingWindowUCB1(
            arm_names=self.model_names,
            exploration_constant=exploration_constant,
            window_size=window_size,
            dataplane=dataplane,
            actor=actor,
            engine_root=engine_root,
        )
        self._reward_normalizer = RunningPercentileNormalizer()
        logger.info(
            "[BanditModelRouter:{}] UCB1 bandit enabled | arms={} c={} W={} "
            "skip_reward_on_acceptor_reject={}",
            name,
            self.model_names,
            exploration_constant,
            window_size,
            skip_reward_on_acceptor_reject,
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

        ``ACCEPTED`` / ``REJECTED_STRATEGY`` use the fitness-derived
        reward. ``REJECTED_ACCEPTOR`` is skipped or zero-recorded based
        on :attr:`_skip_reward_on_acceptor_reject`. Programs without
        ``mutation_model`` metadata are dropped silently.
        """
        model_name = program.get_metadata("mutation_model")
        if not model_name:
            return

        if outcome == MutationOutcome.REJECTED_ACCEPTOR:
            if self._skip_reward_on_acceptor_reject:
                logger.debug(
                    "[BanditModelRouter] Skipping reward update for {} ({}): "
                    "acceptor-rejected, no reliable fitness",
                    model_name,
                    outcome.value,
                )
                return
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
