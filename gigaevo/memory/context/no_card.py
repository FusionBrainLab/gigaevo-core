"""Persisted no-card evidence for read-side abstention gates.

The writer observes no-card controls and natural empty selections while it has
the program pool in hand.  The reader only consumes this compact JSON state; it
never reaches into program storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
import fcntl
import json
import math
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import DecisionContext
from gigaevo.memory.context import ContextKey, GlobalMemoryContext
from gigaevo.memory.context.beta import BetaPrior, coerce_beta_prior
from gigaevo.memory.context.evidence import clean_ids, oriented_delta


class _JsonFileLock(AbstractContextManager["_JsonFileLock"]):
    """Small advisory lock local to the no-card evidence JSON file."""

    def __init__(self, path: Path, *, exclusive: bool) -> None:
        self._path = path
        self._exclusive = exclusive
        self._fh: BinaryIO | None = None

    def __enter__(self) -> _JsonFileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = self._path.open("a+b")
        self._fh = fh
        mode = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        fcntl.flock(fh.fileno(), mode)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class NoCardGateSummary(BaseModel):
    """Read-side no-card abstention summary for one context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prior: BetaPrior
    baseline: float = Field(
        default=0.0,
        description="Robust no-card child-parent delta location for audit only.",
    )
    evidence_n: float = Field(default=0.0, ge=0.0)
    source: str = Field(default="seed")


@runtime_checkable
class NoCardEvidenceProvider(Protocol):
    def summary_for(self, context: Any = None) -> NoCardGateSummary: ...


@runtime_checkable
class NoCardEvidenceRecorder(Protocol):
    def record_outcomes(
        self,
        outcomes: Sequence[NoCardEvidenceOutcome],
        *,
        higher_is_better: bool,
        task_key: str = "",
    ) -> None: ...


@runtime_checkable
class NoCardEvidenceOutcome(Protocol):
    """Outcome fields persisted for read-side no-card abstention evidence."""

    @property
    def id(self) -> str: ...

    @property
    def fitness(self) -> float | None: ...

    @property
    def invalid(self) -> bool: ...

    @property
    def base_selected_ids(self) -> Sequence[str]: ...

    @property
    def base_metrics(self) -> dict[str, float]: ...

    @property
    def base_id(self) -> str: ...

    @property
    def base_fitness(self) -> float | None: ...

    @property
    def no_card_control(self) -> bool: ...


class _Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    task_key: str = ""
    context_key: ContextKey
    delta: float
    randomized_control: bool = False
    invalid: bool = False


_Row = tuple[float, float, bool]
"""(delta, weight, invalid) evidence row for the sign gate."""


def _outcome_context(
    outcome: NoCardEvidenceOutcome, *, task_key: str = ""
) -> DecisionContext:
    return DecisionContext(
        task_key=task_key,
        parent_metrics=dict(outcome.base_metrics or {}),
        parent_id=str(outcome.base_id or ""),
    )


def _weighted_median(values: Sequence[_Row]) -> float:
    # Invalid children carry no honest progress magnitude; they enter the sign
    # counts as failures but must not drag the location estimate toward zero.
    finite = sorted(
        (float(value), float(weight))
        for value, weight, invalid in values
        if not invalid
        and math.isfinite(float(value))
        and math.isfinite(float(weight))
        and weight > 0.0
    )
    if not finite:
        return 0.0
    total = sum(weight for _, weight in finite)
    cursor = 0.0
    target = total / 2.0
    for idx, (value, weight) in enumerate(finite):
        cursor += weight
        if cursor > target:
            return value
        if math.isclose(cursor, target):
            if idx + 1 < len(finite):
                return (value + finite[idx + 1][0]) / 2.0
            return value
    return finite[-1][0]


def _signed_counts(values: Sequence[_Row], baseline: float) -> tuple[float, float]:
    success = 0.0
    failure = 0.0
    for value, weight, invalid in values:
        if invalid:
            failure += weight
        elif value > baseline:
            success += weight
        elif value < baseline:
            failure += weight
        else:
            success += 0.5 * weight
            failure += 0.5 * weight
    return success, failure


class JsonNoCardEvidenceStore(BaseModel):
    """JSON-backed no-card evidence store shared by writer and reader."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    path: Path = Field(description="JSON file under the memory checkpoint dir.")
    context_model: Any = Field(default_factory=GlobalMemoryContext)
    seed_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="No-card sign prior when no usable evidence exists.",
    )
    natural_empty_weight: float = Field(
        default=0.25,
        ge=0.0,
        description="Weight for natural policy-empty rows; randomized controls get 1.",
    )
    local_min_effective_n: float = Field(
        default=8.0,
        ge=0.0,
        description="Minimum local control weight before a context-specific gate is trusted.",
    )
    shrink_events: float = Field(
        default=16.0,
        ge=0.0,
        description="Global pseudo-counts used when local evidence is sparse.",
    )
    sign_strength_cap: float = Field(
        default=64.0,
        gt=0.0,
        description="Maximum effective no-card sign evidence in the Beta gate.",
    )
    max_observations: int = Field(
        default=1024,
        gt=0,
        description="Per-task retention cap on persisted observations; oldest "
        "natural rows are pruned before any randomized control.",
    )

    def record_outcomes(
        self,
        outcomes: Sequence[NoCardEvidenceOutcome],
        *,
        higher_is_better: bool,
        task_key: str = "",
    ) -> None:
        fresh = self._observations_from(
            outcomes, higher_is_better=higher_is_better, task_key=task_key
        )
        if not fresh:
            return
        with self._file_lock(exclusive=True):
            by_id = {(obs.task_key, obs.id): obs for obs in self._read_unlocked()}
            by_id.update({(obs.task_key, obs.id): obs for obs in fresh})
            self._write_unlocked(self._pruned(tuple(by_id.values())))

    def _pruned(
        self, observations: tuple[_Observation, ...]
    ) -> tuple[_Observation, ...]:
        by_task: dict[str, list[_Observation]] = {}
        for obs in observations:
            by_task.setdefault(obs.task_key, []).append(obs)

        drop: set[tuple[str, str]] = set()
        for task_observations in by_task.values():
            overflow = len(task_observations) - self.max_observations
            if overflow <= 0:
                continue
            # Naturals age first because controls cost a withheld injection.
            task_drop_count = 0
            for controls_pass in (False, True):
                for obs in task_observations:
                    if task_drop_count >= overflow:
                        break
                    if obs.randomized_control == controls_pass:
                        drop.add((obs.task_key, obs.id))
                        task_drop_count += 1
                else:
                    continue
                break
        return tuple(obs for obs in observations if (obs.task_key, obs.id) not in drop)

    def _observations_from(
        self,
        outcomes: Sequence[NoCardEvidenceOutcome],
        *,
        higher_is_better: bool,
        task_key: str = "",
    ) -> tuple[_Observation, ...]:
        observations: list[_Observation] = []
        for outcome in outcomes:
            if outcome.base_fitness is None:
                continue
            if clean_ids(tuple(outcome.base_selected_ids or ())):
                continue
            if outcome.invalid:
                delta = 0.0
            else:
                computed = oriented_delta(
                    outcome.fitness, outcome.base_fitness, higher_is_better
                )
                if computed is None or not math.isfinite(computed):
                    continue
                delta = computed
            oid = str(outcome.id or "")
            if not oid:
                continue
            context = _outcome_context(outcome, task_key=task_key)
            observations.append(
                _Observation(
                    id=oid,
                    task_key=task_key,
                    context_key=self.context_model.key_for(context),
                    delta=float(delta),
                    randomized_control=bool(outcome.no_card_control),
                    invalid=bool(outcome.invalid),
                )
            )
        return tuple(observations)

    def summary_for(self, context: Any = None) -> NoCardGateSummary:
        key = self.context_model.key_for(context)
        task_key = context.task_key if context is not None else ""
        observations = tuple(obs for obs in self._read() if obs.task_key == task_key)
        if not observations:
            return self._seed_summary("seed")

        local = [obs for obs in observations if obs.context_key == key]

        local_controls = [
            (obs.delta, 1.0, obs.invalid) for obs in local if obs.randomized_control
        ]
        all_controls = [
            (obs.delta, 1.0, obs.invalid)
            for obs in observations
            if obs.randomized_control
        ]
        if key.kind == "global" and all_controls:
            return self._summary_from(all_controls, "global_control")
        if sum(weight for _, weight, _ in local_controls) >= self.local_min_effective_n:
            return self._summary_from(local_controls, "local_control")
        nonlocal_controls = [
            (obs.delta, 1.0, obs.invalid)
            for obs in observations
            if obs.randomized_control and obs.context_key != key
        ]
        if nonlocal_controls:
            return self._shrunk_summary(local_controls, nonlocal_controls)
        if local_controls:
            return self._summary_from(local_controls, "local_control_sparse")

        weighted = [
            (obs.delta, self.natural_empty_weight, obs.invalid)
            for obs in (local if local else observations)
        ]
        if weighted:
            return self._summary_from(weighted, "natural_empty")
        return self._seed_summary("seed")

    def _shrunk_summary(
        self,
        local: Sequence[_Row],
        global_controls: Sequence[_Row],
    ) -> NoCardGateSummary:
        global_baseline = _weighted_median(global_controls)
        global_success, global_failure = _signed_counts(
            global_controls, global_baseline
        )
        global_n = global_success + global_failure
        if global_n <= 0.0:
            return self._summary_from(local, "local_control_sparse")
        local_baseline = _weighted_median(local) if local else global_baseline
        local_success, local_failure = _signed_counts(local, local_baseline)
        local_n = local_success + local_failure
        pseudo_n = min(self.shrink_events, global_n)
        success = local_success + pseudo_n * (global_success / global_n)
        failure = local_failure + pseudo_n * (global_failure / global_n)
        n = success + failure
        if n > self.sign_strength_cap and n > 0.0:
            scale = self.sign_strength_cap / n
            success *= scale
            failure *= scale
            n = self.sign_strength_cap
        seed = coerce_beta_prior(self.seed_prior, source="seed")
        denominator = local_n + pseudo_n
        local_weight = local_n / denominator if denominator > 0.0 else 0.0
        baseline = (
            local_weight * local_baseline + (1.0 - local_weight) * global_baseline
        )
        return NoCardGateSummary(
            prior=BetaPrior(
                alpha=seed.alpha + success,
                beta=seed.beta + failure,
                source="local_shrunk",
                support_n=n,
            ),
            baseline=baseline,
            evidence_n=n,
            source="local_shrunk",
        )

    def _summary_from(self, values: Sequence[_Row], source: str) -> NoCardGateSummary:
        baseline = _weighted_median(values)
        success, failure = _signed_counts(values, baseline)
        n = success + failure
        seed = coerce_beta_prior(self.seed_prior, source="seed")
        if n > self.sign_strength_cap and n > 0:
            scale = self.sign_strength_cap / n
            success *= scale
            failure *= scale
        prior = BetaPrior(
            alpha=seed.alpha + success,
            beta=seed.beta + failure,
            source=source,
            support_n=min(self.sign_strength_cap, n),
        )
        return NoCardGateSummary(
            prior=prior, baseline=baseline, evidence_n=n, source=source
        )

    def _seed_summary(self, source: str) -> NoCardGateSummary:
        return NoCardGateSummary(
            prior=coerce_beta_prior(self.seed_prior, source=source),
            baseline=0.0,
            evidence_n=0.0,
            source=source,
        )

    def _read(self) -> tuple[_Observation, ...]:
        with self._file_lock(exclusive=False):
            return self._read_unlocked()

    def _read_unlocked(self) -> tuple[_Observation, ...]:
        try:
            if not self.path.exists():
                return ()
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw.get("observations", []) if isinstance(raw, dict) else []
            return tuple(_Observation.model_validate(row) for row in rows)
        except Exception as exc:
            logger.warning(
                "[Memory][NoCard] failed to read {}: {}; using empty evidence",
                self.path,
                exc,
            )
            return ()

    def _write(self, observations: Sequence[_Observation]) -> None:
        with self._file_lock(exclusive=True):
            self._write_unlocked(observations)

    def _write_unlocked(self, observations: Sequence[_Observation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "observations": [obs.model_dump(mode="json") for obs in observations]
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    def _file_lock(self, *, exclusive: bool) -> _JsonFileLock:
        return _JsonFileLock(
            self.path.with_suffix(self.path.suffix + ".lock"), exclusive=exclusive
        )


class NullNoCardEvidenceProvider(BaseModel):
    """Static no-card summary used by legacy policies and tests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed_prior: tuple[float, float] = (3.0, 3.0)

    def summary_for(self, context: Any = None) -> NoCardGateSummary:
        del context
        return NoCardGateSummary(
            prior=coerce_beta_prior(self.seed_prior, source="fixed_no_card"),
            baseline=0.0,
            evidence_n=0.0,
            source="fixed_no_card",
        )

    def record_outcomes(
        self,
        outcomes: Sequence[Any],
        *,
        higher_is_better: bool,
        task_key: str = "",
    ) -> None:
        del outcomes, higher_is_better, task_key
