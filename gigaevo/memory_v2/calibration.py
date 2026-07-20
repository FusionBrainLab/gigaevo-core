"""Prequential safety-prior calibration for immutable memory-v2 ledgers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, cast

import numpy as np
from scipy.special import expit, logit, ndtr
from scipy.stats import norm

from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.memory_v2.features import FeatureConfig, FeatureSpace
from gigaevo.memory_v2.models import (
    ApplicabilityRecord,
    CardSnapshot,
    CausalObservation,
    EnvironmentFingerprint,
    EvolutionContext,
    PolicySpecification,
    RagApplicability,
    SafetyGateMode,
    TerminalOutcome,
    canonical_digest,
    qualified_class_name,
)
from gigaevo.memory_v2.policy import safety_gate_admits
from gigaevo.memory_v2.posterior import (
    StableBayesianLogisticRegressor,
    TerminalUtilityPosteriorConfig,
)

CALIBRATION_SCHEMA = "gigaevo.memory_v2.safety_calibration/v1"
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(24)
_GATE_NODES_48, _GATE_WEIGHTS_48 = np.polynomial.hermite.hermgauss(48)
_GATE_NODES_96, _GATE_WEIGHTS_96 = np.polynomial.hermite.hermgauss(96)


@dataclass(frozen=True)
class SafetyEnvironmentKey:
    """Exact typed environment within which safety priors may be shared."""

    environment: EnvironmentFingerprint

    @classmethod
    def from_environment(
        cls, environment: EnvironmentFingerprint
    ) -> SafetyEnvironmentKey:
        return cls(environment=environment)

    @property
    def task_key(self) -> str:
        return self.environment.task_key

    @property
    def model_name(self) -> str:
        return self.environment.llm.model_name

    @property
    def mutation_operator(self) -> type[MutationOperator]:
        return self.environment.mutation_operator

    @property
    def mutation_operator_path(self) -> str:
        return qualified_class_name(self.mutation_operator)

    @property
    def label(self) -> str:
        return (
            f"{self.task_key} | {self.model_name} | {self.mutation_operator.__name__}"
        )

    def as_dict(self) -> dict[str, Any]:
        return self.environment.model_dump(mode="json")


@dataclass(frozen=True)
class CalibrationDecision:
    decision_id: str
    event_ordinal: int
    context: EvolutionContext
    lineage_registry: tuple[CardSnapshot, ...]
    candidates: tuple[CardSnapshot, ...]
    fitted_observation_ids: tuple[str, ...]
    proposed_treatment_id: str | None
    delivered: bool
    offer_probability: float | None
    proposal_probability: float | None
    joint_action_probability: float | None
    reward_q_hat_control: float | None
    reward_q_hat_treated: float | None
    risk_q_hat_control: float | None
    risk_q_hat_treated: float | None
    policy: PolicySpecification
    applicability: ApplicabilityRecord
    card_kind_contrast: bool
    citation_contrast: bool

    @property
    def proposed_card(self) -> CardSnapshot | None:
        if self.proposed_treatment_id is None:
            return None
        return next(
            (
                card
                for card in self.candidates
                if card.treatment_id == self.proposed_treatment_id
            ),
            None,
        )

    def rag_applicability(self, bank_card_id: str) -> RagApplicability:
        return self.applicability.label(bank_card_id)


@dataclass(frozen=True)
class CalibrationTrajectory:
    ledger_path: Path
    trajectory_id: str
    environment_key: SafetyEnvironmentKey
    decisions: tuple[CalibrationDecision, ...]
    observations: tuple[CausalObservation, ...]


@dataclass(frozen=True)
class SafetyPriorCandidate:
    invalidity_probability: float
    baseline_sd: float
    shared_effect_mean: float
    shared_effect_sd: float
    card_effect_sd: float

    def as_dict(self) -> dict[str, float]:
        return {
            "invalidity_prior_probability": self.invalidity_probability,
            "safety_baseline_prior_sd": self.baseline_sd,
            "safety_shared_effect_prior_mean": self.shared_effect_mean,
            "safety_shared_effect_prior_sd": self.shared_effect_sd,
            "safety_card_effect_prior_sd": self.card_effect_sd,
        }


@dataclass(frozen=True)
class _ReplayCandidate:
    treatment_id: str
    card_history_count: int
    control_design: np.ndarray
    treated_design: np.ndarray


@dataclass(frozen=True)
class _ReplayUnit:
    trajectory: str
    decision_id: str
    event_ordinal: int
    space: FeatureSpace
    history_design: np.ndarray
    history_invalid: np.ndarray
    history_count: int
    candidates: tuple[_ReplayCandidate, ...]
    outcome_treatment_id: str | None
    treatment: bool | None
    invalid: bool | None
    logged_prediction: float | None
    safety_gate_mode: SafetyGateMode
    max_treated_invalid_probability: float | None
    max_incremental_invalid_probability: float
    safety_alpha: float


def discover_ledger_paths(inputs: Iterable[str | Path]) -> tuple[Path, ...]:
    """Resolve direct SQLite paths and run/checkpoint directories."""

    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            found.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        candidates = (
            path / "memory_v2_selection_evidence.sqlite3",
            path / "memory" / "memory_v2_selection_evidence.sqlite3",
        )
        direct = [candidate for candidate in candidates if candidate.is_file()]
        if direct:
            found.update(candidate.resolve() for candidate in direct)
            continue
        recursive = tuple(path.glob("**/memory_v2_selection_evidence.sqlite3"))
        if not recursive:
            raise FileNotFoundError(
                f"no memory_v2_selection_evidence.sqlite3 under {path}"
            )
        found.update(candidate.resolve() for candidate in recursive)
    if not found:
        raise ValueError("at least one causal ledger is required")
    return tuple(sorted(found))


def _environment_from_payload(payload: dict[str, Any]) -> EnvironmentFingerprint:
    return EnvironmentFingerprint.model_validate(payload)


def _parse_decision(payload: dict[str, Any]) -> CalibrationDecision:
    context_payload = dict(payload["context"])
    context_payload["environment"] = _environment_from_payload(
        dict(context_payload["environment"])
    )
    context = EvolutionContext.model_validate(context_payload)
    return CalibrationDecision(
        decision_id=str(payload["decision_id"]),
        event_ordinal=int(payload["event_ordinal"]),
        context=context,
        lineage_registry=tuple(
            CardSnapshot.model_validate(row)
            for row in payload.get("lineage_registry", ())
        ),
        candidates=tuple(
            CardSnapshot.model_validate(row) for row in payload.get("candidates", ())
        ),
        fitted_observation_ids=tuple(payload.get("fitted_observation_ids", ())),
        proposed_treatment_id=payload.get("proposed_treatment_id"),
        delivered=bool(payload.get("delivered", False)),
        offer_probability=payload.get("offer_probability"),
        proposal_probability=payload.get("proposal_probability"),
        joint_action_probability=payload.get("joint_action_probability"),
        reward_q_hat_control=payload.get("reward_q_hat_control"),
        reward_q_hat_treated=payload.get("reward_q_hat_treated"),
        risk_q_hat_control=payload.get("risk_q_hat_control"),
        risk_q_hat_treated=payload.get("risk_q_hat_treated"),
        policy=PolicySpecification.model_validate(payload["policy"]),
        applicability=ApplicabilityRecord.model_validate(payload["applicability"]),
        card_kind_contrast=bool(payload["card_kind_contrast"]),
        citation_contrast=bool(payload.get("citation_contrast", False)),
    )


def load_calibration_trajectory(path: str | Path) -> CalibrationTrajectory:
    """Load and hash-verify one ledger without acquiring its writer lock."""

    ledger_path = Path(path).expanduser().resolve()
    uri = f"file:{ledger_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT d.environment_hash, d.record_json, d.record_hash,
                   t.terminal_json, t.terminal_hash
            FROM decisions AS d
            LEFT JOIN terminals AS t USING(decision_id)
            ORDER BY d.event_ordinal, d.decision_id
            """
        ).fetchall()
        metadata = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'trajectory_id'"
        ).fetchone()
    if not rows:
        raise ValueError(f"causal ledger has no decisions: {ledger_path}")

    decisions: list[CalibrationDecision] = []
    terminals: dict[str, TerminalOutcome] = {}
    for (
        environment_hash,
        record_json,
        record_hash,
        terminal_json,
        terminal_hash,
    ) in rows:
        raw_record = json.loads(record_json)
        if canonical_digest(raw_record) != record_hash:
            raise ValueError(
                f"decision payload hash mismatch: {raw_record.get('decision_id')!r}"
            )
        decision = _parse_decision(raw_record)
        expected_environment_hash = decision.context.environment.digest
        if expected_environment_hash != environment_hash:
            raise ValueError(
                f"decision environment hash mismatch: {decision.decision_id!r}"
            )
        if (
            decision.proposed_treatment_id is not None
            and decision.proposed_card is None
        ):
            raise ValueError(
                f"proposal {decision.decision_id!r} is outside its candidate set"
            )
        decisions.append(decision)
        if terminal_json is None:
            continue
        raw_terminal = json.loads(terminal_json)
        if canonical_digest(raw_terminal) != terminal_hash:
            raise ValueError(
                f"terminal payload hash mismatch: {decision.decision_id!r}"
            )
        parsed_terminal = TerminalOutcome.model_validate(raw_terminal)
        if parsed_terminal.decision_id != decision.decision_id:
            raise ValueError("terminal and decision ids differ")
        terminals[parsed_terminal.decision_id] = parsed_terminal

    environment_keys = {
        SafetyEnvironmentKey.from_environment(row.context.environment)
        for row in decisions
    }
    if len(environment_keys) != 1:
        raise ValueError(f"ledger mixes safety environments: {ledger_path}")

    observations: list[CausalObservation] = []
    for decision in decisions:
        terminal = terminals.get(decision.decision_id)
        card = decision.proposed_card
        if (
            card is None
            or terminal is None
            or terminal.status == "censored"
            or not terminal.ope_eligible
        ):
            continue
        required = (
            decision.offer_probability,
            decision.proposal_probability,
            decision.joint_action_probability,
            decision.reward_q_hat_control,
            decision.reward_q_hat_treated,
            decision.risk_q_hat_control,
            decision.risk_q_hat_treated,
        )
        if any(value is None for value in required):
            raise ValueError(f"proposal {decision.decision_id!r} lacks frozen fields")
        offer_probability = cast(float, decision.offer_probability)
        proposal_probability = cast(float, decision.proposal_probability)
        joint_action_probability = cast(float, decision.joint_action_probability)
        reward_q_hat_control = cast(float, decision.reward_q_hat_control)
        reward_q_hat_treated = cast(float, decision.reward_q_hat_treated)
        risk_q_hat_control = cast(float, decision.risk_q_hat_control)
        risk_q_hat_treated = cast(float, decision.risk_q_hat_treated)
        observations.append(
            CausalObservation(
                decision_id=decision.decision_id,
                event_ordinal=decision.event_ordinal,
                card=card,
                context=decision.context,
                rag_applicability=decision.rag_applicability(card.bank_card_id),
                treatment=decision.delivered,
                card_used=(
                    decision.delivered and card.bank_card_id in terminal.used_card_ids
                ),
                offer_propensity=offer_probability,
                proposal_propensity=proposal_probability,
                joint_action_propensity=joint_action_probability,
                status=terminal.status,
                measurement=terminal.measurement,
                reward_q_hat_control=reward_q_hat_control,
                reward_q_hat_treated=reward_q_hat_treated,
                risk_q_hat_control=risk_q_hat_control,
                risk_q_hat_treated=risk_q_hat_treated,
            )
        )
    return CalibrationTrajectory(
        ledger_path=ledger_path,
        trajectory_id=(str(metadata[0]) if metadata is not None else str(ledger_path)),
        environment_key=next(iter(environment_keys)),
        decisions=tuple(decisions),
        observations=tuple(observations),
    )


def _stack(rows: Sequence[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=float)
    return np.stack(rows)


def _prepare_units(
    trajectories: Sequence[CalibrationTrajectory],
) -> tuple[_ReplayUnit, ...]:
    result: list[_ReplayUnit] = []
    for trajectory in trajectories:
        observations = {row.decision_id: row for row in trajectory.observations}
        for decision in trajectory.decisions:
            if not decision.candidates:
                continue
            history: list[CausalObservation] = []
            for decision_id in decision.fitted_observation_ids:
                row = observations.get(decision_id)
                if row is None:
                    raise ValueError(
                        f"decision {decision.decision_id!r} references unavailable "
                        f"fitted observation {decision_id!r}"
                    )
                history.append(row)
            if any(row.event_ordinal >= decision.event_ordinal for row in history):
                raise ValueError("prequential history contains a future decision")
            cards = tuple(
                [row.card for row in history]
                + list(decision.lineage_registry)
                + list(decision.candidates)
            )
            space = FeatureSpace(
                FeatureConfig(
                    behavior_keys=tuple(
                        coordinate.key
                        for coordinate in decision.context.map_elites.coordinates
                    ),
                    card_kind_contrast=decision.card_kind_contrast,
                    retrieval_applicability_contrast=(
                        decision.applicability.specification.retrieval_applicability_contrast
                    ),
                    citation_contrast=decision.citation_contrast,
                ),
                cards,
            )
            history_design = _stack(
                [
                    space.design(
                        row.card,
                        row.context,
                        row.treatment,
                        rag_contrast=row.rag_applicability.contrast,
                        use_contrast=row.use_contrast,
                    )
                    for row in history
                ],
                space.outcome_dim,
            )
            replay_candidates: list[_ReplayCandidate] = []
            for card in decision.candidates:
                current_bank_id = space.bank_lineage_id(card)
                replay_candidates.append(
                    _ReplayCandidate(
                        treatment_id=card.treatment_id,
                        card_history_count=sum(
                            space.bank_lineage_id(row.card) == current_bank_id
                            for row in history
                        ),
                        control_design=space.design(
                            card,
                            decision.context,
                            False,
                            rag_contrast=decision.rag_applicability(
                                card.bank_card_id
                            ).contrast,
                        ),
                        treated_design=space.design(
                            card,
                            decision.context,
                            True,
                            rag_contrast=decision.rag_applicability(
                                card.bank_card_id
                            ).contrast,
                        ),
                    )
                )
            observation = observations.get(decision.decision_id)
            logged_prediction = None
            if observation is not None:
                logged_prediction = (
                    observation.risk_q_hat_treated
                    if observation.treatment
                    else observation.risk_q_hat_control
                )
            result.append(
                _ReplayUnit(
                    trajectory=trajectory.trajectory_id,
                    decision_id=decision.decision_id,
                    event_ordinal=decision.event_ordinal,
                    space=space,
                    history_design=history_design,
                    history_invalid=np.asarray(
                        [float(row.invalid) for row in history], dtype=float
                    ),
                    history_count=len(history),
                    candidates=tuple(replay_candidates),
                    outcome_treatment_id=(
                        observation.card.treatment_id
                        if observation is not None
                        else None
                    ),
                    treatment=(
                        observation.treatment if observation is not None else None
                    ),
                    invalid=(observation.invalid if observation is not None else None),
                    logged_prediction=logged_prediction,
                    safety_gate_mode=decision.policy.safety_gate_mode,
                    max_treated_invalid_probability=(
                        decision.policy.max_treated_invalid_probability
                    ),
                    max_incremental_invalid_probability=(
                        decision.policy.max_incremental_invalid_probability
                    ),
                    safety_alpha=decision.policy.safety_alpha,
                )
            )
    return tuple(result)


def _logistic_normal_mean(mean: float, variance: float) -> float:
    if variance <= 0.0:
        return float(expit(mean))
    values = expit(mean + math.sqrt(2.0 * variance) * _HERMITE_NODES)
    return float((_HERMITE_WEIGHTS @ values) / math.sqrt(math.pi))


def _gate_probability_quadrature(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    max_treated_invalid_probability: float | None,
    max_incremental_invalid_probability: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized deterministic replay of bivariate safety probabilities.

    The 96-node estimate drives replay and its discrepancy from a 48-node
    estimate is subtracted conservatively. The live policy continues to use
    adaptive certified integration; this approximation only makes exhaustive
    offline candidate-set replay tractable.
    """

    count = len(means)
    if means.shape != (count, 2) or covariances.shape != (count, 2, 2):
        raise ValueError("gate replay requires batches of bivariate moments")
    treated_upper = np.ones(count, dtype=float)
    safe_probability = np.zeros(count, dtype=float)
    discrepancy = np.ones(count, dtype=float)
    variance0 = covariances[:, 0, 0]
    variance1 = covariances[:, 1, 1]
    covariance01 = covariances[:, 0, 1]
    finite = np.isfinite(means).all(axis=1) & np.isfinite(covariances).all(axis=(1, 2))
    positive = (variance0 > 0.0) & (variance1 > 0.0)
    sd0 = np.sqrt(np.maximum(variance0, 0.0))
    sd1 = np.sqrt(np.maximum(variance1, 0.0))
    slope = np.divide(
        covariance01,
        sd0,
        out=np.zeros_like(covariance01),
        where=sd0 > 0.0,
    )
    conditional_variance = variance1 - slope * slope
    nonsingular = conditional_variance > 1e-12 * np.maximum(variance1, 1.0)
    valid = finite & positive & nonsingular
    if not np.any(valid):
        return treated_upper, safe_probability, discrepancy

    indices = np.flatnonzero(valid)
    selected_means = means[indices]
    selected_sd0 = sd0[indices]
    selected_sd1 = sd1[indices]
    selected_slope = slope[indices]
    conditional_sd = np.sqrt(conditional_variance[indices])
    treated_upper[indices] = expit(
        selected_means[:, 1] + norm.ppf(1.0 - alpha) * selected_sd1
    )
    absolute_limit = (
        1.0
        if max_treated_invalid_probability is None
        else max_treated_invalid_probability
    )
    treated_limit = (
        -math.inf
        if absolute_limit <= 0.0
        else math.inf
        if absolute_limit >= 1.0
        else float(logit(absolute_limit))
    )

    def estimate(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
        z = math.sqrt(2.0) * nodes
        eta0 = selected_means[:, 0, None] + selected_sd0[:, None] * z
        target = expit(eta0) + max_incremental_invalid_probability
        finite_boundary = logit(np.clip(target, 1e-15, 1.0 - 1e-15))
        risk_boundary = np.where(
            target <= 0.0,
            -math.inf,
            np.where(target >= 1.0, math.inf, finite_boundary),
        )
        boundary = np.minimum(treated_limit, risk_boundary)
        conditional_mean = selected_means[:, 1, None] + selected_slope[:, None] * z
        conditional_probability = ndtr(
            (boundary - conditional_mean) / conditional_sd[:, None]
        )
        return conditional_probability @ weights / math.sqrt(math.pi)

    estimate_48 = estimate(_GATE_NODES_48, _GATE_WEIGHTS_48)
    estimate_96 = estimate(_GATE_NODES_96, _GATE_WEIGHTS_96)
    selected_discrepancy = np.abs(estimate_96 - estimate_48)
    discrepancy[indices] = selected_discrepancy
    safe_probability[indices] = np.maximum(0.0, estimate_96 - selected_discrepancy)
    return treated_upper, safe_probability, discrepancy


def _prediction_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    treated: np.ndarray,
    trajectories: np.ndarray,
) -> dict[str, Any]:
    clipped = np.clip(predicted, 1e-9, 1.0 - 1e-9)
    losses = -(actual * np.log(clipped) + (1.0 - actual) * np.log1p(-clipped))

    def arm(mask: np.ndarray) -> dict[str, float | int]:
        if not np.any(mask):
            return {"observations": 0, "mean_predicted": 0.0, "realized_rate": 0.0}
        return {
            "observations": int(mask.sum()),
            "mean_predicted": float(predicted[mask].mean()),
            "realized_rate": float(actual[mask].mean()),
        }

    bins: list[dict[str, float | int]] = []
    order = np.argsort(predicted)
    for indices in np.array_split(order, min(5, len(order))):
        if not len(indices):
            continue
        bins.append(
            {
                "observations": int(len(indices)),
                "mean_predicted": float(predicted[indices].mean()),
                "realized_rate": float(actual[indices].mean()),
            }
        )
    trajectory_losses = np.asarray(
        [losses[trajectories == key].mean() for key in np.unique(trajectories)]
    )
    return {
        "observations": int(len(actual)),
        "invalid": int(actual.sum()),
        "mean_predicted": float(predicted.mean()),
        "realized_rate": float(actual.mean()),
        "calibration_bias": float(predicted.mean() - actual.mean()),
        "brier_score": float(np.mean((predicted - actual) ** 2)),
        "log_loss": float(losses.mean()),
        "trajectory_clustered_log_loss_se": (
            float(trajectory_losses.std(ddof=1) / math.sqrt(len(trajectory_losses)))
            if len(trajectory_losses) > 1
            else None
        ),
        "treated": arm(treated),
        "control": arm(~treated),
        "calibration_bins": bins,
    }


def _score_candidate(
    units: Sequence[_ReplayUnit], candidate: SafetyPriorCandidate
) -> dict[str, Any]:
    config = TerminalUtilityPosteriorConfig(
        invalidity_prior_probability=candidate.invalidity_probability,
        safety_baseline_prior_sd=candidate.baseline_sd,
        safety_shared_effect_prior_mean=candidate.shared_effect_mean,
        safety_shared_effect_prior_sd=candidate.shared_effect_sd,
        safety_card_effect_prior_sd=candidate.card_effect_sd,
    )
    regressor = StableBayesianLogisticRegressor(config)
    predicted: list[float] = []
    actual: list[float] = []
    treated: list[bool] = []
    prediction_trajectories: list[str] = []
    gate_admitted: dict[str, list[bool]] = {
        "all_candidates": [],
        "cold_start": [],
        "new_card_after_history": [],
        "seen_card": [],
    }
    gate_decision_admitted: dict[str, list[bool]] = {key: [] for key in gate_admitted}
    safe_probabilities: list[float] = []
    treated_upper: list[float] = []
    gate_discrepancies: list[float] = []
    for unit in units:
        prior_mean = np.zeros(unit.space.outcome_dim, dtype=float)
        prior_mean[0] = float(logit(candidate.invalidity_probability))
        prior_mean[unit.space.baseline_dim] = candidate.shared_effect_mean
        prior_variance = unit.space.prior_variance(
            baseline_sd=candidate.baseline_sd,
            shared_effect_sd=candidate.shared_effect_sd,
            card_effect_sd=candidate.card_effect_sd,
        )
        posterior = regressor.fit(
            unit.history_design,
            unit.history_invalid,
            prior_mean,
            prior_variance,
        )
        by_treatment = {row.treatment_id: row for row in unit.candidates}
        if unit.outcome_treatment_id is not None:
            outcome_candidate = by_treatment[unit.outcome_treatment_id]
            if unit.treatment is None or unit.invalid is None:
                raise ValueError("eligible replay outcome is incomplete")
            design = (
                outcome_candidate.treated_design
                if unit.treatment
                else outcome_candidate.control_design
            )
            latent_mean = float(design @ posterior.mean)
            latent_variance = float(design @ posterior.covariance @ design)
            predicted.append(_logistic_normal_mean(latent_mean, latent_variance))
            actual.append(float(unit.invalid))
            treated.append(unit.treatment)
            prediction_trajectories.append(unit.trajectory)

        admitted_in_decision: dict[str, list[bool]] = {key: [] for key in gate_admitted}
        safety_designs = np.stack(
            [
                np.stack((row.control_design, row.treated_design))
                for row in unit.candidates
            ]
        )
        safety_means = np.einsum("cad,d->ca", safety_designs, posterior.mean)
        safety_covariances = np.einsum(
            "cai,ij,cbj->cab",
            safety_designs,
            posterior.covariance,
            safety_designs,
        )
        bounds, safety_probabilities, discrepancies = _gate_probability_quadrature(
            safety_means,
            safety_covariances,
            max_treated_invalid_probability=unit.max_treated_invalid_probability,
            max_incremental_invalid_probability=(
                unit.max_incremental_invalid_probability
            ),
            alpha=unit.safety_alpha,
        )
        for replay_candidate, treated_bound, probability_safe, discrepancy in zip(
            unit.candidates,
            bounds,
            safety_probabilities,
            discrepancies,
            strict=True,
        ):
            admitted = safety_gate_admits(
                gate_mode=unit.safety_gate_mode,
                probability_acceptable=float(probability_safe),
                alpha=unit.safety_alpha,
            )
            stratum = (
                "cold_start"
                if unit.history_count == 0
                else "new_card_after_history"
                if replay_candidate.card_history_count == 0
                else "seen_card"
            )
            for key in ("all_candidates", stratum):
                gate_admitted[key].append(admitted)
                admitted_in_decision[key].append(admitted)
            treated_upper.append(treated_bound)
            safe_probabilities.append(probability_safe)
            gate_discrepancies.append(discrepancy)
        for key, values in admitted_in_decision.items():
            if values:
                gate_decision_admitted[key].append(any(values))
    metrics = _prediction_metrics(
        np.asarray(predicted),
        np.asarray(actual),
        np.asarray(treated, dtype=bool),
        np.asarray(prediction_trajectories),
    )

    def gate_slice(key: str) -> dict[str, float | int]:
        admitted_array = np.asarray(gate_admitted[key], dtype=bool)
        decision_array = np.asarray(gate_decision_admitted[key], dtype=bool)
        if not len(admitted_array):
            return {
                "candidates": 0,
                "admitted": 0,
                "admitted_fraction": 0.0,
                "decisions": 0,
                "decisions_with_admitted_candidate": 0,
                "decision_retention": 0.0,
            }
        return {
            "candidates": int(len(admitted_array)),
            "admitted": int(admitted_array.sum()),
            "admitted_fraction": float(admitted_array.mean()),
            "decisions": int(len(decision_array)),
            "decisions_with_admitted_candidate": int(decision_array.sum()),
            "decision_retention": float(decision_array.mean()),
        }

    return {
        "prior": candidate.as_dict(),
        **metrics,
        "gate_replay": {key: gate_slice(key) for key in gate_admitted}
        | {
            "method": "conservative_gauss_hermite_96_vs_48",
            "mean_probability_safe": float(np.mean(safe_probabilities)),
            "mean_treated_invalid_upper": float(np.mean(treated_upper)),
            "mean_quadrature_discrepancy": float(np.mean(gate_discrepancies)),
            "max_quadrature_discrepancy": float(np.max(gate_discrepancies)),
        },
    }


def _offer_rate_diagnostics(
    *, proposal_count: int, offer_rates: Sequence[float]
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for rate in offer_rates:
        if not 0.0 < rate < 1.0:
            raise ValueError("offer rates must lie strictly between zero and one")
        result.append(
            {
                "offer_probability": rate,
                "control_probability": 1.0 - rate,
                "expected_treated": proposal_count * rate,
                "expected_controls": proposal_count * (1.0 - rate),
                "homoscedastic_information_proxy_vs_50_50": (
                    rate * (1.0 - rate) / 0.25
                ),
                "homoscedastic_variance_proxy_vs_50_50": (0.25 / (rate * (1.0 - rate))),
            }
        )
    return result


def calibrate_safety_priors(
    trajectories: Sequence[CalibrationTrajectory],
    *,
    prior_probabilities: Sequence[float],
    baseline_sds: Sequence[float],
    shared_effect_sd: float,
    card_effect_sd: float,
    shared_effect_means: Sequence[float] = (0.0,),
    offer_rates: Sequence[float] = (0.5, 0.7, 0.75, 0.8),
    min_observations: int = 50,
    min_gate_retention: float = 0.25,
) -> dict[str, Any]:
    """Replay and rank safety priors separately for every environment key."""

    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    if not 0.0 <= min_gate_retention <= 1.0:
        raise ValueError("minimum gate retention must be in [0, 1]")
    if not prior_probabilities or not baseline_sds:
        raise ValueError("the calibration grid cannot be empty")
    if any(not 0.0 < value < 0.5 for value in prior_probabilities):
        raise ValueError("prior invalidity probabilities must be in (0, 0.5)")
    if any(value <= 0.0 for value in baseline_sds):
        raise ValueError("baseline prior standard deviations must be positive")
    if not shared_effect_means:
        raise ValueError("the shared-effect prior grid cannot be empty")
    if any(not math.isfinite(value) for value in shared_effect_means):
        raise ValueError("shared effect prior means must be finite")

    grouped: dict[SafetyEnvironmentKey, list[CalibrationTrajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.environment_key, []).append(trajectory)

    reports: list[dict[str, Any]] = []
    for key, group in sorted(
        grouped.items(), key=lambda item: item[0].environment.digest
    ):
        trajectory_ids = [row.trajectory_id for row in group]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError(
                f"environment contains duplicate trajectory snapshots: {key.label}"
            )
        decision_ids = [
            decision.decision_id for row in group for decision in row.decisions
        ]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError(
                f"environment contains duplicate causal decisions: {key.label}"
            )
        units = _prepare_units(group)
        outcome_units = [unit for unit in units if unit.invalid is not None]
        if not outcome_units:
            raise ValueError(
                f"environment has no closed eligible proposals: {key.label}"
            )
        actual = np.asarray([float(cast(bool, unit.invalid)) for unit in outcome_units])
        treated = np.asarray(
            [bool(unit.treatment) for unit in outcome_units], dtype=bool
        )
        logged = np.asarray(
            [float(cast(float, unit.logged_prediction)) for unit in outcome_units]
        )
        deployed = _prediction_metrics(
            logged,
            actual,
            treated,
            np.asarray([unit.trajectory for unit in outcome_units]),
        )
        treated_mask = treated
        has_both_arms = bool(treated_mask.any() and (~treated_mask).any())
        treated_rate = (
            float((actual[treated_mask].sum() + 0.5) / (treated_mask.sum() + 1.0))
            if treated_mask.any()
            else None
        )
        control_rate = (
            float((actual[~treated_mask].sum() + 0.5) / ((~treated_mask).sum() + 1.0))
            if (~treated_mask).any()
            else None
        )
        empirical_shared_effect = (
            float(logit(treated_rate) - logit(control_rate))
            if has_both_arms and treated_rate is not None and control_rate is not None
            else None
        )
        candidates = [
            _score_candidate(
                units,
                SafetyPriorCandidate(
                    invalidity_probability=probability,
                    baseline_sd=baseline_sd,
                    shared_effect_mean=effect_mean,
                    shared_effect_sd=shared_effect_sd,
                    card_effect_sd=card_effect_sd,
                ),
            )
            for probability in sorted(set(prior_probabilities))
            for baseline_sd in sorted(set(baseline_sds))
            for effect_mean in sorted(set(shared_effect_means))
        ]
        candidates.sort(
            key=lambda row: (
                row["log_loss"],
                row["brier_score"],
                abs(row["calibration_bias"]),
            )
        )
        calibration_best = candidates[0]

        def retains(stratum: dict[str, float | int]) -> bool:
            return (
                stratum["candidates"] == 0
                or stratum["admitted_fraction"] >= min_gate_retention
            )

        bootstrap_viable = [
            row
            for row in candidates
            if retains(row["gate_replay"]["cold_start"])
            and retains(row["gate_replay"]["new_card_after_history"])
        ]
        best = bootstrap_viable[0] if bootstrap_viable else None
        enough = len(outcome_units) >= min_observations
        independent = len(group) >= 2
        status = (
            "insufficient_evidence"
            if not enough
            else "no_bootstrap_viable_prior"
            if best is None
            else "provisional_single_trajectory"
            if not independent
            else "multi_trajectory_development_candidate"
        )
        overrides = None
        if enough and best is not None:
            prior = best["prior"]
            overrides = [
                "memory.posterior_config.invalidity_prior_probability="
                f"{prior['invalidity_prior_probability']}",
                "memory.posterior_config.safety_baseline_prior_sd="
                f"{prior['safety_baseline_prior_sd']}",
                "memory.posterior_config.safety_shared_effect_prior_mean="
                f"{prior['safety_shared_effect_prior_mean']}",
                "memory.posterior_config.safety_shared_effect_prior_sd="
                f"{prior['safety_shared_effect_prior_sd']}",
                "memory.posterior_config.safety_card_effect_prior_sd="
                f"{prior['safety_card_effect_prior_sd']}",
            ]
        reports.append(
            {
                "environment": key.as_dict(),
                "status": status,
                "trajectory_count": len(group),
                "decision_count": sum(len(row.decisions) for row in group),
                "eligible_proposal_outcomes": len(outcome_units),
                "gate_replay_decisions": len(units),
                "minimum_gate_retention": min_gate_retention,
                "jeffreys_smoothed_control_invalidity": control_rate,
                "jeffreys_smoothed_treated_invalidity": treated_rate,
                "empirical_shared_effect_log_odds": empirical_shared_effect,
                "deployed_prequential": deployed,
                "best_calibrated_prior": calibration_best,
                "recommended_prior": best if enough else None,
                "calibration_winner_bootstrap_viable": (calibration_best is best),
                "hydra_overrides": overrides,
                "grid_ranking": candidates,
                "offer_rate_diagnostics": _offer_rate_diagnostics(
                    proposal_count=sum(
                        decision.proposed_treatment_id is not None
                        for trajectory in group
                        for decision in trajectory.decisions
                    ),
                    offer_rates=offer_rates,
                ),
            }
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "retrospectively selected prequential Bernoulli log loss",
        "scope_note": (
            "Priors are grouped by the complete typed environment fingerprint. "
            "The emitted candidate minimizes prequential log loss among priors "
            "that retain the configured fraction of cold-start and later-new-card "
            "candidates under the frozen safety gate. Gate replay covers every "
            "logged candidate set, including decisions where the deployed policy "
            "abstained. The unconstrained calibration winner is also "
            "reported. A single-trajectory recommendation is provisional and "
            "requires a fresh validation run. Candidate means are supplied "
            "independently of ledger outcomes; the full-sample empirical effect "
            "is descriptive only. Selecting the grid winner on these same "
            "trajectories is retrospective model selection, so its loss is a "
            "development estimate rather than independent validation. Offer-rate "
            "rows are analytical "
            "homoscedastic overlap proxies, not policy-effect estimates. Gate "
            "replay is evaluated on logged contexts and does not identify the "
            "trajectory induced by changing admission."
        ),
        "sources": [str(row.ledger_path) for row in trajectories],
        "groups": reports,
    }
