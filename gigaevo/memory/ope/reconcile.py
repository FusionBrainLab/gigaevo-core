"""Reconcile memory assignments against terminal child records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from statistics import NormalDist, fmean
from typing import Any, Literal

ASSIGNMENT_EVENT = "MEMORY_ASSIGNMENT"
DELIVERY_EVENT = "MEMORY_DELIVERY"
MEMORY_TERMINAL_EVENTS = {"MEMORY_OUTCOME"}
DEFAULT_AA_TOLERANCE = 0.05
DEFAULT_PROBE_ITT_TOLERANCE = 0.05
DEFAULT_PROBE_DR_ALPHA = 0.05
DEFAULT_PROPENSITY_EPS = 1e-6
LOW_POWER_ARM_N = 30
DECISION_ID_KEYS = ("decision_id", "memory_assignment_decision_id")
OUTCOME_KEYS = ("outcome_value", "gain", "reward", "fitness_delta", "effect")
CENSOR_KEYS = ("censor_reason", "censored_reason", "cancellation_reason")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerminalRecord:
    decision_id: str
    kind: Literal["outcome", "invalid", "censor"]
    outcome: float | None
    source: str


@dataclass(frozen=True)
class Reconciliation:
    assignments: dict[str, dict[str, Any]]
    terminals: dict[str, tuple[TerminalRecord, ...]]
    orphans: tuple[str, ...]
    dupes: dict[str, tuple[TerminalRecord, ...]]
    duplicate_assignments: tuple[str, ...]

    @property
    def reconciled_ids(self) -> tuple[str, ...]:
        return tuple(
            decision_id
            for decision_id in self.assignments
            if len(self.terminals.get(decision_id, ())) == 1
        )

    @property
    def has_errors(self) -> bool:
        return bool(self.orphans or self.dupes or self.duplicate_assignments)


@dataclass(frozen=True)
class AASummary:
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    difference: float
    tolerance: float


@dataclass(frozen=True)
class ProbeITTSummary:
    n_treated: int
    n_control: int
    n_observational: int
    realized_treated_fraction: float
    mean_recorded_propensity: float
    difference: float
    tolerance: float


@dataclass(frozen=True)
class ProbeDRBaselineSummary:
    n_randomized: int
    n_with_outcome: int
    n_aipw: int
    dr_probe_effect: float | None


@dataclass(frozen=True)
class ProbeDRITTSummary:
    n: int
    tau_dr: float
    se_dr: float
    ci: tuple[float, float]
    z_score: float
    p_value: float
    tau_ips: float
    n_treated: int
    n_control: int
    realized_treated_fraction: float
    mean_propensity: float
    propensity_difference: float
    propensity_warning: bool
    n_ips_fallback: int
    clipped: int
    low_power: bool
    ips_within_dr_ci: bool

    def format(self) -> str:
        warnings: list[str] = []
        if self.low_power:
            warnings.append("LOW_POWER")
        if self.clipped:
            warnings.append("PROPENSITY_CLIPPED")
        if self.propensity_warning:
            warnings.append("PROPENSITY_MISMATCH")
        if not self.ips_within_dr_ci:
            warnings.append("IPS_OUTSIDE_DR_CI")
        warning_text = "" if not warnings else f" warnings={','.join(warnings)}"
        return (
            "Probe DR ITT "
            f"n=({self.n_treated},{self.n_control}) "
            f"tau_dr={self.tau_dr:.6g} se={self.se_dr:.6g} "
            f"ci=({self.ci[0]:.6g},{self.ci[1]:.6g}) "
            f"z={self.z_score:.6g} p={self.p_value:.6g} "
            f"tau_ips={self.tau_ips:.6g} "
            f"treated_fraction={self.realized_treated_fraction:.6g} "
            f"mean_propensity={self.mean_propensity:.6g} "
            f"propensity_diff={self.propensity_difference:.6g} "
            f"ips_fallback={self.n_ips_fallback} clipped={self.clipped}"
            f"{warning_text}"
        )


def read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                row.setdefault("_source_path", str(path))
                row.setdefault("_source_line", line_number)
                rows.append(row)
    return rows


def _mappings(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _mappings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _mappings(nested)


def _decision_id(row: Mapping[str, Any]) -> str:
    for mapping in _mappings(row):
        for key in DECISION_ID_KEYS:
            value = mapping.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _outcome(row: Mapping[str, Any]) -> float | None:
    mappings = tuple(_mappings(row))
    for key in OUTCOME_KEYS:
        for mapping in mappings:
            if (value := _number(mapping.get(key))) is not None:
                return value
    for mapping in mappings:
        metrics = mapping.get("metrics")
        if isinstance(metrics, Mapping):
            if (value := _number(metrics.get("fitness"))) is not None:
                return value
    return None


def _censor_reason(row: Mapping[str, Any]) -> str:
    for mapping in _mappings(row):
        for key in CENSOR_KEYS:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _is_invalid(row: Mapping[str, Any]) -> bool:
    for mapping in _mappings(row):
        if mapping.get("invalid") is True:
            return True
        validity = _number(mapping.get("is_valid"))
        if validity is not None and validity < 0.5:
            return True
        for key in ("status", "outcome", "terminal"):
            value = mapping.get(key)
            if isinstance(value, str) and value.lower() == "invalid":
                return True
    event = str(row.get("event", "")).upper()
    return "INVALID" in event


def _terminal(row: Mapping[str, Any]) -> TerminalRecord | None:
    if row.get("event") == ASSIGNMENT_EVENT:
        return None
    decision_id = _decision_id(row)
    if not decision_id:
        return None
    source = f"{row.get('_source_path', '<memory>')}:{row.get('_source_line', '?')}"
    outcome = _outcome(row)
    if _censor_reason(row):
        return TerminalRecord(decision_id, "censor", None, source)
    if _is_invalid(row):
        return TerminalRecord(decision_id, "invalid", outcome, source)
    event = str(row.get("event", "")).upper()
    if event.startswith("MEMORY_") and event not in MEMORY_TERMINAL_EVENTS:
        return None
    if any(token in event for token in ("OUTCOME", "EXPOSURE", "GAIN")):
        return TerminalRecord(decision_id, "outcome", outcome, source)
    state = str(row.get("state", "")).lower()
    if outcome is not None and state in {"done", "complete", "completed", "evaluated"}:
        return TerminalRecord(decision_id, "outcome", outcome, source)
    if outcome is not None and event not in {"MEMORY_READ_SELECTION", ASSIGNMENT_EVENT}:
        return TerminalRecord(decision_id, "outcome", outcome, source)
    return None


def reconcile_rows(rows: Iterable[Mapping[str, Any]]) -> Reconciliation:
    assignments: dict[str, dict[str, Any]] = {}
    deliveries: dict[str, dict[str, Any]] = {}
    assignment_counts: Counter[str] = Counter()
    terminals: defaultdict[str, list[TerminalRecord]] = defaultdict(list)
    for row in rows:
        if row.get("event") == DELIVERY_EVENT:
            payload = row.get("assignment")
            if isinstance(payload, Mapping):
                decision_id = payload.get("decision_id") or row.get("decision_id")
                if isinstance(decision_id, str) and decision_id:
                    deliveries[decision_id] = dict(payload)
            continue
        if row.get("event") == ASSIGNMENT_EVENT:
            payload = row.get("assignment", row.get("record"))
            if not isinstance(payload, Mapping):
                continue
            decision_id = payload.get("decision_id") or row.get("decision_id")
            if not isinstance(decision_id, str) or not decision_id:
                continue
            assignment_counts[decision_id] += 1
            assignments.setdefault(decision_id, dict(payload))
            continue
        terminal = _terminal(row)
        if terminal is not None:
            terminals[terminal.decision_id].append(terminal)

    for decision_id, delivery in deliveries.items():
        if decision_id in assignments:
            assignments[decision_id] = delivery

    relevant_terminals = {
        decision_id: tuple(terminals.get(decision_id, ()))
        for decision_id in assignments
    }
    orphans = tuple(
        decision_id
        for decision_id, records in relevant_terminals.items()
        if not records
    )
    dupes = {
        decision_id: records
        for decision_id, records in relevant_terminals.items()
        if len(records) > 1
    }
    duplicate_assignments = tuple(
        decision_id for decision_id, count in assignment_counts.items() if count > 1
    )
    return Reconciliation(
        assignments=assignments,
        terminals=relevant_terminals,
        orphans=orphans,
        dupes=dupes,
        duplicate_assignments=duplicate_assignments,
    )


def neutral_arm(decision_id: str) -> Literal["a", "b"]:
    digest = hashlib.sha256(decision_id.encode()).digest()
    return "a" if digest[0] & 1 == 0 else "b"


def assert_aa_balance(
    reconciliation: Reconciliation, *, tolerance: float = DEFAULT_AA_TOLERANCE
) -> AASummary:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    outcomes: dict[str, list[float]] = {"a": [], "b": []}
    for decision_id in reconciliation.reconciled_ids:
        (terminal,) = reconciliation.terminals[decision_id]
        if terminal.outcome is not None:
            outcomes[neutral_arm(decision_id)].append(terminal.outcome)
    if not outcomes["a"] or not outcomes["b"]:
        raise AssertionError(
            "A/A requires at least one numeric outcome in each hash arm"
        )
    mean_a = fmean(outcomes["a"])
    mean_b = fmean(outcomes["b"])
    difference = mean_a - mean_b
    if abs(difference) > tolerance:
        raise AssertionError(
            f"A/A difference {difference:.6g} exceeds tolerance {tolerance:.6g}"
        )
    return AASummary(
        n_a=len(outcomes["a"]),
        n_b=len(outcomes["b"]),
        mean_a=mean_a,
        mean_b=mean_b,
        difference=difference,
        tolerance=tolerance,
    )


def assert_probe_itt_calibration(
    reconciliation: Reconciliation,
    *,
    tolerance: float = DEFAULT_PROBE_ITT_TOLERANCE,
) -> ProbeITTSummary:
    """Validate the randomized-probe ledger before an ITT contrast is estimated."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    n_treated = 0
    n_control = 0
    n_observational = 0
    recorded_propensities: list[float] = []
    for decision_id, assignment in reconciliation.assignments.items():
        probe_arm = assignment.get("probe_arm", "none")
        if probe_arm == "none":
            n_observational += 1
            continue
        if probe_arm not in {"treated", "control"}:
            raise AssertionError(
                f"decision {decision_id} has invalid probe_arm={probe_arm!r}"
            )
        if assignment.get("randomized") is not True:
            raise AssertionError(
                f"decision {decision_id} has probe_arm={probe_arm!r} but is not randomized"
            )
        if assignment.get("propensity_kind") != "probe_bernoulli":
            raise AssertionError(
                f"decision {decision_id} has probe_arm={probe_arm!r} without "
                "probe_bernoulli propensity_kind"
            )
        propensities = assignment.get("propensities")
        if not isinstance(propensities, Mapping) or len(propensities) != 1:
            raise AssertionError(
                f"decision {decision_id} must record exactly one offered-card propensity"
            )
        propensity = _number(next(iter(propensities.values())))
        if propensity is None or not 0.0 <= propensity <= 1.0:
            raise AssertionError(
                f"decision {decision_id} has an invalid offered-card propensity"
            )
        recorded_propensities.append(propensity)
        if probe_arm == "treated":
            n_treated += 1
        else:
            n_control += 1

    randomized_count = n_treated + n_control
    if not randomized_count:
        raise AssertionError("probe ITT requires at least one treated/control decision")
    realized_treated_fraction = n_treated / randomized_count
    mean_recorded_propensity = fmean(recorded_propensities)
    difference = realized_treated_fraction - mean_recorded_propensity
    if abs(difference) > tolerance:
        raise AssertionError(
            f"probe ITT treated fraction {realized_treated_fraction:.6g} differs from "
            f"mean recorded propensity {mean_recorded_propensity:.6g} by "
            f"{difference:.6g}, exceeding tolerance {tolerance:.6g}"
        )
    return ProbeITTSummary(
        n_treated=n_treated,
        n_control=n_control,
        n_observational=n_observational,
        realized_treated_fraction=realized_treated_fraction,
        mean_recorded_propensity=mean_recorded_propensity,
        difference=difference,
        tolerance=tolerance,
    )


def assert_probe_dr_baselines(
    reconciliation: Reconciliation,
) -> ProbeDRBaselineSummary:
    """Validate probe q-hats and compute a minimal complete-case AIPW contrast."""
    n_randomized = 0
    n_with_outcome = 0
    aipw_scores: list[float] = []
    for decision_id, assignment in reconciliation.assignments.items():
        probe_arm = assignment.get("probe_arm", "none")
        if probe_arm == "none":
            continue
        if probe_arm not in {"treated", "control"}:
            raise AssertionError(
                f"decision {decision_id} has invalid probe_arm={probe_arm!r}"
            )
        n_randomized += 1
        propensities = assignment.get("propensities")
        if not isinstance(propensities, Mapping) or len(propensities) != 1:
            raise AssertionError(
                f"decision {decision_id} must record exactly one offered-card propensity"
            )
        offered_id = next(iter(propensities))
        propensity = _number(propensities[offered_id])
        if propensity is None or not 0.0 <= propensity <= 1.0:
            raise AssertionError(
                f"decision {decision_id} has an invalid offered-card propensity"
            )
        assigned_ids = assignment.get("assigned_ids", ())
        if not isinstance(assigned_ids, (list, tuple)) or not all(
            isinstance(card_id, str) for card_id in assigned_ids
        ):
            raise AssertionError(f"decision {decision_id} has invalid assigned_ids")
        recorded_ids = {*assigned_ids, offered_id}

        predictions: dict[str, Mapping[str, Any]] = {}
        for field_name in (
            "predicted_help",
            "predicted_gain",
            "predicted_no_card_gain",
        ):
            values = assignment.get(field_name)
            if not isinstance(values, Mapping):
                raise AssertionError(
                    f"decision {decision_id} is missing {field_name} q-hats"
                )
            if not set(values).issubset(recorded_ids):
                raise AssertionError(
                    f"decision {decision_id} has {field_name} keys outside the "
                    "assigned/offered slate"
                )
            predictions[field_name] = values

        predicted_help = predictions["predicted_help"]
        if set(predicted_help) != recorded_ids:
            raise AssertionError(
                f"decision {decision_id} predicted_help keys do not match "
                "assigned/offered cards"
            )
        for card_id, value in predicted_help.items():
            probability = _number(value)
            if probability is None or not 0.0 <= probability <= 1.0:
                raise AssertionError(
                    f"decision {decision_id} has invalid predicted_help for {card_id}"
                )
        for field_name in ("predicted_gain", "predicted_no_card_gain"):
            for card_id, value in predictions[field_name].items():
                if _number(value) is None:
                    raise AssertionError(
                        f"decision {decision_id} has non-finite {field_name} for {card_id}"
                    )

        terminal_records = reconciliation.terminals.get(decision_id, ())
        if len(terminal_records) != 1:
            continue
        (terminal,) = terminal_records
        if terminal.kind != "outcome" or terminal.outcome is None:
            continue
        n_with_outcome += 1
        q_hat_1 = _number(assignment.get("q_hat_treated"))
        q_hat_0 = _number(assignment.get("q_hat_control"))
        if q_hat_1 is None or q_hat_0 is None or not 0.0 < propensity < 1.0:
            continue
        treated = float(probe_arm == "treated")
        score = (
            q_hat_1
            - q_hat_0
            + treated / propensity * (terminal.outcome - q_hat_1)
            - (1.0 - treated) / (1.0 - propensity) * (terminal.outcome - q_hat_0)
        )
        aipw_scores.append(score)

    if not n_randomized:
        raise AssertionError("probe DR baseline requires a treated/control decision")
    return ProbeDRBaselineSummary(
        n_randomized=n_randomized,
        n_with_outcome=n_with_outcome,
        n_aipw=len(aipw_scores),
        dr_probe_effect=fmean(aipw_scores) if aipw_scores else None,
    )


def _offered_propensity(
    decision_id: str, assignment: Mapping[str, Any]
) -> tuple[str, float]:
    propensities = assignment.get("propensities")
    if not isinstance(propensities, Mapping) or len(propensities) != 1:
        raise AssertionError(
            f"decision {decision_id} must record exactly one offered-card propensity"
        )
    offered_id = next(iter(propensities))
    if not isinstance(offered_id, str) or not offered_id:
        raise AssertionError(f"decision {decision_id} has an invalid offered card id")
    propensity = _number(propensities[offered_id])
    if propensity is None:
        raise AssertionError(
            f"decision {decision_id} has a non-finite offered-card propensity"
        )
    return offered_id, propensity


def _clipped_propensity(propensity: float, eps: float) -> tuple[float, bool]:
    clipped = min(max(propensity, eps), 1.0 - eps)
    return clipped, clipped != propensity


def _prediction(
    assignment: Mapping[str, Any], field_name: str, offered_id: str
) -> float | None:
    values = assignment.get(field_name)
    if not isinstance(values, Mapping):
        return None
    return _number(values.get(offered_id))


def _mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    mean = fmean(values)
    if len(values) < 2:
        return mean, math.nan
    sum_squared = math.fsum((value - mean) ** 2 for value in values)
    variance = sum_squared / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def _normal_test(tau: float, se: float) -> tuple[float, float]:
    if not math.isfinite(se):
        return math.nan, math.nan
    if se == 0.0:
        if tau == 0.0:
            return 0.0, 1.0
        return math.copysign(math.inf, tau), 0.0
    z_score = tau / se
    return z_score, math.erfc(abs(z_score) / math.sqrt(2.0))


def estimate_probe_itt_dr(
    reconciliation: Reconciliation,
    *,
    eps: float = DEFAULT_PROPENSITY_EPS,
    alpha: float = DEFAULT_PROBE_DR_ALPHA,
    propensity_tolerance: float = DEFAULT_PROBE_ITT_TOLERANCE,
) -> ProbeDRITTSummary:
    """Estimate the randomized cold-probe ITT using DR/AIPW and IPS scores."""
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if propensity_tolerance < 0.0:
        raise ValueError("propensity_tolerance must be non-negative")

    dr_scores: list[float] = []
    ips_scores: list[float] = []
    recorded_propensities: list[float] = []
    clipped_decision_ids: list[str] = []
    n_treated = 0
    n_control = 0
    n_ips_fallback = 0

    for decision_id in reconciliation.reconciled_ids:
        assignment = reconciliation.assignments[decision_id]
        probe_arm = assignment.get("probe_arm", "none")
        if probe_arm == "none":
            continue
        if probe_arm not in {"treated", "control"}:
            raise AssertionError(
                f"decision {decision_id} has invalid probe_arm={probe_arm!r}"
            )
        if assignment.get("randomized") is not True:
            raise AssertionError(
                f"decision {decision_id} has probe_arm={probe_arm!r} but is not randomized"
            )
        if assignment.get("propensity_kind") != "probe_bernoulli":
            raise AssertionError(
                f"decision {decision_id} has probe_arm={probe_arm!r} without "
                "probe_bernoulli propensity_kind"
            )

        (terminal,) = reconciliation.terminals[decision_id]
        if terminal.kind != "outcome" or terminal.outcome is None:
            continue
        outcome = terminal.outcome
        offered_id, raw_propensity = _offered_propensity(decision_id, assignment)
        if not 0.0 <= raw_propensity <= 1.0:
            raise AssertionError(
                f"decision {decision_id} has an invalid offered-card propensity"
            )
        recorded_propensities.append(raw_propensity)
        propensity, was_clipped = _clipped_propensity(raw_propensity, eps)
        if was_clipped:
            clipped_decision_ids.append(decision_id)

        treated = probe_arm == "treated"
        treatment = float(treated)
        if treated:
            n_treated += 1
        else:
            n_control += 1
        ips_score = treatment / propensity * outcome - (
            (1.0 - treatment) / (1.0 - propensity) * outcome
        )
        ips_scores.append(ips_score)

        q_hat_1 = _number(assignment.get("q_hat_treated"))
        q_hat_0 = _number(assignment.get("q_hat_control"))
        if q_hat_1 is None or q_hat_0 is None:
            dr_scores.append(ips_score)
            n_ips_fallback += 1
            continue
        dr_scores.append(
            q_hat_1
            - q_hat_0
            + treatment / propensity * (outcome - q_hat_1)
            - (1.0 - treatment) / (1.0 - propensity) * (outcome - q_hat_0)
        )

    if not dr_scores:
        raise AssertionError(
            "probe DR ITT requires a reconciled treated/control outcome"
        )
    if not n_treated or not n_control:
        raise AssertionError(
            "probe DR ITT requires at least one treated and one control outcome"
        )

    tau_dr, se_dr = _mean_and_se(dr_scores)
    tau_ips = fmean(ips_scores)
    z_score, p_value = _normal_test(tau_dr, se_dr)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    if math.isfinite(se_dr):
        ci = (tau_dr - critical * se_dr, tau_dr + critical * se_dr)
    else:
        ci = (math.nan, math.nan)

    n = n_treated + n_control
    realized_treated_fraction = n_treated / n
    mean_propensity = fmean(recorded_propensities)
    propensity_difference = realized_treated_fraction - mean_propensity
    propensity_warning = abs(propensity_difference) > propensity_tolerance
    low_power = n_treated < LOW_POWER_ARM_N or n_control < LOW_POWER_ARM_N
    ips_within_dr_ci = ci[0] - 1e-12 <= tau_ips <= ci[1] + 1e-12

    if clipped_decision_ids:
        LOGGER.warning(
            "Probe DR ITT clipped %s propensity denominator(s): %s",
            len(clipped_decision_ids),
            _ids(clipped_decision_ids),
        )
    if low_power:
        LOGGER.warning(
            "Probe DR ITT low power: n_treated=%s n_control=%s",
            n_treated,
            n_control,
        )
    if propensity_warning:
        LOGGER.warning(
            "Probe DR ITT treated fraction %.6g differs from mean propensity %.6g "
            "by %.6g",
            realized_treated_fraction,
            mean_propensity,
            propensity_difference,
        )
    if not ips_within_dr_ci:
        LOGGER.warning(
            "Probe DR ITT IPS cross-check %.6g falls outside DR CI (%.6g, %.6g)",
            tau_ips,
            ci[0],
            ci[1],
        )

    return ProbeDRITTSummary(
        n=n,
        tau_dr=tau_dr,
        se_dr=se_dr,
        ci=ci,
        z_score=z_score,
        p_value=p_value,
        tau_ips=tau_ips,
        n_treated=n_treated,
        n_control=n_control,
        realized_treated_fraction=realized_treated_fraction,
        mean_propensity=mean_propensity,
        propensity_difference=propensity_difference,
        propensity_warning=propensity_warning,
        n_ips_fallback=n_ips_fallback,
        clipped=len(clipped_decision_ids),
        low_power=low_power,
        ips_within_dr_ci=ips_within_dr_ci,
    )


def _input_paths(target: Path, extras: Sequence[Path]) -> list[Path]:
    if target.is_file():
        memory_path = target
        run_dir = target.parent
    else:
        run_dir = target
        candidates = (
            run_dir / "memory" / "memory_events.jsonl",
            run_dir / "memory_events.jsonl",
        )
        memory_path = next(
            (path for path in candidates if path.is_file()), candidates[0]
        )
    if not memory_path.is_file():
        raise FileNotFoundError(f"memory events not found: {memory_path}")

    discovered: set[Path] = set()
    if run_dir.is_dir():
        for path in run_dir.rglob("*.jsonl"):
            name = path.name.lower()
            if path != memory_path and (
                "outcome" in name or "program" in name or name == "events.jsonl"
            ):
                discovered.add(path)
    paths = [memory_path, *sorted(discovered), *extras]
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in unique:
            if not resolved.is_file():
                raise FileNotFoundError(f"event file not found: {resolved}")
            unique.append(resolved)
    return unique


def _ids(values: Iterable[str], limit: int = 10) -> str:
    items = list(values)
    visible = ", ".join(items[:limit])
    return visible if len(items) <= limit else f"{visible}, … (+{len(items) - limit})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or memory_events.jsonl")
    parser.add_argument(
        "--events",
        type=Path,
        action="append",
        default=[],
        help="Additional program/outcome JSONL file; repeatable.",
    )
    parser.add_argument("--aa-tolerance", type=float, default=DEFAULT_AA_TOLERANCE)
    parser.add_argument(
        "--probe-itt-tolerance",
        type=float,
        default=DEFAULT_PROBE_ITT_TOLERANCE,
    )
    args = parser.parse_args(argv)

    paths = _input_paths(args.run, args.events)
    reconciliation = reconcile_rows(read_jsonl(paths))
    terminal_count = sum(len(records) for records in reconciliation.terminals.values())
    print(
        f"assignments={len(reconciliation.assignments)} terminals={terminal_count} "
        f"reconciled={len(reconciliation.reconciled_ids)} "
        f"orphans={len(reconciliation.orphans)} dupes={len(reconciliation.dupes)} "
        f"duplicate_assignments={len(reconciliation.duplicate_assignments)}"
    )
    if reconciliation.orphans:
        print(f"orphans: {_ids(reconciliation.orphans)}")
    if reconciliation.dupes:
        print(f"dupes: {_ids(reconciliation.dupes)}")
    if reconciliation.duplicate_assignments:
        print(f"duplicate assignments: {_ids(reconciliation.duplicate_assignments)}")

    aa_failed = False
    try:
        aa = assert_aa_balance(reconciliation, tolerance=args.aa_tolerance)
        print(
            f"A/A n=({aa.n_a},{aa.n_b}) means=({aa.mean_a:.6g},{aa.mean_b:.6g}) "
            f"diff={aa.difference:.6g} tolerance={aa.tolerance:.6g}"
        )
    except AssertionError as exc:
        aa_failed = True
        print(f"A/A failed: {exc}")

    probe_itt_failed = False
    probe_dr_itt_failed = False
    randomized_probe_count = sum(
        assignment.get("probe_arm") in {"treated", "control"}
        for assignment in reconciliation.assignments.values()
    )
    if randomized_probe_count:
        try:
            probe_itt = assert_probe_itt_calibration(
                reconciliation, tolerance=args.probe_itt_tolerance
            )
            print(
                "Probe ITT "
                f"n=({probe_itt.n_treated},{probe_itt.n_control}) "
                f"observational={probe_itt.n_observational} "
                f"treated_fraction={probe_itt.realized_treated_fraction:.6g} "
                f"mean_propensity={probe_itt.mean_recorded_propensity:.6g} "
                f"diff={probe_itt.difference:.6g} "
                f"tolerance={probe_itt.tolerance:.6g}"
            )
        except AssertionError as exc:
            probe_itt_failed = True
            print(f"Probe ITT failed: {exc}")
        try:
            print(estimate_probe_itt_dr(reconciliation).format())
        except AssertionError as exc:
            probe_dr_itt_failed = True
            print(f"Probe DR ITT failed: {exc}")
    else:
        print("Probe ITT randomized decisions=0 (calibration unavailable)")
    return (
        1
        if reconciliation.has_errors
        or aa_failed
        or probe_itt_failed
        or probe_dr_itt_failed
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
