"""Auto-compute and persist the probe-ITT effect on every memory run.

The offline estimator (``reconcile``) is pure and read-only. This reporter is
the plumbing that runs it automatically: the writer calls :meth:`refresh` at the
end of each increment (live sweep and final), so ``tau`` is written next to the
ledger in-progress, not only at completion. It reads the run's
``memory_events.jsonl``, reconciles it, estimates the DR-AIPW probe-ITT, writes
``ope_summary.json`` beside the ledger, and emits a ``MEMORY_OPE_SUMMARY`` event.

Emission must never affect evolution: every path here swallows its errors. A
ledger with no reconciled probe outcome degrades to ``insufficient_data`` (never
a raise), and a torn trailing line from a concurrent append is skipped.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from loguru import logger

from gigaevo.memory.events import MemoryOpeSummary, resolve_memory_event_path
from gigaevo.memory.ope.reconcile import (
    DEFAULT_PROBE_DR_ALPHA,
    DEFAULT_PROBE_ITT_TOLERANCE,
    DEFAULT_PROPENSITY_EPS,
    ProbeDRITTSummary,
    Reconciliation,
    estimate_probe_itt_dr,
    reconcile_rows,
)
from gigaevo.monitoring.emit import emit

OPE_SUMMARY_FILENAME = "ope_summary.json"


def _finite(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read one row per line, skipping a torn trailing line.

    The strict :func:`reconcile.read_jsonl` raises on malformed JSON — right for
    the offline CLI over a finished run, wrong here where the writer may still be
    appending: a mid-run read can catch a half-written final line.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(
                    "[Memory][OPE] skipping unparseable ledger line {}", line_number
                )
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


class MemoryOpeReporter:
    """Reconcile the run ledger and persist the probe-ITT effect.

    Args:
        checkpoint_dir: The run's memory checkpoint dir. The ledger is read from
            (and ``ope_summary.json`` written into) this directory.
        eps: Propensity-clip epsilon for the DR denominator.
        alpha: Two-sided significance level for the DR confidence interval.
        propensity_tolerance: Allowed gap between the realized treated fraction
            and the mean recorded propensity before a calibration warning fires.

    ``eps``/``alpha``/``propensity_tolerance`` are OPE-computation defaults
    (estimator hygiene), not memory-policy magnitudes; they default to the
    estimator's own constants.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        eps: float = DEFAULT_PROPENSITY_EPS,
        alpha: float = DEFAULT_PROBE_DR_ALPHA,
        propensity_tolerance: float = DEFAULT_PROBE_ITT_TOLERANCE,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._eps = eps
        self._alpha = alpha
        self._propensity_tolerance = propensity_tolerance

    def refresh(self) -> None:
        """Recompute and persist the summary. Never raises."""
        try:
            self._refresh()
        except Exception:
            logger.exception("[Memory][OPE] summary refresh failed; skipping")

    def _refresh(self) -> None:
        ledger = resolve_memory_event_path(self._checkpoint_dir)
        if ledger is None or not ledger.is_file():
            return
        reconciliation = reconcile_rows(_read_jsonl_tolerant(ledger))
        estimate = self._estimate(reconciliation)
        self._write_summary(reconciliation, estimate)
        emit(self._event(reconciliation, estimate))

    def _estimate(self, reconciliation: Reconciliation) -> ProbeDRITTSummary | None:
        try:
            return estimate_probe_itt_dr(
                reconciliation,
                eps=self._eps,
                alpha=self._alpha,
                propensity_tolerance=self._propensity_tolerance,
            )
        except AssertionError:
            # No reconciled treated/control outcome yet — expected early in a run.
            return None

    @staticmethod
    def _health(reconciliation: Reconciliation) -> dict[str, int]:
        return {
            "assignments": len(reconciliation.assignments),
            "reconciled": len(reconciliation.reconciled_ids),
            "orphans": len(reconciliation.orphans),
            "dupes": len(reconciliation.dupes),
            "duplicate_assignments": len(reconciliation.duplicate_assignments),
        }

    def _event(
        self, reconciliation: Reconciliation, estimate: ProbeDRITTSummary | None
    ) -> MemoryOpeSummary:
        health = self._health(reconciliation)
        if estimate is None:
            return MemoryOpeSummary(status="insufficient_data", **health)
        return MemoryOpeSummary(
            status="ok",
            n=estimate.n,
            n_treated=estimate.n_treated,
            n_control=estimate.n_control,
            tau_dr=_finite(estimate.tau_dr),
            se_dr=_finite(estimate.se_dr),
            ci_lo=_finite(estimate.ci[0]),
            ci_hi=_finite(estimate.ci[1]),
            z_score=_finite(estimate.z_score),
            p_value=_finite(estimate.p_value),
            tau_ips=_finite(estimate.tau_ips),
            low_power=estimate.low_power,
            propensity_warning=estimate.propensity_warning,
            **health,
        )

    def _write_summary(
        self, reconciliation: Reconciliation, estimate: ProbeDRITTSummary | None
    ) -> None:
        payload = {
            "status": "ok" if estimate is not None else "insufficient_data",
            "reconciliation": self._health(reconciliation),
            "probe_dr_itt": (
                None if estimate is None else self._summary_dict(estimate)
            ),
        }
        path = self._checkpoint_dir / OPE_SUMMARY_FILENAME
        text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)
        # Unique temp per refresh: off-loop refreshes overlap, so a fixed .tmp
        # name would let two writers corrupt each other's half-written summary.
        fd, tmp_name = tempfile.mkstemp(
            dir=self._checkpoint_dir, prefix=".ope_summary.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            # Atomic swap so a concurrent reader never sees a half-written summary.
            os.replace(tmp_name, path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _summary_dict(summary: ProbeDRITTSummary) -> dict[str, Any]:
        return {
            "n": summary.n,
            "n_treated": summary.n_treated,
            "n_control": summary.n_control,
            "tau_dr": summary.tau_dr,
            "se_dr": summary.se_dr,
            "ci_lo": summary.ci[0],
            "ci_hi": summary.ci[1],
            "z_score": summary.z_score,
            "p_value": summary.p_value,
            "tau_ips": summary.tau_ips,
            "realized_treated_fraction": summary.realized_treated_fraction,
            "mean_propensity": summary.mean_propensity,
            "propensity_difference": summary.propensity_difference,
            "propensity_warning": summary.propensity_warning,
            "n_ips_fallback": summary.n_ips_fallback,
            "clipped": summary.clipped,
            "low_power": summary.low_power,
            "ips_within_dr_ci": summary.ips_within_dr_ci,
        }
