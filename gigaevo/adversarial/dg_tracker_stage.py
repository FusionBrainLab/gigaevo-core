"""DGTrackerStage — records per-opponent fitness deltas into DGImprovementTracker.

Wiring:
  FetchOpponentIdsStage     -> opponent_ids        (Box[Any] of opponent program IDs)
  CallValidatorFunction     -> validation_result   (Box[(metrics, artifact)] tuple)

Role-aware recording (uses real program.id from the DAG-supplied Program):
  improver  (D run): pair = (d_id=program.id, g_id=opponent_id, delta)
  constructor (G run): pair = (d_id=opponent_id, g_id=program.id, delta)

Filtered: NaN deltas are skipped (no measurement). Non-positive deltas reach
DGTracker.record_batch (single pipelined write for ALL five key families:
dg_improvements, dg_best_pairs, dg_d_wins, dg_g_resisted, dg_metrics).
The metrics-dict schema is authored exclusively inside record_batch so this
stage never duplicates it.

Output: VoidOutput — this stage is a pure side-effect (Redis write).
Only real failure modes log (role mismatch, length mismatch, unexpected
shape) — per-pair and per-stage INFO chatter was removed (I-15) because
every mutation produced 2*n_opp lines that buried genuine errors in the
run log. The tracker itself is the source of truth; use DGTracker CLI /
monitoring tools to inspect recorded pairs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from gigaevo.programs.core_types import StageIO, VoidOutput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import Box

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker


class DGTrackerStageInputs(StageIO):
    """Inputs for DGTrackerStage."""

    opponent_ids: Box[Any]
    """Opponent program IDs (list[str]) from FetchOpponentIdsStage."""

    validation_result: Box[Any]
    """Box wrapping the (metrics, artifact) tuple from CallValidatorFunction.
    Artifact carries per_opp_pre/post/delta aligned with opponent_ids."""


class DGTrackerStage(Stage):
    """Records per-opponent fitness deltas into DGImprovementTracker.

    Side-effect-only stage. Runs after CallValidatorFunction (which produces
    the per-opponent artifact) and FetchOpponentIdsStage (which produces the
    aligned opponent IDs). Receives the real Program via Stage.execute, so
    program.id is the authoritative G or D identifier.
    """

    InputsModel = DGTrackerStageInputs
    OutputModel = VoidOutput
    cache_handler = NO_CACHE  # Always re-record; tracker uses ZADD GT for dedup.

    def __init__(
        self,
        *,
        dg_tracker: DGImprovementTracker,
        role: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if role not in ("constructor", "improver"):
            raise ValueError(f"role must be 'constructor' or 'improver', got {role!r}")
        self._tracker = dg_tracker
        self._role = role

    async def compute(self, program: Program) -> None:
        program_id = program.id
        params = cast(DGTrackerStageInputs, self.params)
        opponent_ids: list[str] = list(params.opponent_ids.data or [])
        validation_payload = params.validation_result.data
        if not isinstance(validation_payload, tuple) or len(validation_payload) != 2:
            logger.warning(
                "[DGTrackerStage {}] {} unexpected validation_result shape: {!r}",
                self._role,
                program_id[:8],
                type(validation_payload).__name__,
            )
            return None
        _metrics, artifact = validation_payload

        # F31: artifact role must match this stage's role. A mismatch means the
        # wrong evaluate.py was loaded for this population (D wired to G's
        # validator or vice versa) — silently swapped (D, G) recordings would
        # corrupt the tracker beyond recovery.
        artifact_role = artifact.get("role") if isinstance(artifact, dict) else None
        if artifact_role is not None and artifact_role != self._role:
            logger.error(
                "[DGTrackerStage {}] {} ROLE MISMATCH: artifact.role={!r} but "
                "stage role={!r}. Wiring bug — refusing to record. "
                "Check that the correct evaluate.py is wired for this population.",
                self._role,
                program_id[:8],
                artifact_role,
                self._role,
            )
            return None

        per_opp_delta: list[float] = (
            list(artifact.get("per_opp_delta", []) or [])
            if isinstance(artifact, dict)
            else []
        )
        n_opp = len(opponent_ids)

        if len(per_opp_delta) != n_opp:
            # F22: three legitimate mismatch shapes, only one is a real bug.
            #   (a) candidate-failed: artifact reports n_opponents=0 /
            #       is_valid=False / empty per_opp_delta while opponent_ids
            #       is non-empty. The user's evaluate.py correctly returns no
            #       deltas because the candidate itself never executed. Benign.
            #   (b) init-race: per_opp_delta populated but opponent_ids empty
            #       (gen-0 seed fallback when archive is empty). Benign.
            #   (c) real mismatch: both sides have data but lengths disagree →
            #       possible cache leak / TOCTOU between FetchOpponentIdsStage
            #       and CallValidatorFunction. This is the one we want to grep.
            candidate_failed = (
                isinstance(artifact, dict)
                and len(per_opp_delta) == 0
                and n_opp >= 1
                and (
                    artifact.get("is_valid") is False
                    or artifact.get("n_opponents") == 0
                )
            )
            init_race = len(per_opp_delta) > 0 and n_opp == 0
            if candidate_failed:
                logger.debug(
                    "[DGTrackerStage {}] {} candidate failed validation "
                    "(is_valid={}, n_opponents={}); no deltas recorded.",
                    self._role,
                    program_id[:8],
                    artifact.get("is_valid"),
                    artifact.get("n_opponents"),
                )
            elif init_race:
                logger.debug(
                    "[DGTrackerStage {}] {} gen-0 seed fallback: per_opp_delta "
                    "length {} while opponent_ids is empty (archive not yet "
                    "populated). Skipping.",
                    self._role,
                    program_id[:8],
                    len(per_opp_delta),
                )
            else:
                # Silent drop on length mismatch was the prior behavior. Now
                # logged at ERROR so it shows up in production grep and
                # surfaces cache-leak / TOCTOU between FetchOpponentIdsStage
                # and CallValidatorFunction. Skip semantics preserved.
                logger.error(
                    "[DGTrackerStage {}] {} per_opp_delta length {} != opponent_ids length {} "
                    "(artifact role={}); SKIP batch — possible cache leak between "
                    "FetchOpponentIdsStage and CallValidatorFunction.",
                    self._role,
                    program_id[:8],
                    len(per_opp_delta),
                    n_opp,
                    artifact_role,
                )
            return None

        pairs: list[tuple[str, str, float]] = []
        n_skip_nan = 0
        n_neg = 0

        for opponent_id, delta in zip(opponent_ids, per_opp_delta):
            if delta is None or (isinstance(delta, float) and math.isnan(delta)):
                n_skip_nan += 1
                continue
            d_val = float(delta)
            if d_val <= 0:
                n_neg += 1
            if self._role == "improver":
                pairs.append((program_id, opponent_id, d_val))
            else:
                pairs.append((opponent_id, program_id, d_val))

        if pairs:
            await self._tracker.record_batch(pairs)

        # I-15: per-pair and per-stage summary INFO logs removed — they
        # produced 2*n_opp lines per mutation and drowned out real errors.
        # Counters (n_skip_nan, n_neg, n_opp) are retained as locals only
        # to document the filter logic; they no longer emit log lines.
        _ = (n_skip_nan, n_neg, n_opp)
        return None
